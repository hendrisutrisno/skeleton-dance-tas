import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from pathlib import Path
from pre_process import batch_gen
from loguru import logger
from tqdm import tqdm
import scipy.ndimage
from MS-TCN.default import get_cfg_defaults

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.add("model.log", level="WARNING", mode="w", encoding="utf-8")

class Dilatedlayer(nn.Module):
    def __init__(self, diated_num, in_channel, out_channel):
        super(Dilatedlayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channel, out_channel, 3, padding=diated_num, dilation=diated_num)
        self.drop = nn.Dropout()
        self.output = nn.Conv1d(out_channel, out_channel, 1)

    def forward(self, input, mask):
        x = F.relu(self.conv_dilated(input))
        x = self.output(x)
        x = self.drop(x)
        return (x + input)

class Single_stage(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, channel_mask_rate):
        super(Single_stage, self).__init__()
        self.pointwise_conv = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(copy.deepcopy(Dilatedlayer(2 **i, num_f_maps, num_f_maps)) for i in range(num_layers))
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.drop = nn.Dropout(p=channel_mask_rate)
    
    def forward(self, input, mask):
        x = self.pointwise_conv(input)
        for layer in self.layers:
            x = layer(x, mask)
        out = self.conv_out(x) * mask[:, 0:1, :]
        return out

class stage_plus(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, channel_mask_rate):
        super(stage_plus, self).__init__()
        self.pointwise_conv = nn.Conv1d(dim, num_f_maps, 1)
        self.dilated_conv = nn.ModuleList(
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - 1 - i), dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        )
        self.dilated_conv2 = nn.ModuleList(
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i)
            for i in range(num_layers)
        )
        self.conv_fusion = nn.ModuleList(
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
            for _ in range(num_layers)
        )
        self.drop = nn.Dropout(p=channel_mask_rate)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, input, mask = None):
        x = self.pointwise_conv(input)
        for i in range(len(self.dilated_conv)):
            x_in = x
            x = self.conv_fusion[i](torch.cat([self.dilated_conv[i](x), self.dilated_conv2[i](x)], 1))
            x = F.relu(x)
            if i != len(self.dilated_conv) - 1:
                x = self.drop(x)
            if mask == None:
                x = (x_in + x)
            else:
                x = (x_in + x) * mask[:, 0:1, :]
        out = self.conv_out(x)
        return out

class Mttcn(nn.Module):
    def __init__(self, num_stage, num_layers, num_f_maps, dim, num_classes, channel_mask_rate):
        super(Mttcn, self).__init__()
        self.stage1 = stage_plus(num_layers, num_f_maps, dim, num_classes, channel_mask_rate)
        self.stages = nn.ModuleList(
            copy.deepcopy(Single_stage(num_layers, num_f_maps, num_classes, num_classes, channel_mask_rate)) for _ in range(num_stage - 1)
        )

    def forward(self, input, mask):
        """
        b, c, t = input.size()
        assert c == 36
        input = input.view(b, 2, t, 18, 1)
        """
        x = self.stage1(input, mask)

        out = x.unsqueeze(0)

        for s in self.stages:
            x = s(torch.softmax(x, dim=1) * mask[:, 0:1, :], mask)
            out = torch.cat((out, x.unsqueeze(0)), dim=0)
        return out
    
def setup_cfg(cfg_file=[], set_cfgs=None, default = None, logdir="log/"):
    """
    update default cfg according to cmd line input
    and automatic generate experiment name
    """
    cfg = get_cfg_defaults()
    L = len(set_cfgs) if set_cfgs else 0
    new_set_cfgs = []
    for i in range(L//2):
        k = set_cfgs[i*2]
        v = set_cfgs[i*2+1]

        if not isinstance(k, list):
                k = [k]
        for k_ in k:
            tgt = _get_var(cfg, k_.split('.'))
            v_ = int2float_check(v, tgt)
            new_set_cfgs.extend([k_, v_])



class MSTCN_Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, channel_mask_rate, loss_func = None):
        self.cfg = setup_cfg(cfg)
        self.model = Mttcn(num_blocks, num_layers, num_f_maps, dim, num_classes, channel_mask_rate)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss = loss_func
        self.num_classes = num_classes
        self.label_file = None
    
    def train(self, save_dir, num_epochs, batch_size, video_list, valid_list, label_file, learning_rate, device, sample_rate, thr = 0.5):
        self.model.train()
        self.model.to(device)

        try:
            label_file = Path(label_file)
            if not label_file.exists():
                raise FileNotFoundError(f"{label_file} does not exist")
        except FileNotFoundError as e:
            logger.error(f"folder not found: {e}")
            return
        self.label_file = label_file
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        best_acc =0.
        loss_list =[]
        v_losses = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            pbar = tqdm(total = len(video_list), desc = "training")
            epoch_loss =0
            correct =0
            b_c = 0
            total =0
            index =0

            while index < len(video_list):
                batch_input, batch_target, mask, _ = batch_gen(video_list[index:index+batch_size], label_file, sample_rate)
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)
                n = len(predictions)
                loss = 0.

                for p in predictions:
                    loss += self.loss(p, batch_target, mask, sim_index = batch_input) / n

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(predictions[-1].detach(), 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
                index += batch_size
                pbar.update(1)
            
            v_loss = self.valid(valid_list, sample_rate)
            pbar.close()
            loss_list.append(epoch_loss/len(video_list))
            v_losses.append(v_loss/len(valid_list))
            accuracy = float(correct) / total if total >0 else 0
            logger.info("[epoch %d]: epoch loss = %f, acc = %f, v_loss = %f"% (epoch + 1, epoch_loss / len(video_list), accuracy, v_losses[-1]))

            if best_acc < accuracy:
                best_acc = accuracy
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_dir / "best_mstcn.model")
                torch.save(optimizer.state_dict(), save_dir / "best_mstcn.opt")
        logger.info("best epoch in %d" %(best_epoch))
        torch.save(self.model.state_dict(), save_dir / f"last_epoch-mstcn{str(epoch + 1)}.model")
        torch.save(optimizer.state_dict(), save_dir / f"last_epoch-mstcn{str(epoch + 1)}.opt")
        return v_losses, loss_list
    
    def valid(self, test_data, sample_rate):
        self.model.eval()
        epoch_loss = 0
        index = 0
        with torch.no_grad():
            while index < len(test_data):
                    batch_input, batch_target, mask, b_target = batch_gen(test_data[index:index+1], self.label_file, sample_rate)
                    batch_input = batch_input.to(device)
                    batch_target = batch_target.to(device)
                    b_target =b_target.to(device)
                    mask = mask.to(device)
                    predictions = self.model(batch_input, mask)
                    n = len(predictions)
                    loss = 0.

                    for p in predictions:
                        loss += self.loss(p, batch_target, mask, sim_index = batch_input) / n

                    epoch_loss += loss.item()
                    index += 1
        self.model.train()
        return epoch_loss

    def test(self, model_dir, result_dir, test_data, actions_dict, sample_rate, window_size):
        self.model.eval()
        label = []
        result_dir.mkdir(parents=True, exist_ok=True)
        r_result_dir = result_dir / "refinement"
        r_result_dir.mkdir(parents=True, exist_ok=True)
        c_result_dir = result_dir/ "class_result"
        c_result_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir / "best_mstcn.model"))
            
            for test in test_data:
                #features = np.load(test)[:, :28].T
                features = np.load(test).T
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)
                mask = torch.ones(input_x.size(), device=device)
                predictions = self.model(input_x, mask)

                result = predictions.data[-1].cpu().detach()

                result = torch.softmax(result, dim =1)
                r_result = scipy.ndimage.gaussian_filter1d(np.array(result), sigma = window_size * (30//sample_rate), axis =2)
                #r_result = scipy.ndimage.uniform_filter1d(np.array(result), size = window_size *(30//sample_rate), axis =2)
                classifiction(np.max(np.array(r_result), axis = 1).squeeze(0), test.stem, c_result_dir)
                predict = np.argmax(np.array(result), axis=1).squeeze(0)
                r_result = np.argmax(np.array(r_result), axis =1).squeeze(0)

                
                rev_actions_dict = {v: k for k, v in actions_dict.items()}
                label = []
                for j in predict:
                    action_name = rev_actions_dict.get(j.item(), str(j.item()))
                    label.extend([action_name] * sample_rate)
                r_label = []
                for j in r_result:
                    action_name = rev_actions_dict.get(j.item(), str(j.item()))
                    r_label.extend([action_name] * sample_rate)

                with open(result_dir / f"{str(test.stem)}.txt", "w") as f_ptr:
                    f_ptr.write(" ".join(label) + '\n')
                with open(r_result_dir / f"{str(test.stem)}.txt", "w") as f_ptr:
                    f_ptr.write(" ".join(r_label) + '\n')

        return label, rev_actions_dict
