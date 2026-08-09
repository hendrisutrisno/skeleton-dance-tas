import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
from tqdm import tqdm
import scipy.ndimage
from FACT.dataset import DataLoader, InferenceDataLoader
from FACT.fact import FACT
from FACT.util import setup_cfg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Fact_Trainer:
    def __init__(self, cfg, dataset, test_dataset, batch_size, class_weight):
        self.trainloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
        self.vaildloader = DataLoader(test_dataset, 1, shuffle=False)

        self.cfg = setup_cfg(cfg)
        self.model = FACT(self.cfg, dataset.input_dimension, dataset.nclasses, dataset.bg_class, class_weight)
        if self.cfg.Loss.nullw == -1:
            compute_null_weight(self.cfg, dataset)
    
    def train(self, save_dir):
        self.model.train()
        self.model.to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr = self.cfg.lr, weight_decay=self.cfg.weight_decay)
        best_acc =0.
        loss_list =[]
        v_losses = []

        for epoch in range(self.cfg.epoch):
            print(f"Epoch {epoch+1}/{self.cfg.epoch}")
            pbar = tqdm(total = len(self.trainloader), desc = "training")
            epoch_loss =0
            final_acc = 0

            for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(self.trainloader):
                seq_list = [s.to(device, dtype=torch.float32) for s in seq_list ]
                train_label_list = [ s.to(device) for s in train_label_list]
                optimizer.zero_grad()
                loss, video_saves, acc = self.model(seq_list, train_label_list, compute_loss=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                ### evalute
                epoch_loss += loss
                final_acc += acc
                pbar.update()

            v_loss = self.valid()
            pbar.close()
            epoch_loss = epoch_loss.detach().cpu().numpy()
            v_loss = v_loss.detach().cpu().numpy()
            loss_list.append(epoch_loss/len(self.trainloader))
            v_losses.append(v_loss / len(self.vaildloader))
            final_acc /= len(self.trainloader)
            logger.info("[epoch %d]: epoch loss = %f, acc = %f, v_loss = %f"% (epoch + 1, epoch_loss / len(self.trainloader), final_acc, v_losses[-1]))

            if best_acc < final_acc:
                best_acc = final_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_dir / "best_fact.model")
                torch.save(optimizer.state_dict(), save_dir / "best_fact.opt")
        logger.info("best epoch in %d" %(best_epoch))
        torch.save(self.model.state_dict(), save_dir / f"last_epoch-fact.model")
        torch.save(optimizer.state_dict(), save_dir / f"last_epoch-fact.opt")
        torch.cuda.empty_cache()
        return loss_list, v_losses
    
    def valid(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(self.vaildloader):
                    seq_list = [s.to(device, dtype=torch.float32) for s in seq_list ]
                    train_label_list = [ s.to(device) for s in train_label_list]
                    loss, video_saves, acc = self.model(seq_list, train_label_list, compute_loss=True)
                    epoch_loss += loss
        
        self.model.train()
        return epoch_loss

    def test(self, inference_dataset, model_dir, result_dir, actions_dict, sample_rate, window_size, last_model = False):
        self.model.eval()
        self.model.to(device)
        rev_actions_dict = {v: k for k, v in actions_dict.items()}
        result_dir.mkdir(parents=True, exist_ok=True)
        loader =  InferenceDataLoader(inference_dataset, 1, shuffle=False)

        with torch.no_grad():
            if last_model:
                ckpt = torch.load(model_dir / "last_epoch-fact.model", map_location="cpu")
            else:
                ckpt = torch.load(model_dir / "best_fact.model", map_location="cpu")
            state = ckpt.get('state_dict', ckpt)
            if 'frame_pe.pe' in state: del state['frame_pe.pe']
            self.model.load_state_dict(state, strict=False)
            rev_actions_dict = {v: k for k, v in actions_dict.items()}

            for i, (vname, batch_seq, _ , _) in enumerate(loader):
                seq_list = [s.to(device, dtype=torch.float32) for s in batch_seq ]
                video_saves = self.model.test(seq_list)
                assert len(video_saves) == 1
                for record in video_saves:
                    pred = record['prob']
                    pred = scipy.ndimage.gaussian_filter1d(np.array(pred), sigma = window_size * (30/sample_rate), axis =0)
                    pred  = pred.argmax(1)
                    label = []
                    for j in pred:
                        action_name = rev_actions_dict.get(j.item(), str(j.item()))
                        label.extend([action_name] * sample_rate)
                    with open(result_dir / f"{str(vname[0].stem)}.txt", "w") as f_ptr:
                        f_ptr.write(" ".join(label) + '\n')
        return
    

def compute_null_weight(cfg, dataset):
    """
    normalized the frequency of null class to 1/num_classes
    """
    average_trans_len = 10
    ntoken = cfg.FACT.ntoken
    num_null = ntoken - average_trans_len
    null_weight = ntoken / (num_null * dataset.nclasses)
    cfg.defrost()
    cfg.Loss.nullw = null_weight
    cfg.freeze()
    return cfg