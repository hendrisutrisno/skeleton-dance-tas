import torch
import torch.nn as nn
from FACT.module import PositionalEncoding, InputBlock, UpdateBlock, UpdateBlockTDU, logit2prob
from yacs.config import CfgNode
import random
from FACT.loss import MatchCriterion

def update_from(cfg: CfgNode, ref: CfgNode, inplace=False) -> CfgNode:
    if not inplace:
        cfg = cfg.clone()
    cfg.defrost()

    for k in cfg:
        if k not in ref:
            continue

        if cfg[k] is None and ref[k] is not None:
            cfg[k] = ref[k]
    return cfg


class FACT(nn.Module):
    def __init__(self, cfg, in_dim, n_classes, bg_class, class_weight):
        super(FACT, self).__init__()
        self.cfg = cfg
        self.num_classes = n_classes
        self.mcriterion = MatchCriterion(self.cfg, n_classes, bg_ids=bg_class, class_weight=class_weight)
        base_cfg = self.cfg.Bi
        self.frame_pe = PositionalEncoding(base_cfg.hid_dim, max_len=10000, empty=(not cfg.FACT.fpos))
        self.channel_masking_dropout = nn.Dropout1d(p=cfg.FACT.cmr)

        if not cfg.FACT.trans : # when video transcript is not available at training and inference
            self.action_query = nn.Parameter(torch.randn([cfg.FACT.ntoken, 1, base_cfg.a_dim]))
        else: # when video transcript is available
            self.action_pe = PositionalEncoding(base_cfg.a_dim, max_len=1000)
            self.action_embed = nn.Embedding(n_classes, base_cfg.a_dim)

        # block configuration
        block_list = []
        for i, t in enumerate(cfg.FACT.block):
            if t == 'i':
                block = InputBlock(cfg, in_dim, n_classes)
            elif t == 'u':
                update_from(cfg.Bu, base_cfg, inplace=True)
                base_cfg = cfg.Bu
                block = UpdateBlock(cfg, n_classes)
            elif t == 'U':
                update_from(cfg.BU, base_cfg, inplace=True)
                base_cfg = cfg.BU
                block = UpdateBlockTDU(cfg, n_classes)

            block_list.append(block)
        self.block_list = nn.ModuleList(block_list)
    
    def _forward_one_video(self, seq, transcript=None):
        # prepare frame feature
        frame_feature = seq
        frame_pe = self.frame_pe(seq)
        if self.cfg.FACT.cmr:  #(T,1,H)
            frame_feature = frame_feature.permute([1, 2, 0]) #(1, H, T)
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.permute([2, 0, 1]) #(T, 1, H)

        if self.cfg.TM.use and self.training:
            frame_feature = time_mask(frame_feature, 
                        self.cfg.TM.t, self.cfg.TM.m, self.cfg.TM.p, 
                        replace_with_zero=True)

        # prepare action feature
        if not self.cfg.FACT.trans:
            action_pe = self.action_query # M, B(=1), H
            action_feature = torch.zeros_like(action_pe)
        else:
            action_pe = self.action_pe(transcript)
            action_feature = self.action_embed(transcript).unsqueeze(1)

            action_feature = action_feature + action_pe
            action_pe = torch.zeros_like(action_pe)

        # forward
        # frame_feature: T, B(=1), H
        # action_feature: M, B(=1), H
        block_output = []
        for i, block in enumerate(self.block_list):
            frame_feature, action_feature = block(frame_feature, action_feature, frame_pe, action_pe)
            block_output.append([frame_feature, action_feature])
        return block_output

    def _loss_one_video(self, label):
        mcriterion = self.mcriterion
        mcriterion.set_label(label)

        block = self.block_list[-1]
        cprob = logit2prob(block.action_clogit, dim=-1)
        match = mcriterion.match(cprob, block.a2f_attn)

        ######## per block loss
        loss_list = []
        for block in self.block_list:
            loss = block.compute_loss(mcriterion, match)
            loss_list.append(loss)

        self.loss_list = loss_list
        final_loss = sum(loss_list) / len(loss_list)
        return final_loss

    def forward(self, seq_list, label_list, compute_loss=False):
        save_list = []
        final_loss= []
        acc_list = []

        for i, (seq, label) in enumerate(zip(seq_list, label_list)):
            seq = seq.unsqueeze(1)  #(T,1,H)
            trans = torch_class_label_to_segment_label(label)[0]
            self._forward_one_video(seq, trans)

            prob, pred = self.block_list[-1].eval(trans)
            save_data = {
                'pred': pred.detach().cpu().numpy(),
                'prob': prob.detach().cpu().numpy(),
            }       
            save_list.append(save_data)

            acc = (pred == label).float().mean().item()
            acc_list.append(acc)

            if compute_loss:
                loss = self._loss_one_video(label)
                final_loss.append(loss)
                save_data['loss'] = {'loss': loss.item()}

        if compute_loss:
            final_loss = sum(final_loss) / len(final_loss)
            final_acc = sum(acc_list) / len(acc_list)
            return final_loss, save_list, final_acc
        else:
            return save_list

    def test(self, seq_list):
        save_list = []
        self.eval()
        with torch.no_grad():
            for seq in seq_list:
                seq = seq.unsqueeze(1)
                self._forward_one_video(seq, transcript=None)

                block = self.block_list[-1]
                try:
                    prob, pred = block.eval(None)
                except TypeError:
                    prob, pred = block.eval()

                save_data = {
                    "pred": pred.detach().cpu().numpy(),
                    "prob": prob.detach().cpu().numpy(),
                }

                save_list.append(save_data)
        return save_list

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)

def torch_class_label_to_segment_label(label):
    segment_label = torch.zeros_like(label)
    current = label[0]
    transcript = [label[0]]
    aid = 0
    for i, l in enumerate(label):
        if l == current:
            pass
        else:
            current = l
            aid += 1
            transcript.append(l)
        segment_label[i] = aid
    transcript = torch.LongTensor(transcript).to(label.device)
    return transcript, segment_label

def time_mask(feature, T, num_masks, p, replace_with_zero=False, clone=False):
    """
    T: max drop length - cfg.t
    num_masks: num drop - cfg.m
    p: max drop ratio - cfg.p

    feature: T, B, H
    """
    if clone:
        feature = feature.clone()

    len_spectro = feature.shape[0]
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t = min( int(p*len_spectro), t )
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): 
            return feature

        if (replace_with_zero): 
            feature[t_zero:t_zero+t] = 0
        else: 
            feature[t_zero:t_zero+t] = feature.mean()
    return feature