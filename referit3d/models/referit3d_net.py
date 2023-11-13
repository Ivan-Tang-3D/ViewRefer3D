# new_ones_384
import torch
import argparse
from torch import nn, Tensor
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional
import numpy as np
from . import DGCNN
from .utils import get_siamese_features, my_get_siamese_features
from ..in_out.vocabulary import Vocabulary
import math, copy

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers import BertTokenizer, BertModel, BertConfig
from referit3d.models import MLP
import time
from timm.models.layers import DropPath, trunc_normal_


def _translate(point_set, bxyz):
        """
        input:point_set:B,N,P,3 
        bxyz&oxyz:B,N,3
        """
        # unpack
        coords = point_set[:, :, :, :3]

        # translation factors
        factor = torch.rand(3) - 0.5
        factor = factor.cuda()

        # dump
        coords += factor
        point_set[:, :, :, :3] = coords
        bxyz[:, :, :3] += factor

        return point_set, bxyz

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoderLayer_(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout #, batch_first=True
        )
        #NOTE: inter
        self.inter_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout #, batch_first=True
        )
        ######
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout #, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self, tgt, memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        B = int(tgt.shape[1] / 4) #NOTE
        tgt2 = self.norm1(tgt)
        
        ###
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        ######
        #NOTE: inter
        tgt2 = tgt2.transpose(0, 1).reshape(B, 4, -1, 768).permute(0, 2, 1, 3).reshape(-1, 4, 768) #NOTE
        tgt2, _ = self.inter_attn(
            tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt2 = tgt2.reshape(-1, B, 4, 768).reshape(-1, 4 * B, 768)#NOTE
        tgt = tgt + self.dropout3(tgt2)
        tgt2 = self.norm4(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)
        return tgt #, self_attn_matrices, cross_attn_matrices

class ReferIt3DNet_transformer(nn.Module):

    def __init__(self,
                 args,
                 n_obj_classes,
                 class_name_tokens,
                 ignore_index):

        super().__init__()
        print("Model: Bestmax or Bestmax Concate")

        self.bert_pretrain_path = args.bert_pretrain_path

        self.view_number = args.view_number
        self.rotate_number = args.rotate_number

        self.label_lang_sup = args.label_lang_sup
        self.aggregate_type = args.aggregate_type

        self.encoder_layer_num = args.encoder_layer_num
        self.decoder_layer_num = args.decoder_layer_num
        self.decoder_nhead_num = args.decoder_nhead_num

        self.object_dim = args.object_latent_dim
        self.inner_dim = args.inner_dim
        
        self.dropout_rate = args.dropout_rate
        self.lang_cls_alpha = args.lang_cls_alpha
        self.obj_cls_alpha = args.obj_cls_alpha
  
        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                        sa_n_samples=[[32], [32], [None]],
                                        sa_radii=[[0.2], [0.4], [None]],
                                        sa_mlps=[[[3, 64, 64, 128]],
                                                [[128, 128, 128, 256]],
                                                [[256, 256, self.object_dim, self.object_dim]]])

        self.language_encoder = BertModel.from_pretrained(self.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:self.encoder_layer_num]

        #####
        decoder_layer = TransformerDecoderLayer_(d_model=self.inner_dim, 
            nhead=self.decoder_nhead_num, dim_feedforward=2048, activation="gelu")
        self.refer_encoder = _get_clones(decoder_layer, self.decoder_layer_num)
        
        # Classifier heads
        self.language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
                                        nn.ReLU(), nn.Dropout(self.dropout_rate), 
                                        nn.Linear(self.inner_dim, n_obj_classes))

        self.object_language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
                                                nn.ReLU(), nn.Dropout(self.dropout_rate), 
                                                nn.Linear(self.inner_dim, 1))

        if not self.label_lang_sup:
            self.obj_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, n_obj_classes], dropout_rate=self.dropout_rate)

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(4, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.class_name_tokens = class_name_tokens

        self.logit_loss = nn.CrossEntropyLoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # weight_gen
        self.feat_trans = nn.Sequential(nn.Linear(self.inner_dim, 384), nn.ReLU())
        self.view_relu = nn.ReLU()
        #self.view_speciality = nn.Parameter(torch.ones(self.view_number, 384), requires_grad=True)
        self.view_speciality = nn.Parameter(torch.randn(self.view_number, 384), requires_grad=True)
        self.view_lin = nn.Sequential(nn.Linear(384, 384), nn.ReLU())
        self.feat_lin = nn.Sequential(nn.Linear(384, 384), nn.ReLU())
        trunc_normal_(self.view_speciality, std=0.02)
        # text prompt
        self.view_text_trans = nn.Sequential(nn.Linear(384, 768), nn.ReLU())
        self.text_speciality = nn.Parameter(torch.randn(self.view_number, 384), requires_grad=True)
        self.prompt_attn = nn.MultiheadAttention(embed_dim=384, num_heads=8, dropout=0.15)
        self.gamma = nn.Parameter(torch.ones(4, 768) * 1e-3)#NOTE
        trunc_normal_(self.text_speciality, std=0.02)

        #self.cross_view = CrossView()

    @torch.no_grad()
    def aug_input(self, input_points, box_infos, tokens, tokens2, tokens3, tokens4):
        input_points = input_points.float().to(self.device)
        box_infos = box_infos.float().to(self.device)
        xyz = input_points[:, :, :, :3]
        bxyz = box_infos[:,:,:3] # B,N,3
        B,N,P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor([i*2.0*np.pi/self.rotate_number for i in range(self.rotate_number)]).to(self.device)
        view_theta_arr = torch.Tensor([i*2.0*np.pi/self.view_number for i in range(self.view_number)]).to(self.device)
        
        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            theta = rotate_theta_arr[torch.randint(0,self.rotate_number,(B,))]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.Tensor([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0]]).to(self.device)[None].repeat(B,1,1)
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[:, :, :, :3] = torch.matmul(xyz.reshape(B,N*P,3), rotate_matrix).reshape(B,N,P,3)
            bxyz = torch.matmul(bxyz.reshape(B,N,3), rotate_matrix).reshape(B,N,3)

            input_points, bxyz = _translate(input_points, bxyz)
        
        # multi-view
        bsize = box_infos[:,:,-1:]
        boxs=[]
        for theta in view_theta_arr:
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                        [math.sin(theta), math.cos(theta),  0.0],
                                        [0.0,           0.0,            1.0]]).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B*N, 3),rotate_matrix).reshape(B,N,3)
            boxs.append(torch.cat([rxyz,bsize],dim=-1))
        boxs=torch.stack(boxs,dim=1)
        
        #GPT
        new_tokens = []
        for i in range(len(tokens)):
            new_tokens.append(tokens[i])
            new_tokens.append(tokens3[i])
            new_tokens.append(tokens2[i])
            new_tokens.append(tokens4[i])

        tokenizer = BertTokenizer.from_pretrained("./data/bert")
        lang_tokens = tokenizer(new_tokens, return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        return input_points, boxs, lang_tokens#, tokens_late

    def compute_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, AUX_LOGITS=None):
        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])
        total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss + self.lang_cls_alpha * lang_clf_loss
        return total_loss


    def forward(self, batch: dict, epoch=None):
        # batch['class_labels']: GT class of each obj
        # batch['target_class']：GT class of target obj
        # batch['target_pos']: GT id

        self.device = self.obj_feature_mapping[0].weight.device
        ## rotation augmentation and multi_view generation
        obj_points, boxs,lang_tokens = self.aug_input(batch['objects'], batch['box_info'], batch['tokens'],batch['tokens2'],batch['tokens3'],batch['tokens4'])
        
        B,N,P = obj_points.shape[:3]

        ## obj_encoding
        objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack)
        
        ## obj_encoding
        obj_feats = self.obj_feature_mapping(objects_features)
        box_infos = self.box_feature_mapping(boxs)
        obj_infos = obj_feats[:, None].repeat(1, self.view_number, 1, 1) + box_infos

        # <LOSS>: obj_cls
        if self.label_lang_sup:
            label_lang_infos = self.language_encoder(**self.class_name_tokens)[0][:,0]
            CLASS_LOGITS = torch.matmul(obj_feats.reshape(B*N,-1), label_lang_infos.permute(1,0)).reshape(B,N,-1)        
        else:
            CLASS_LOGITS = self.obj_clf(obj_feats.reshape(B*N,-1)).reshape(B,N,-1)

        ## language_encoding
        lang_infos = self.language_encoder(**lang_tokens)[0]
        lang_infos = lang_infos.reshape(B, 4, -1, self.inner_dim)#NOTE
    
        text_view_speciality = self.view_speciality
        lang_prompt = self.view_text_trans(self.prompt_attn(query=self.text_speciality.unsqueeze(1), key=text_view_speciality.unsqueeze(1), value=text_view_speciality.unsqueeze(1))[0].squeeze())
                    
        ########
        # <LOSS>: lang_cls
        clf_lang_input = lang_infos[:,0,0,:].clone()
        LANG_LOGITS = self.language_clf(clf_lang_input)
        ## multi-modal_fusion
        cat_infos = obj_infos.reshape(B*self.view_number, -1, self.inner_dim)
        lang_infos[:,:,0,:]=lang_infos[:,:,0,:].clone()+(self.gamma*lang_prompt).unsqueeze(0)
        mem_infos = lang_infos.reshape(B*self.view_number, -1, self.inner_dim)

        cat_infos = cat_infos.transpose(0, 1)
     
        for layer in self.refer_encoder:
            cat_infos = layer(cat_infos, mem_infos.transpose(0, 1))
        
        out_feats = cat_infos.transpose(0, 1).reshape(B, self.view_number, -1, self.inner_dim)
        ############

        ## view_aggregation
        refer_feat = out_feats  #([24, 4, 52, 768])

        ## weight gen
        weight_feat = self.feat_trans(refer_feat)
        weight_feat = torch.max(weight_feat, dim=2)[0].unsqueeze(2)
        #weight_feat = (torch.max(weight_feat, dim=2)[0] + weight_feat.mean(dim=2)).unsqueeze(2)
        #weight_feat = weight_feat.sum(dim=2).unsqueeze(2)
        #weight_feat = weight_feat.mean(dim=2).unsqueeze(2)
        weight_feat = self.feat_lin(weight_feat)
        view_speciality = self.view_lin(self.view_speciality)
        logits_weight = torch.matmul(weight_feat, view_speciality.unsqueeze(-1).unsqueeze(0).repeat(refer_feat.shape[0], 1, 1, 1))
        
        logits_weight = self.view_relu(logits_weight)
        logits_weight = logits_weight.squeeze(-1)


        # aggregrate logits
        VIEW_LOGITS = self.object_language_clf(refer_feat).squeeze(-1)  #torch.Size([24, 4, 52])
        LOGITS = torch.matmul(VIEW_LOGITS.permute(0, 2, 1), logits_weight).squeeze(-1)

        ################
        LOSS = self.compute_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOGITS)

        return LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS