import pickle

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import dgl
from dgl.nn.pytorch import GATConv
import scipy.sparse as sp


import numpy as np
import dgl.function as fn
from dgl import DropEdge
from hgmae.models.han import HAN
from hgmae.utils import create_norm



class Model(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, args, g_dgl, meta_path_patterns, user_key, item_key, user_num, item_num, in_size, out_size, dropout, dev,
                 num_metapath_u, focused_feature_dim_u,
                 num_metapath_i, focused_feature_dim_i,
                 feats_u, mps_u, feats_i, mps_i):
    # def __init__(self, args, g_dgl, meta_path_patterns, user_key, item_key, in_size, out_size, num_heads, dropout, dev,
    #              num_metapath_u, focused_feature_dim_u,
    #              num_metapath_i, focused_feature_dim_i):
        super(Model, self).__init__()
        # self.__dict__.update(cf.get_model_conf())
        self.g_dgl = g_dgl
        self.userkey = user_key
        self.itemkey = item_key
        self.unum = user_num
        self.inum = item_num
        self.dropout = dropout
        self.dev = dev
        self.meta_path_patterns = meta_path_patterns
        self.temperature = args.tem


        self.mask_model_u = PreModel(args, num_metapath_u, focused_feature_dim_u)
        self.mask_model_i = PreModel(args, num_metapath_i, focused_feature_dim_i)
        self.feats_u = feats_u
        self.feats_i = feats_i
        self.mps_u = mps_u
        self.mps_i = mps_i



        self.RelationalAGG = RelationalAGG(g_dgl, in_size, out_size, self.dropout)
        self.initializer = nn.init.xavier_uniform_
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g_dgl.num_nodes(ntype), in_size))) for ntype in
            g_dgl.ntypes
        })

        # self.gcn = GCN_layer()
        # self.ui = sp.load_npz("./data/yelp/s_pre_adj_mat_1.npz")


        self.user_layer1 = nn.Linear(out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(out_size, out_size, bias=True)
        self.user_layer2 = nn.Linear(2 * out_size, out_size, bias=True)
        self.item_layer2 = nn.Linear(2 * out_size, out_size, bias=True)

    def forward(self, user_idx, item_idx, neg_item_idx):#, users, pos, neg

        h1 = self.RelationalAGG(self.g_dgl, self.feature_dict)
        user_emb = h1[self.userkey]
        item_emb = h1[self.itemkey]

        user_emb = self.user_layer1(user_emb)
        item_emb = self.item_layer1(item_emb)
        user_emb = self.user_layer2(torch.cat((user_emb, self.feature_dict[self.userkey]), 1))
        item_emb = self.item_layer2(torch.cat((item_emb, self.feature_dict[self.itemkey]), 1))


        # ui_embeddings = torch.cat((user_emb, item_emb), 0)
        # ui_embeddings = torch.cat([self.feature_dict[self.userkey], self.feature_dict[self.itemkey]], 0)
        # for i in range(3):
        #     if i == 0:
        #         uiembedding = self.gcn(ui_embeddings, self.ui)
        #     else:
        #         uiembedding = self.gcn(uiembedding, self.ui)
        # user_emb, item_emb = \
        #     torch.split(uiembedding, [self.unum, self.inum])


        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        neg_feat = item_emb[neg_item_idx]

        return user_feat, item_feat, neg_feat


    def total_loss(self, users, pos, neg, epoch):
        # user_emb, item_emb, neg_emb, sub_user_emb, sub_item_emb = self.forward(users, pos, neg)
        user_emb, item_emb, neg_emb = self.forward(users, pos, neg)

        # temperature = 0.1
        temperature = self.temperature

        sub_user = self.mask_model_u(self.feats_u, self.mps_u, epoch)
        sub_item = self.mask_model_i(self.feats_i, self.mps_i, epoch)
        # sub_user = self.mask_model_u(feats_u, mps_u, epoch)
        # sub_item = self.mask_model_i(feats_i, mps_i, epoch)
        sub_user = torch.nn.functional.normalize(sub_user)
        sub_item = torch.nn.functional.normalize(sub_item)

        sub_user_emb = sub_user[users]
        sub_item_emb = sub_item[pos]

        # user_emb = 0.8 * user_emb + 0.2 * sub_user_emb
        # item_emb = 0.8 * item_emb + 0.2 * sub_item_emb

        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                              item_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(user_emb, item_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        pos_scores = torch.mul(self.feature_dict['user'][users], self.feature_dict['business'][pos])
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(self.feature_dict['user'][users], self.feature_dict['business'][neg])
        neg_scores = torch.sum(neg_scores, dim=1)
        loss2 = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))


        pos_score_user = (user_emb * sub_user_emb).sum(dim=-1)
        pos_score_user = torch.exp(pos_score_user / temperature)

        ttl_score_user = torch.matmul(user_emb, sub_user_emb.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / temperature).sum(dim=1)
        loss_user = - torch.log(pos_score_user / ttl_score_user + 10e-6)
        # loss_user = torch.nan_to_num(loss_user, nan=100)
        loss_user = torch.mean(loss_user)

        pos_score_item = (item_emb * sub_item_emb).sum(dim=-1)
        pos_score_item = torch.exp(pos_score_item / temperature)
        ttl_score_item = torch.matmul(item_emb, sub_item_emb.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / temperature).sum(dim=1)
        loss_item = - torch.log(pos_score_item / ttl_score_item + 10e-6)
        # loss_item=torch.nan_to_num(loss_item, nan=100)
        loss_item = torch.mean(loss_item)

        infonce_loss = torch.sum(loss_user + loss_item)

        return loss, reg_loss, infonce_loss#, loss2

    def getUsersRating(self, user_idx):#users
        # # items = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        # users_emb, all_items = self.computer(users, None, None)
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.dev)
        users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        # users_emb, all_items, _, sub_user, sub_item = self.forward(user_idx, item_idx, None)
        rating = torch.matmul(users_emb, all_items.t())
        return rating


class RelationalAGG(nn.Module):
    def __init__(self, g, in_size, out_size, dropout):
        super(RelationalAGG, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # Transform weights for different types of edges
        self.W_T = nn.ModuleDict({
            name: nn.Linear(in_size, out_size, bias=False) for name in g.etypes
        })

        # Attention weights for different types of edges
        self.W_A = nn.ModuleDict({
            name: nn.Linear(out_size, 1, bias=False) for name in g.etypes
        })

        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.nodes[dsttype].data['h'] = feat_dict[dsttype]  # nodes' original feature
            g.nodes[srctype].data['h'] = feat_dict[srctype]
            g.nodes[srctype].data['t_h'] = self.W_T[etype](feat_dict[srctype])  # src nodes' transformed feature

            # compute the attention numerator (exp)
            # Update the features of the specified edges by the provided function.
            g.apply_edges(fn.u_mul_v('t_h', 'h', 'x'), etype=etype)
            g.edges[etype].data['x'] = torch.exp(self.W_A[etype](g.edges[etype].data['x']))

            # first update to compute the attention denominator (\sum exp)
            funcs[etype] = (fn.copy_e('x', 'm'), fn.sum('m', 'att'))
        g.multi_update_all(funcs, 'sum')

        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.apply_edges(fn.e_div_v('x', 'att', 'att'),
                          etype=etype)  # compute attention weights (numerator/denominator)
            funcs[etype] = (fn.u_mul_e('h', 'att', 'm'), fn.sum('m', 'h'))  # \sum(h0*att) -> h1
        # second update to obtain h1
        g.multi_update_all(funcs, 'sum')

        # apply activation, layernorm, and dropout
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.dropout(
                self.layernorm(F.relu_(g.nodes[ntype].data['h'])))  # apply activation, layernorm, and dropout

        return feat_dict



class PreModel(nn.Module):
    def __init__(
            self,
            args,
            num_metapath: int,
            focused_feature_dim: int):
        super(PreModel, self).__init__()

        self.num_metapath = num_metapath
        self.focused_feature_dim = focused_feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.activation = args.activation
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm
        # self.feat_mask_rate = args.feat_mask_rate
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        # self.loss_fn = args.loss_fn
        self.enc_dec_input_dim = self.focused_feature_dim
        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        # num head: encoder
        if self.encoder_type in ("gat", "dotgat", "han"):
            enc_num_hidden = self.hidden_dim // self.num_heads
            enc_nhead = self.num_heads
        else:
            enc_num_hidden = self.hidden_dim
            enc_nhead = 1

        # num head: decoder
        if self.decoder_type in ("gat", "dotgat", "han"):
            dec_num_hidden = self.hidden_dim // self.num_out_heads
            dec_nhead = self.num_out_heads
        else:
            dec_num_hidden = self.hidden_dim
            dec_nhead = 1
        dec_in_dim = self.hidden_dim

        # encoder
        self.encoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.encoder_type,
            enc_dec="encoding",
            in_dim=self.enc_dec_input_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=self.num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
        )

        # decoder
        self.decoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=128,
            num_layers=1,
            nhead=enc_nhead,
            nhead_out=dec_nhead,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
            concat_out=True,
        )#self.enc_dec_input_dim

        self.__cache_gs = None
        self.mp_edge_mask_rate = args.mp_edge_mask_rate
        self.use_mp2vec_feat_pred = args.use_mp2vec_feat_pred
        self.use_mp_edge_recon = args.use_mp_edge_recon
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.mask_rate = args.mask_rate


    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):#
        try:
            return float(input_mask_rate)
        except ValueError:
            if "~" in input_mask_rate:  # 0.6~0.8 Uniform sample
                mask_rate = [float(i) for i in input_mask_rate.split('~')]
                assert len(mask_rate) == 2
                if get_min:
                    return mask_rate[0]
                else:
                    return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
            elif "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                mask_rate = [float(i) for i in input_mask_rate.split(',')]
                assert len(mask_rate) == 3
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate

                    # cur_mask_rate = self.mask_rate
                    # return cur_mask_rate
            else:
                raise NotImplementedError


    def mask_mp_edge_reconstruction(self, feat, mps, epoch):  #
        masked_gs = self.mps_to_gs(mps)
        cur_mp_edge_mask_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)  #
        drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        for i in range(len(masked_gs)):
            masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i])  # we need to add self loop
        enc_rep, _ = self.encoder(masked_gs, feat, return_hidden=False)
        rep = self.encoder_to_decoder_edge_recon(enc_rep)

        # if self.decoder_type == "mlp":
        #     feat_recon = self.decoder(rep)
        # else:
        #     feat_recon, att_mp = self.decoder(masked_gs, rep)
        feat_recon, att_mp = self.decoder(masked_gs, rep)

        # gs_recon = torch.mm(feat_recon, feat_recon.T)

        # loss = None
        # for i in range(len(mps)):
        #     if loss is None:
        #         loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
        #         # loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon_only_masked_places_list[i], mps_only_masked_places_list[i])  # loss only on masked places
        #     else:
        #         loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
        #         # loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon_only_masked_places_list[i], mps_only_masked_places_list[i])
        return feat_recon

    def forward(self, feats, mps, epoch):#**kwargs
        # prepare for mp2vec feat pred

        if self.use_mp2vec_feat_pred:
            # mp2vec_feat = feats[0][:, self.focused_feature_dim:]
            origin_feat = feats[0][:, :self.focused_feature_dim]
        else:
            origin_feat = feats[0]

        # mp based edge reconstruction
        if self.use_mp_edge_recon:
            edge_recon_loss = self.mask_mp_edge_reconstruction(origin_feat, mps, epoch)#kwargs.get("epoch", None)
            loss = edge_recon_loss
            # loss = self.mp_edge_recon_loss_weight * edge_recon_loss

        return loss

    def mps_to_gs(self, mps):
        if self.__cache_gs is None:
            gs = []
            for mp in mps:
                indices = mp._indices()
                cur_graph = dgl.graph((indices[0], indices[1]), num_nodes=self.enc_dec_input_dim)#
                gs.append(cur_graph)
            return gs
        else:
            return self.__cache_gs

def setup_module(m_type, num_metapath, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual,
                 norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "han":
        mod = HAN(
            num_metapath=num_metapath,
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod




