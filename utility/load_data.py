import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import HeteroData

data_folder = "data/"

# sp.load_npz("movielens/uiu.npz")


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def process_data_in_pyg(neigs):
    d = defaultdict(dict)
    metapaths = []
    for mp_i, nei1 in enumerate(neigs):
        dst_array_concat = np.concatenate(nei1)
        src_array_concat = []
        for src_id, dst_array in enumerate(nei1):
            src_array_concat.extend([src_id] * len(dst_array))
        src_array_concat = np.array(src_array_concat)
        src_name = f"target"
        dst_name = f"dst_{mp_i}"
        relation = f"relation_{mp_i}"
        d[(src_name, relation + "-->", dst_name)]["edge_index"] = th.LongTensor(np.vstack([src_array_concat, dst_array_concat]))
        metapaths.append((src_name, relation + "-->", dst_name))
        d[(dst_name, "<--" + relation, src_name)]["edge_index"] = th.LongTensor(np.vstack([dst_array_concat, src_array_concat]))
        metapaths.append((dst_name, "<--" + relation, src_name))
    g = HeteroData(d)
    return g, metapaths

def load_movielens_user(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = data_folder + "movielens/"
    feat_u = sp.eye(type_num[0])
    feat_i = sp.eye(type_num[1])
    # feat_g = sp.eye(type_num[2])
    uiu = sp.load_npz(path + "uiu.npz")
    uigiu = sp.load_npz(path + "uigiu.npz")
    feat_u = th.FloatTensor(preprocess_features(feat_u))
    feat_i = th.FloatTensor(preprocess_features(feat_i))
    # feat_g = th.FloatTensor(preprocess_features(feat_g))
    uiu = sparse_mx_to_torch_sparse_tensor(normalize_adj(uiu))
    uigiu = sparse_mx_to_torch_sparse_tensor(normalize_adj(uigiu))
    return [feat_u, feat_i], [uiu, uigiu]#, label, pos, train, val, test, [nei_a, nei_s][nei_i],


def load_movielens_item(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = data_folder + "movielens/"
    feat_u = sp.eye(type_num[0])
    feat_i = sp.eye(type_num[1])
    feat_g = sp.eye(type_num[2])
    iui = sp.load_npz(path + "iui.npz")
    igi = sp.load_npz(path + "igi.npz")
    feat_u = th.FloatTensor(preprocess_features(feat_u))
    feat_i = th.FloatTensor(preprocess_features(feat_i))
    feat_g = th.FloatTensor(preprocess_features(feat_g))
    iui = sparse_mx_to_torch_sparse_tensor(normalize_adj(iui))
    igi = sparse_mx_to_torch_sparse_tensor(normalize_adj(igi))
    return [feat_i, feat_u, feat_g], [iui, igi]#, label, pos, train, val, test, [nei_a, nei_s][nei_u, nei_g],

def load_amazon_user(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = data_folder + "amazon/"
    feat_u = sp.eye(type_num[0])
    feat_i = sp.eye(type_num[1])
    # feat_c = sp.eye(type_num[2])

    uiu = sp.load_npz(path + "uiu.npz")
    uiciu = sp.load_npz(path + "uiciu.npz")

    feat_u = th.FloatTensor(preprocess_features(feat_u))
    feat_i = th.FloatTensor(preprocess_features(feat_i))
    # feat_c = th.FloatTensor(preprocess_features(feat_c))
    uiu = sparse_mx_to_torch_sparse_tensor(normalize_adj(uiu))
    uiciu = sparse_mx_to_torch_sparse_tensor(normalize_adj(uiciu))
    return [feat_u, feat_i], [uiu, uiciu]#, label, pos, train, val, test, [nei_a, nei_s][nei_i],


def load_amazon_item(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = data_folder + "amazon/"
    feat_u = sp.eye(type_num[0])
    feat_i = sp.eye(type_num[1])
    feat_v = sp.eye(type_num[2])
    feat_c = sp.eye(type_num[3])
    feat_b = sp.eye(type_num[4])
    iui = sp.load_npz(path + "iui.npz")
    ibi = sp.load_npz(path + "ibi.npz")
    ici = sp.load_npz(path + "ici.npz")

    feat_u = th.FloatTensor(preprocess_features(feat_u))
    feat_i = th.FloatTensor(preprocess_features(feat_i))
    feat_c = th.FloatTensor(preprocess_features(feat_c))
    feat_b = th.FloatTensor(preprocess_features(feat_b))
    iui = sparse_mx_to_torch_sparse_tensor(normalize_adj(iui))
    ibi = sparse_mx_to_torch_sparse_tensor(normalize_adj(ibi))
    ici = sparse_mx_to_torch_sparse_tensor(normalize_adj(ici))
    return [feat_i, feat_u, feat_c, feat_b], [iui, ici, ibi]#, label, pos, train, val, test, [nei_a, nei_s][nei_u, nei_c],

def load_yelp_user(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = data_folder + "yelp/"
    feat_u = sp.eye(type_num[0])
    feat_b = sp.eye(type_num[1])
    # feat_o = sp.eye(type_num[2])

    ubu = sp.load_npz(path + "ubu.npz")
    # uou = sp.load_npz(path + "uou.npz")

    feat_u = th.FloatTensor(preprocess_features(feat_u))
    feat_b = th.FloatTensor(preprocess_features(feat_b))
    # feat_o = th.FloatTensor(preprocess_features(feat_o))
    ubu = sparse_mx_to_torch_sparse_tensor(normalize_adj(ubu))
    # uou = sparse_mx_to_torch_sparse_tensor(normalize_adj(uou))
    return [feat_u, feat_b], [ubu]#, feat_o, nei_o, uou, label, pos, train, val, test, [nei_a, nei_s][nei_b],


def load_yelp_item(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = data_folder + "yelp/"
    feat_u = sp.eye(type_num[0])
    feat_b = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_i = sp.eye(type_num[3])
    bub = sp.load_npz(path + "bub.npz")
    bab = sp.load_npz(path + "bab.npz")
    bib = sp.load_npz(path + "bib.npz")
    feat_u = th.FloatTensor(preprocess_features(feat_u))
    feat_b = th.FloatTensor(preprocess_features(feat_b))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_i = th.FloatTensor(preprocess_features(feat_i))
    bub = sparse_mx_to_torch_sparse_tensor(normalize_adj(bub))
    bab = sparse_mx_to_torch_sparse_tensor(normalize_adj(bab))
    bib = sparse_mx_to_torch_sparse_tensor(normalize_adj(bib))
    return [feat_b, feat_u, feat_a, feat_i], [bub, bab, bib]

def load_lastfm_user(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = data_folder + "lastfm/"
    feat_u = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    uau = sp.load_npz(path + "uau.npz")
    feat_u = th.FloatTensor(preprocess_features(feat_u))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    uau = sparse_mx_to_torch_sparse_tensor(normalize_adj(uau))
    return [feat_u, feat_a], [uau]


def load_lastfm_item(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = data_folder + "lastfm/"
    feat_u = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_t = sp.eye(type_num[2])
    aua = sp.load_npz(path + "aua.npz")
    ata = sp.load_npz(path + "ata.npz")
    feat_u = th.FloatTensor(preprocess_features(feat_u))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_t = th.FloatTensor(preprocess_features(feat_t))
    aua = sparse_mx_to_torch_sparse_tensor(normalize_adj(aua))
    ata = sparse_mx_to_torch_sparse_tensor(normalize_adj(ata))
    return [feat_a, feat_u, feat_t], [aua, ata]


def load_data(dataset, type, ratio, type_num):
    if dataset == "movielens" and type == 'user':
        data = load_movielens_user(ratio, type_num)
    elif dataset == "movielens" and type == 'item':
        data = load_movielens_item(ratio, type_num)
    elif dataset == "amazon" and type == 'user':
        data = load_amazon_user(ratio, type_num)
    elif dataset == "amazon" and type == 'item':
        data = load_amazon_item(ratio, type_num)
    elif dataset == "yelp" and type == 'user':
        data = load_yelp_user(ratio, type_num)
    elif dataset == "yelp" and type == 'item':
        data = load_yelp_item(ratio, type_num)
    elif dataset == "lastfm" and type == 'user':
        data = load_lastfm_user(ratio, type_num)
    elif dataset == "lastfm" and type == 'item':
        data = load_lastfm_item(ratio, type_num)
    # g, metapaths = process_data_in_pyg(data[0])
    return data#, g, metapaths

