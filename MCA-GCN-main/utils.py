import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import  roc_auc_score
from sklearn.preprocessing import StandardScaler

np.random.seed(123)

def stastic_indicators(output,labels):
    TP = ((output.max(1)[1] == 1) & (labels == 1)).sum()
    TN = ((output.max(1)[1] == 0) & (labels == 0)).sum()
    FN = ((output.max(1)[1] == 0) & (labels == 1)).sum()
    FP = ((output.max(1)[1] == 1) & (labels == 0)).sum()
    ACC = (TP + TN) / (TP + TN + FP + FN)
    SEN = TP / (TP + FN)
    P = TP / (TP + FP)
    SPE = TN / (FP + TN)
    BAC = (SEN + SPE) / 2
    F1 = ((2 * P) * SEN) / (P+SEN)
    MCC = ((TP * TN) - (TP * FN)) / (((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
    y_pred = output.max(1)[1]
    labels = labels.cpu().numpy()
    predictions = y_pred.cpu().numpy()
    AUC = roc_auc_score(labels, predictions)

    return ACC,SEN,SPE,BAC,F1,MCC,AUC


def get_adj_and_feat(mean):
    data_paths = '../data/data_file'
    data = np.load(data_paths)
    roi_features = data['fmri_data']
    gene_features = data['gene_codes']
    node_features = []
    roi_feats = []
    gene_feats = []
    adj_matrixs = []

    for i in tqdm(range(len(roi_features))):
        roi_feature = roi_features[i]
        gene_feature = gene_features[i]
        a = roi_feature
        b = gene_feature
        node_feature = np.concatenate([roi_feature, gene_feature])
        scaler = StandardScaler()
        roi_feature = scaler.fit_transform(roi_feature)
        gene_feature = scaler.fit_transform(gene_feature)
        node_feature = scaler.fit_transform(node_feature)
        pcc = np.corrcoef(roi_feature,gene_feature)
        adj = np.where(pcc > mean, 1, 0)
        row, col = np.diag_indices_from(adj)
        adj[row, col] = 1
        adj = normalize_adj(adj)
        node_features.append(node_feature)
        roi_feats.append(a)
        gene_feats.append(b)
        adj_matrixs.append(adj)

    return np.array(node_features),np.array(roi_feats),np.array(gene_feats),np.array(adj_matrixs)


def load_data(mean):
    node_features, roi_features, gene_features, adj_matrixs = get_adj_and_feat(mean)
    labels = torch.LongTensor([1] * 222 + [0] * 213)
    node_features = torch.from_numpy(node_features).float()
    roi_features = torch.from_numpy(roi_features).float()
    gene_features = torch.from_numpy(gene_features).float()
    adj_matrixs = torch.from_numpy(adj_matrixs).float()
    return adj_matrixs,roi_features,gene_features,node_features,labels

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

