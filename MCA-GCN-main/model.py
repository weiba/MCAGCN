import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Positional_encoding import LearnablePositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MCA_GCN(nn.Module):
    def __init__(self, feat_dim, hidden_size, num_classes, nhead, dropout):
        super(MCA_GCN, self).__init__()
        self.LPE = LearnablePositionalEncoding(90,116)
        self.multiAttn1 = nn.MultiheadAttention(embed_dim=feat_dim,
                                               num_heads=nhead,
                                               dropout=dropout,
                                               batch_first=True)
        self.multiAttn2 = nn.MultiheadAttention(embed_dim=feat_dim,
                                                num_heads=nhead,
                                                dropout=dropout,
                                                batch_first=True)
        self.gcn1 = GraphConvolution(feat_dim, hidden_size, dropout)
        self.layerNorm1 = nn.LayerNorm(feat_dim)
        # prediction
        self.fc1 = nn.Linear(158*hidden_size, 158*hidden_size//2)
        self.fc2 = nn.Linear(158*hidden_size//2, 158*hidden_size//4)
        self.fc3 = nn.Linear(158*hidden_size // 4, num_classes)

    def forward(self, r_feat, g_feat, adj): # inputs: (batch, seq, feature)
        rx = r_feat
        rx = self.LPE(rx)
        gx = g_feat
        x = torch.cat((gx,rx),dim=1)
        attn_output1, seq_attn1 = self.multiAttn1(rx, gx, gx)
        attn_output2, seq_attn2 = self.multiAttn2(gx, rx, rx)
        attn_output = torch.cat((attn_output1,attn_output2),dim=1)

        node_scores_matrix1 = torch.matmul(seq_attn1, gx)

        node_scores_matrix2 = torch.matmul(seq_attn2, rx)

        node_scores1 = torch.sum(node_scores_matrix1, dim=2)
        node_scores2 = torch.sum(node_scores_matrix2, dim=2)

        normalized_node_scores1 = torch.softmax(node_scores1, dim=-1)
        normalized_node_scores2 = torch.softmax(node_scores2, dim=-1)
        node_scores = torch.cat((normalized_node_scores1, normalized_node_scores2), dim=1)

        normalized_node_scores = torch.mean(node_scores,dim=0)

        x = x + self.layerNorm1(attn_output)

        x = self.gcn1(x,adj)

        gc2_rl = x.reshape(-1, 158 * x.shape[2])
        fc1 = F.relu(self.fc1(gc2_rl))
        fc2 = F.relu(self.fc2(fc1))
        prediction_scores = self.fc3(fc2)

        return prediction_scores,normalized_node_scores


class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(1.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)






