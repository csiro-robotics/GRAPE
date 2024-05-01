import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Sequential, SAGEConv
import torch.nn.init as init


class simpleMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, act = 'relu', dropout=0.0):
        super().__init__() 
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU() if act =='relu' else nn.Identity(),
            nn.Dropout(dropout) if dropout>0.0 else nn.Identity(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU() if act =='relu' else nn.Identity(),
            nn.Dropout(dropout) if dropout>0.0 else nn.Identity(),
            nn.Linear(hidden_size2, output_size)
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x
    
class LogisticRegression(nn.Module):
    def __init__(self, dim_u, dim_v, p=0.0):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Sequential(
            nn.Dropout(p) if p>0.0 else nn.Identity(),
            nn.Linear(dim_u+dim_v+1, 1),           
            nn.Sigmoid()
        ) 

    def encode(self, xu, xv, edge_index, edge_index1, d, edge_attr=None):
        eu, ev = edge_index
        x = torch.cat((xu[eu], xv[ev-444], d), dim=1)
        x = self.lr(x)
        
        return x
    

class mlp_model(nn.Module):
    def __init__(self, dim_u, dim_v, h_dim1, h_dim2, h_dim3, out_dim, dropout):
        super().__init__()
        self.mlp_u = simpleMLP(dim_u, h_dim1, h_dim2, h_dim3)
        self.mlp_v = simpleMLP(dim_v, h_dim1, h_dim2, h_dim3)
        self.mlp_d = EdgeMLP(1, h_dim2, h_dim3)
        self.dropout = nn.Dropout(dropout) 
        
        self.layers = nn.Sequential(
            nn.Linear(h_dim3, h_dim3),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout>0.0 else nn.Identity(),
            nn.Linear(h_dim3, 1),  # Output layer with 1 unit for binary classification
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def encode(self, xu, xv, edge_index, edge_index1, d, edge_attr=None):
        eu, ev = edge_index
        x_u = self.mlp_u(xu)
        x_v = self.mlp_v(xv)
        x_d = self.mlp_d(d)
        x = torch.vstack((x_u, x_v))
        x_eu, x_ev = x[eu], x[ev]
        x = x_ev + x_eu + x_d
        x = self.layers(x)
        return x
    

class GCN_MLP_combined_mod(nn.Module):
    def __init__(self, dim_u, dim_v, h_dim1, h_dim2, h_dim3, out_dim, dropout):
        super().__init__()
        self.mlp_u = simpleMLP(dim_u, h_dim1, h_dim2, h_dim3)
        self.mlp_v = simpleMLP(dim_v, h_dim1, h_dim2, h_dim3)
        self.mlp_d = EdgeMLP(1, h_dim2, h_dim3)
        self.dropout = nn.Dropout(dropout) 

        self.gnn = Sequential('x, edge_index', [
            (SAGEConv(h_dim3, h_dim3, aggr='mean'), 'x, edge_index-> x'),
            nn.BatchNorm1d(h_dim3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            
            
            (SAGEConv(h_dim3, h_dim3, aggr='mean'), 'x, edge_index-> x'),
            nn.BatchNorm1d(h_dim3),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.ReLU(),
            nn.Dropout(dropout),
        ])
        
        self.layers = nn.Sequential(
            nn.Linear(h_dim3, h_dim3),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout>0.0 else nn.Identity(),
            nn.Linear(h_dim3, 1),  # Output layer with 1 unit for binary classification
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def encode(self, xu, xv, edge_index, edge_index1, d, edge_attr=None):
        eu, ev = edge_index
        x_u = self.mlp_u(xu)
        x_v = self.mlp_v(xv)
        x_d = self.mlp_d(d)
        x = torch.vstack((x_u, x_v))
        x = self.gnn(x, edge_index1)
        x_eu, x_ev = x[eu], x[ev]
        x = x_ev + x_eu +x_d
        x = self.layers(x)
        return x
    
    
class EdgeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(EdgeMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, edge_attr):
        return self.mlp(edge_attr)

class GCNE_MLP_combined(nn.Module):
    def __init__(self, dim_u, dim_v, h_dim1, h_dim2, h_dim3, out_dim, dropout):
        super().__init__()
        self.mlp_u = simpleMLP(dim_u, h_dim1, h_dim2, h_dim3)
        self.mlp_v = simpleMLP(dim_v, h_dim1, h_dim2, h_dim3)
        self.mlp_e = EdgeMLP(1, h_dim3, out_dim)
        self.dropout = nn.Dropout(dropout) 
        
        self.gnn = Sequential('x, edge_index, edge_attr', [
            (GATConv(h_dim3, h_dim3, heads=2, edge_dim=out_dim, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            (GATConv(h_dim3*2, out_dim, heads = 1, edge_dim=out_dim), 'x, edge_index, edge_attr -> x'),
        ])


    def encode(self, xu, xv, edge_index, edge_attr=None):
        x_u = self.mlp_u(xu)
        x_v = self.mlp_v(xv)
        x = torch.cat((x_u, x_v), dim=0)  # Concatenating along the nodes dimension
        # print(edge_attr.shape, x_u.shape, x_v.shape, x.shape)       
        edge_attr = self.mlp_e(edge_attr.view(-1,1))  # Reshaping edge_attr and applying the MLP
        # print(edge_attr.shape)
        x = self.gnn(x, edge_index, edge_attr)
        return x
    
    def decode(self, z, edge_index):
        return (z[edge_index[:,0]] * z[edge_index[:,1]]).sum(dim=-1) 
    

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
