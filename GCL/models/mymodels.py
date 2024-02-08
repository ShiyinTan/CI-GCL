import torch
from torch_geometric.nn import SAGEConv, to_hetero, GATConv, GCNConv, TransformerConv, global_mean_pool, global_add_pool, MessagePassing
from torch import optim, Tensor
import torch.nn as nn
import torch.nn.functional as F
import GCL.augmentors.learnable_augs as LeA
from torch.nn import CosineSimilarity, PairwiseDistance, Dropout, MultiheadAttention, Sigmoid
import numpy as np
from torch.nn.functional import softmax
from typing import Callable, Optional, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.utils import to_dense_adj, to_undirected, is_undirected

### Convs

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class GINEConv(MessagePassing):
    def __init__(self, input_channels, hidden_channels, edge_dim: Optional[int] = None, 
                 eps: float = 0., train_eps: bool = False, 
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINEConv, self).__init__(**kwargs)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        self.nn = nn.Sequential(
                    nn.Linear(input_channels, hidden_channels), 
                    nn.ReLU(), 
                    nn.Linear(hidden_channels, hidden_channels))
        
        if edge_dim is not None:
            self.lin = nn.Linear(edge_dim, input_channels)
        else:
            self.lin = None
        
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_weight: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        if edge_weight is None:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu((x_j + edge_attr) * edge_weight.view(-1, 1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)



class WGINConv(MessagePassing):
	def __init__(self, input_channels, hidden_channels: int, eps: float = 0., train_eps: bool = False,
                 edge_dim=0, 
				 **kwargs):
		kwargs.setdefault('aggr', 'add')
		super(WGINConv, self).__init__(**kwargs)
		self.nn = nn.Sequential(
                    nn.Linear(input_channels, hidden_channels), 
                    nn.ReLU(), 
                    nn.Linear(hidden_channels, hidden_channels))
		self.initial_eps = eps
		if train_eps:
			self.eps = torch.nn.Parameter(torch.Tensor([eps]))
		else:
			self.register_buffer('eps', torch.Tensor([eps]))
		self.reset_parameters()

	def reset_parameters(self):
		reset(self.nn)
		self.eps.data.fill_(self.initial_eps)

	def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight = None, edge_attr=None, 
				size: Size = None) -> Tensor:
		""""""
		if isinstance(x, Tensor):
			x: OptPairTensor = (x, x)
		# propagate_type: (x: OptPairTensor)
		out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
		x_r = x[1]
		if x_r is not None:
			out += (1 + self.eps) * x_r
		return self.nn(out)

	def message(self, x_j: Tensor, edge_weight) -> Tensor:
		return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)

	def __repr__(self):
	    return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GraphNodeEncoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, edge_dim=0, num_layers=2, dropout=0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bn_norms = torch.nn.ModuleList()
        if edge_dim==0:
            self.GCONV_CLASS = WGINConv
        else:
            self.GCONV_CLASS = GINEConv
        for i in range(num_layers):
            if i==0:
                self.convs.append(self.GCONV_CLASS(input_channels, hidden_channels, edge_dim=edge_dim))
            else:
                self.convs.append(self.GCONV_CLASS(hidden_channels, hidden_channels, edge_dim=edge_dim))
            self.bn_norms.append(nn.BatchNorm1d(hidden_channels))
        self.project_dim = hidden_channels * num_layers
        self.project = torch.nn.Sequential(
                    nn.Linear(self.project_dim, self.project_dim),
                    nn.ELU(inplace=True),
                    nn.Linear(self.project_dim, self.project_dim))

    def forward(self, x, edge_index, node_batch_id, edge_weight=None, edge_attr=None, **kwargs):
        z = x
        zs = []
        # 1. Obtain node embeddings
        for conv, bn in zip(self.convs, self.bn_norms):
            z = conv(z, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        # 2. Readout layer
        gs = [global_add_pool(z, node_batch_id) for z in zs] # [batch_size, hidden_channels]
        z = torch.cat(zs, dim=1)
        g = torch.cat(gs, dim=1)
        # g = self.post_transform(g)
        return z, g

class GraphEncoder(torch.nn.Module):
    def __init__(self, graph_node_encoder, augmentor, x_norm_part=1):
        super().__init__()
        if isinstance(graph_node_encoder, list):
            self.graph_node_encoder = graph_node_encoder[0]
            self.graph_node_encoder2 = graph_node_encoder[1]
        else:
            self.graph_node_encoder = graph_node_encoder # output: z, zs
            self.graph_node_encoder2 = None
        self.augmentor = augmentor
        self.x_norm_part = x_norm_part
    
    def forward(self, x, edge_index, node_batch_id, edge_weights=None, edge_attr=None, gnn_edge_attr=None, 
                cluster_labels=None, 
                eigen_values=None, eigen_vectors=None, 
                x_eigen_distance=None, mode="train", **kwargs):
        
        aug1, aug2 = self.augmentor
        
        if isinstance(self.x_norm_part, int) or isinstance(self.x_norm_part, float):
            x = x
        else:
            x = x/self.x_norm_part.to(x.device)
        # if is_undirected(edge_index, edge_weights):
        #     pre_adj = to_dense_adj(edge_index, edge_attr=edge_weights)
        # else:
        #     edge_index = to_undirected(edge_index, edge_attr=edge_weight_2)
        #     pre_adj = to_dense_adj(edge_index, edge_attr=edge_weights)
        # pre_D = torch.diag(pre_adj.sum(1))
        # pre_U, pre_S, pre_V = torch.linalg.svd(pre_adj)            

        # edge_attr = torch.concat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        z, g = self.graph_node_encoder(x, edge_index, node_batch_id, edge_weight=edge_weights, edge_attr=gnn_edge_attr)

        if mode=="train":
            aug_g1 = aug1(x, edge_index, edge_weights, edge_attr=edge_attr, cluster_labels=cluster_labels, node_batch_id=node_batch_id,
                                                    eigen_values=eigen_values, eigen_vectors=eigen_vectors, x_eigen_distance=x_eigen_distance)
            aug_g2 = aug2(x, edge_index, edge_weights, edge_attr=edge_attr, cluster_labels=cluster_labels, node_batch_id=node_batch_id,
                                                    eigen_values=eigen_values, eigen_vectors=eigen_vectors, x_eigen_distance=x_eigen_distance)
            x1, edge_index_1, edge_weight_1 = aug_g1[:3]
            x2, edge_index_2, edge_weight_2 = aug_g2[:3]
            
            """
            While in cases of edge adding, the shape of edge_index will change, 
            the edge_weight shape is keep same with edge_index in edge adding module,
            but gnn_edge_attr is not ensured, so we need to append some manually edge attrs.
            We always append added edges to the last of edge_index, so append manually edge attrs to the last is fine.
            """
            if gnn_edge_attr==None:
                gnn_edge_attr_1 = None
                gnn_edge_attr_2 = None
            else:
                if edge_index.shape[1] != edge_index_1.shape[1]:
                    padding_length = edge_index_1.shape[1] - edge_index.shape[1]
                    assert padding_length>=0, "the value of padding length must be non-negative"
                    gnn_edge_attr_1 = F.pad(gnn_edge_attr, (0, 0, 0, padding_length), mode='constant', value=0)
                else:
                    gnn_edge_attr_1 = gnn_edge_attr
                
                if edge_index.shape[1] != edge_index_2.shape[1]:
                    padding_length = edge_index_2.shape[1] - edge_index.shape[1]
                    assert padding_length>=0, "the value of padding length must be non-negative"
                    gnn_edge_attr_2 = F.pad(gnn_edge_attr, (0, 0, 0, padding_length), mode='constant', value=0)
                else:
                    gnn_edge_attr_2 = gnn_edge_attr

            """
            Encoding
            """
            z1, g1 = self.graph_node_encoder(x1, edge_index_1, node_batch_id, edge_weight=edge_weight_1, edge_attr=gnn_edge_attr_1)
            if self.graph_node_encoder2 == None:
                z2, g2 = self.graph_node_encoder(x2, edge_index_2, node_batch_id, edge_weight=edge_weight_2, edge_attr=gnn_edge_attr_2)
            else:
                z2, g2 = self.graph_node_encoder2(x2, edge_index_2, node_batch_id, edge_weight=edge_weight_2, edge_attr=gnn_edge_attr_2)

            return z, g, z1, g1, z2, g2, ((aug_g1), (aug_g2))
        else:
            return z, g
    
# Predictors
class Predictor(torch.nn.Module):
    def __init__(self, input_channels, label_num=2, num_layers=1, loss_name="nll", semisup=False):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_channels)
        self.linears = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.linears.append(
                nn.Linear(input_channels, input_channels))
            self.acts.append(nn.ReLU())
        
        self.linear2 = nn.Linear(input_channels, label_num)
        if loss_name == "nll":
            self.m = nn.LogSoftmax(dim=1)
        elif loss_name == "ce":
            self.m = nn.Softmax(dim=1)
        else:
            self.m = nn.Identity()
        self.semisup = semisup

    def forward(self, features):
        """
        features: either node or edge features
        """
        if not self.semisup:
            x = self.batch_norm(features)
        else:
            x = features
        
        for ln, act in zip(self.linears, self.acts):
            x = act(ln(x))
        
        preds = self.m(self.linear2(x).squeeze())
        return preds




import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3



class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, emb_dim, hidden_channels, edge_dim=0, num_layer=2, drop_ratio=0):
        # not like input_channels, emb_dim could be any size
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        if edge_dim==0:
            self.GCONV_CLASS = WGINConv
        else:
            self.GCONV_CLASS = GINEConv

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer==0:
                self.gnns.append(self.GCONV_CLASS(emb_dim, hidden_channels, edge_dim=edge_dim))
            else:
                self.gnns.append(self.GCONV_CLASS(hidden_channels, hidden_channels, edge_dim=edge_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.project_dim = hidden_channels * num_layer
        self.project = torch.nn.Sequential(
                    nn.Linear(self.project_dim, self.project_dim),
                    nn.ELU(inplace=True),
                    nn.Linear(self.project_dim, self.project_dim))
    

    def forward(self, x, edge_index, node_batch_id, edge_weight=None, edge_attr=None, **kwargs):
        # z = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        z = x
        zs = []

        # 1. Obtain node embeddings
        for conv, bn in zip(self.gnns, self.batch_norms):
            z = conv(z, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        # 2. Readout layer
        gs = [global_add_pool(z, node_batch_id) for z in zs] # [batch_size, hidden_channels]
        z = torch.cat(zs, dim=1)
        g = torch.cat(gs, dim=1)
        # g = self.post_transform(g)
        return z, g



class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, emb_dim, hidden_channels, num_tasks, edge_dim=0, num_layer=2, drop_ratio=0):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(emb_dim, hidden_channels, edge_dim=edge_dim, num_layer=2, drop_ratio=0)

        project_dim = hidden_channels * num_layer
        self.graph_pred_linear = torch.nn.Linear(project_dim, self.num_tasks)


    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, x, edge_index, node_batch_id, edge_weight=None, edge_attr=None, **kwargs):

        z, g = self.gnn(x, edge_index, node_batch_id, edge_weight=edge_weight, edge_attr=edge_attr)

        return self.graph_pred_linear(g)