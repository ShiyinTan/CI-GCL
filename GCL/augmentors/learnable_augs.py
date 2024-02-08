from GCL.augmentors.augmentor import Graph, Augmentor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce, batched_negative_sampling, to_undirected
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
from torch.utils.data import DataLoader


gumble = lambda u: -torch.log(-torch.log(u))

class LearnableAugmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass


class Compose(LearnableAugmentor):
    def __init__(self, augmentors: List[LearnableAugmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def __call__(self, x, edge_index, edge_weights, **kwargs):
        for aug in self.augmentors:
            x, edge_index, edge_weights = aug(x, edge_index, edge_weights, **kwargs)
        return x, edge_index, edge_weights

class LearnableNodeDropping(LearnableAugmentor, nn.Module):
    def __init__(self, input_dim, hidden_dim, temp=1.0):
        super(LearnableNodeDropping, self).__init__()
        nn.Module.__init__(self)

        self.prob_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, 1))
        self.temp = temp
    
    def forward(self, x, edge_index, edge_weights, **kwargs) -> Graph:
        # x, edge_index, edge_weights = g.unfold()
        logits = self.prob_encoder(x)
        uni_rand = torch.rand(logits.shape, device=x.device)
        possibilities = torch.sigmoid((logits+gumble(uni_rand))/self.temp)
        x = torch.mul(x, possibilities)
        g = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        return g.unfold()

class LearnableFeatureDropping(LearnableAugmentor, nn.Module):
    def __init__(self, input_dim, hidden_dim, temp=1.0):
        """
        input_dim: batch num_nodes
        """
        super(LearnableFeatureDropping, self).__init__()
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.prob_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, 1))
        self.temp = temp
        self.aug_ratio = 0
    
    def forward(self, x, edge_index, edge_weights, **kwargs) -> Graph:
        # x, edge_index, edge_weights = g.unfold()
        rand_sample_idx = torch.randint(0, x.shape[0], (self.input_dim,))
        logits = self.prob_encoder(x[rand_sample_idx].T) # [F,1]
        uni_rand = torch.rand(logits.shape, device=x.device)
        possibilities = torch.sigmoid((logits+gumble(uni_rand))/self.temp).T # [1,F]
        aug_ratio = 1-(possibilities.sum().item()/x.shape[1])
        self.aug_ratio = aug_ratio
        x = torch.mul(x, possibilities)
        g = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        return g.unfold()


class LearnableEdgeDropping(LearnableAugmentor, nn.Module):
    def __init__(self, input_dim, hidden_dim, temp=1.0, control_by_ratio=False, drop_edge_ratio=1.0):
        """
        drop_edge_ratio: how many edges are selected to be droped, effect if control_by_ratio==True
        """
        super(LearnableEdgeDropping, self).__init__()
        nn.Module.__init__(self)

        self.prob_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, 2))
        self.temp = temp
        self.aug_ratio = 0
        self.control_by_ratio = control_by_ratio
        self.drop_edge_ratio = drop_edge_ratio
    
    def forward(self, x, edge_index, edge_weights, edge_attr, batch_train=False, return_changed_part=False, **kwargs) -> Graph:
        # x, edge_index, edge_weights = g.unfold()
        assert edge_index.shape[1] == edge_weights.shape[0], "edge_index shape must match edge_weights shape"
        # assert edge_index.shape[1] == edge_attr.shape[0], "edge_index shape must match edge_attr shape"
        all_possibilities = []
        indices = torch.arange(edge_index.shape[1])
        try:
            if self.control_by_ratio:
                num_of_edge_drop = int(self.drop_edge_ratio * edge_index.shape[1])
            else:
                num_of_edge_drop = edge_index.shape[1]
        except:
            num_of_edge_drop = edge_index.shape[1]
        

        logits = self.prob_encoder(edge_attr)

        if self.prob_encoder[-1].out_features == 1: # sigmoid
            logits = torch.log(torch.sigmoid(logits))
            uni_rand = torch.rand(logits.shape, device=x.device)
            possibilities = torch.sigmoid((logits+gumble(uni_rand))/self.temp)
        else: # softmax
            logits = torch.log(torch.softmax(logits, 1)+1e-8)
            possibilities = F.gumbel_softmax(logits, tau=self.temp, hard=False)[:,0].view(-1, 1)

        possibilities = torch.clamp(1.0 - possibilities, 1e-6, 1.0) # clamp avoid nan 

        try:
            if self.control_by_ratio:
                sorted_indices = torch.argsort(possibilities, descending=True)
                mask = torch.zeros((possibilities.shape[0],), dtype=torch.bool)
                mask[sorted_indices[:num_of_edge_drop]] = True # largest possibility
                
                # smallest probability
                possibilities[~mask].requires_grad_ = False
                
                edge_weights[mask] = torch.mul(edge_weights[mask], possibilities[mask])
            else:
                edge_weights = torch.mul(edge_weights, possibilities)
        except:
            edge_weights = torch.mul(edge_weights, possibilities)

        edge_index, edge_weights = to_undirected(edge_index, edge_attr=edge_weights, reduce = 'mean')
        return x, edge_index, edge_weights


class LearnableEdgeAdding(LearnableAugmentor, nn.Module):
    def __init__(self, input_dim, hidden_dim, temp=1.0, sample_edges_ratio=1.0, ratio_of_edge_insert=0, num_of_edge_insert=50):
        """
        ratio_of_edge_insert: the ratio of edges are selected to insert
        num_of_edge_insert: how many edges are selected to insert, effect if ratio_of_edge_insert!=0
        """
        super(LearnableEdgeAdding, self).__init__()
        nn.Module.__init__(self)

        self.prob_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, 2))
        self.temp = temp
        self.sample_edges_ratio = sample_edges_ratio
        self.ratio_of_edge_insert = ratio_of_edge_insert
        self.num_of_edge_insert = num_of_edge_insert

    def forward(self, x, edge_index, edge_weights, node_batch_id=None, eigen_vectors=None, return_changed_part=False, **kwargs):
        """
        sampled_edges: uniformly random sample edges from graph
        sampled_edges_attr: sampled_edges_attr
        return_changed_part: True if you want further use added_edges and added_edges_weight in future, 
                            in our case it is used to calculate spectral loss
        edge_weights: [edge_num, 1]
        """
        # sample edges stick in same graph
        num_edges = edge_index.shape[1]
        # num_sample_edges = int(num_edges*self.sample_edges_ratio)
        try:
            if self.ratio_of_edge_insert == 0:
                num_of_edge_insert = self.num_of_edge_insert
            else:
                num_of_edge_insert = int(self.ratio_of_edge_insert * num_edges)
        except:
            num_of_edge_insert = self.num_of_edge_insert
        
        sampled_edges = batched_negative_sampling(edge_index, batch=node_batch_id).long()
        perm_indices = torch.randperm(sampled_edges.shape[1]) # random perm sampled edges
        sampled_edges = sampled_edges[:, perm_indices]
        
        # sampled_edges_weight is the edge weights of graphs
        sampled_edges_weight = torch.ones((sampled_edges.shape[1], 1), device=edge_index.device)
        assert edge_index.dim()==sampled_edges.dim(), "dim not match"
        assert edge_weights.dim()==sampled_edges_weight.dim(), "dim not match"

        # batching 
        all_possibilities = []
        indices = torch.arange(sampled_edges.shape[1])
        dataloader = DataLoader(indices, batch_size=10240, shuffle=False)
        for batch_indices in dataloader:
            batch_sampled_edges = sampled_edges[:, batch_indices]
            # for edge adding, we use the inverse of distance as edges_attr
            sampled_edges_attr = torch.pow(eigen_vectors[batch_sampled_edges[0]] - eigen_vectors[batch_sampled_edges[1]], 2)
            sampled_edges_attr = torch.concat([sampled_edges_attr, torch.concat([x[batch_sampled_edges[0]], x[batch_sampled_edges[1]]], dim=1)], dim=1)

            logits = self.prob_encoder(sampled_edges_attr)
            if self.prob_encoder[-1].out_features == 1: # sigmoid
                logits = torch.log(torch.sigmoid(logits)+1e-8)
                uni_rand = torch.rand(logits.shape, device=x.device)
                possibilities = torch.sigmoid((logits+gumble(uni_rand))/self.temp)
            else: # softmax
                logits = torch.log(torch.softmax(logits, dim=1)+1e-8)
                possibilities = F.gumbel_softmax(logits, tau=self.temp, hard=False)[:,0].view(-1, 1)
            
            possibilities = torch.clamp(possibilities, 1e-6, 1.0)
            all_possibilities.append(possibilities)
        
        possibilities = torch.concat(all_possibilities) # [sample_edge_num, 1]

        sorted_indices = torch.argsort(possibilities, dim=0, descending=True)
        # print(possibilities.shape, sorted_indices)
        mask = torch.zeros((possibilities.shape[0],), dtype=torch.bool)
        mask[sorted_indices[:num_of_edge_insert]] = True # largest possibility
        
        sampled_edges = sampled_edges[:, mask]
        sampled_edges_weight = sampled_edges_weight[mask]
        # small probability
        possibilities[~mask].requires_grad_ = False
        possibilities = possibilities[mask]

        # adding sampled edges to edges, and deduplicate
        sampled_edges_weight = torch.mul(sampled_edges_weight, possibilities)
        sampled_edges, sampled_edges_weight = to_undirected(sampled_edges, edge_attr=sampled_edges_weight, reduce = 'mean')
        
        edge_index = torch.concat([edge_index, sampled_edges], dim=1)
        edge_weights = torch.concat([edge_weights, sampled_edges_weight])

        edge_index, edge_weights = coalesce(edge_index, edge_weights, reduce='sum')

        if return_changed_part:
            return x, edge_index, edge_weights, sampled_edges, sampled_edges_weight
        return x, edge_index, edge_weights


class LearnableEdgePerturbation(LearnableAugmentor, nn.Module):
    def __init__(self, input_dim_drop, input_dim_add, hidden_dim, temp=1.0, 
                 sample_edges_ratio=1.0, ratio_of_edge_insert=0, num_of_edge_insert=50, 
                 control_by_ratio=False, drop_edge_ratio=1.0):
        """
        num_of_edge_insert: how many edges are selected to insert
        """
        super(LearnableEdgePerturbation, self).__init__()
        nn.Module.__init__(self)

        if control_by_ratio==False:
            self.leA_ed = LearnableEdgeDropping(input_dim_drop, hidden_dim, temp=temp)
        else:
            self.leA_ed = LearnableEdgeDropping(input_dim_drop, hidden_dim, temp=temp, 
                                                control_by_ratio=control_by_ratio, drop_edge_ratio=drop_edge_ratio)
        if ratio_of_edge_insert==0:
            self.leA_ea = LearnableEdgeAdding(input_dim_add, hidden_dim, temp=temp, 
                                            sample_edges_ratio=sample_edges_ratio, 
                                            num_of_edge_insert=num_of_edge_insert)
        else:
            self.leA_ea = LearnableEdgeAdding(input_dim_add, hidden_dim, temp=temp, 
                                            sample_edges_ratio=sample_edges_ratio, 
                                            ratio_of_edge_insert=ratio_of_edge_insert, 
                                            num_of_edge_insert=num_of_edge_insert)

    def forward(self, x, edge_index, edge_weights, edge_attr=None, node_batch_id=None, eigen_vectors=None, **kwargs):
        """
        sampled_edges: uniformly random sample edges from graph
        sampled_edges_attr: sampled_edges_attr

        returns: x, edge_index, edge_weights are the augmented graph after Edge Perturbation
                 edge_index_ed, edge_weights_ed are the augmented graph after Edge Dropping
                 added_edge_index, added_edge_weights are the added edges in the Edge Adding
        """
        x_ed, edge_index_ed, edge_weights_ed = self.leA_ed(x, edge_index, edge_weights, edge_attr=edge_attr)
        x, edge_index, edge_weights, added_edge_index, added_edge_weights = self.leA_ea(x_ed, edge_index_ed, edge_weights_ed, 
                                                                                        node_batch_id=node_batch_id, 
                                                                                        eigen_vectors=eigen_vectors, 
                                                                                        return_changed_part=True)
        

        return x, edge_index, edge_weights, edge_index_ed, edge_weights_ed, added_edge_index, added_edge_weights


class LearnableFeatureDroppingBySpectral(LearnableAugmentor, nn.Module):
    def __init__(self, input_dim, hidden_dim, temp=1.0, num_drop_ratio=0.0, num_drop=10):
        """
        input_dim: batch num_nodes
        """
        super(LearnableFeatureDroppingBySpectral, self).__init__()
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.prob_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, 2))
        self.temp = temp
        self.aug_ratio = 0
        self.num_drop_ratio = num_drop_ratio
        self.num_drop = num_drop

    def forward(self, x, edge_index, edge_weights, x_eigen_distance, **kwargs) -> Graph:
        """
        x_eigen_distance: [n_nodes, n_features, eigen_feas]. 
                        For a single graph, x_eigen_distance[i,j,:]: [eigen vector distance between node i and feature j]
        @sample: t1: [n_nodes, eigen_feas] t2: [n_features, eigen_feas]
            t1_expanded = t1.unsqueeze(1)  # [n_nodes, 1, eigen_feas]
            t2_expanded = t2.unsqueeze(0)  # [1, n_features, eigen_feas]
            x_eigen_distance = t1_expanded - t2_expanded  # [n_nodes, n_features, eigen_feas]
        """
        x_shape = x.shape
        # flatten x
        flattened_x = x_eigen_distance.contiguous().view(-1, x_eigen_distance.size(-1)) # [n_nodes*n_feas, eigen_feas]

        if self.num_drop_ratio==0:
            num_drop = self.num_drop
        else:
            num_drop = int(flattened_x.shape[0] * self.num_drop_ratio)
        
        logits = self.prob_encoder(flattened_x) # [n_nodes*n_feas, 1]
        if self.prob_encoder[-1].out_features == 1: # sigmoid
            logits = torch.log(torch.sigmoid(logits))
            uni_rand = torch.rand(logits.shape, device=x.device)
            possibilities = torch.sigmoid((logits+gumble(uni_rand))/self.temp)
        else: # softmax
            logits = torch.log(torch.softmax(logits, 1)+1e-8)
            possibilities = F.gumbel_softmax(logits, tau=self.temp, hard=False)[:,0].view(-1, 1)

        possibilities = torch.clamp(1-possibilities, 1e-6, 1.0)

        flattened_p = possibilities.flatten()
        flattened_vals, flattened_index = torch.topk(flattened_p, num_drop, largest=False)

        # possibilities = possibilities.view(x.shape[0], x.shape[1])
        # aug_ratio = 1-(possibilities.sum().item()/(x.shape[0] * x.shape[1]))
        aug_ratio = 1-(flattened_vals.sum().item()/num_drop)
        self.aug_ratio = aug_ratio

        x = torch.scatter_reduce(x.flatten(), dim=0, index=flattened_index, src=flattened_vals, reduce='prod').view(x_shape)
        # x = torch.mul(x, possibilities)

        g = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        return g.unfold()


class LearnableNodeAdding(LearnableAugmentor, nn.Module):
    def __init__(self, input_dim, hidden_dim, temp):
        super(LearnableNodeAdding, self).__init__()
        nn.Module.__init__(self)

        self.prob_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, 1))
        self.temp = temp
    def forward(self, x, edge_index, edge_weights, **kwargs) -> Graph:
        # x, edge_index, edge_weights = g.unfold()
        logits = self.prob_encoder(x)
        uni_rand = torch.rand(logits.shape, device=x.device)
        logits = torch.log(torch.sigmoid(logits))
        possibilities = torch.sigmoid((logits+gumble(uni_rand))/self.temp)
        x = torch.mul(x, possibilities)
        g = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        return g.unfold()


def calculate_spectral_loss():
    """
    @inputs:
        eig_vals: eigen values
        eig_vecs: eigen vectors
        pre_A: adjacency matrix before augmentation
        pre_D: degree matrix before augmentation
        after_A: adjacency matrix after augmentation
        after_D: degree matrix after augmentation
    @outputs:
        spectral loss
    """
    return