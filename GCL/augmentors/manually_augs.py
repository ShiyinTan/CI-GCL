from GCL.augmentors.augmentor import Graph, Augmentor
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import coalesce
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List


class ManuallyAugmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, x, edge_index, edge_weights, **kwargs):
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weights: Optional[torch.FloatTensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(x, edge_index, edge_weights, **kwargs)


def sample_cross_cluster_edges(edge_index, num_sample, cluster_labels):
    labels = cluster_labels.unique()
    nodes_each_cluster = [(cluster_labels == l).nonzero()[:,0].to(edge_index.device) for l in labels] # nodes index of each cluster
    nodes = torch.arange(edge_index.max()+1-cluster_labels.shape[0], edge_index.max()+1, device=edge_index.device)
    each_sample = (num_sample//len(nodes_each_cluster))+1
    sampled_edges = []
    # print("each graph: ", cluster_labels.shape, nodes.min(), nodes.max())
    for i in range(len(nodes_each_cluster)):
        src_cluster_idx = nodes_each_cluster[i] # label i's nodes index
        dst_cluster_idx = torch.concat(nodes_each_cluster[0:i]+nodes_each_cluster[i+1:]) # clusters except i
        src_cluster = nodes[src_cluster_idx] # src cluster nodes
        dst_cluster = nodes[dst_cluster_idx] # dst cluster nodes
        # src_indices = torch.randperm(len(src_cluster), device=edge_index.device)[:each_sample]
        # dst_indices = torch.randperm(len(dst_cluster), device=edge_index.device)[:each_sample]
        src_indices = torch.randint(0, len(src_cluster), (each_sample, ), device=edge_index.device)
        dst_indices = torch.randint(0, len(dst_cluster), (each_sample, ), device=edge_index.device)
        # print(src_cluster.device, src_indices.device, edge_index.device)
        src_nodes = src_cluster[src_indices]
        dst_nodes = dst_cluster[dst_indices]
        sampled_edges.append(torch.stack([src_nodes, dst_nodes]))

    sampled_edges = torch.concat(sampled_edges, dim=1)
    # print(num_sample, sampled_edges.shape)
    return sampled_edges

def sample_within_cluster_edges(edge_index, num_sample, cluster_labels):
    labels = cluster_labels.unique()
    nodes_each_cluster = [(cluster_labels == l).nonzero()[:,0].to(edge_index.device) for l in labels] # nodes index of each cluster
    nodes = torch.arange(edge_index.max()+1-cluster_labels.shape[0], edge_index.max()+1, device=edge_index.device)
    each_sample = (num_sample//len(nodes_each_cluster))+1
    sampled_edges = []
    # print("each graph: ", cluster_labels.shape, nodes.min(), nodes.max())
    for i in range(len(nodes_each_cluster)): # for each cluster label
        src_cluster_idx = nodes_each_cluster[i] # label i's nodes index
        dst_cluster_idx = nodes_each_cluster[i]
        src_cluster = nodes[src_cluster_idx] # src cluster nodes
        dst_cluster = nodes[dst_cluster_idx] # dst cluster nodes
        # src_indices = torch.randperm(len(src_cluster), device=edge_index.device)[:each_sample]
        # dst_indices = torch.randperm(len(dst_cluster), device=edge_index.device)[:each_sample]
        src_indices = torch.randint(0, len(src_cluster), (each_sample, ), device=edge_index.device)
        dst_indices = torch.randint(0, len(dst_cluster), (each_sample, ), device=edge_index.device)
        # print(src_cluster.device, src_indices.device, edge_index.device)
        src_nodes = src_cluster[src_indices]
        dst_nodes = dst_cluster[dst_indices]
        sampled_edges.append(torch.stack([src_nodes, dst_nodes]))

    sampled_edges = torch.concat(sampled_edges, dim=1)
    # print(num_sample, sampled_edges.shape)
    return sampled_edges


# manully add/drop edge only suit for one graph
# manully add cross cluster edges to destroy clusters
class CrossClusterEdgeAdding(ManuallyAugmentor):
    def __init__(self, ratio):
        super(CrossClusterEdgeAdding, self).__init__()
        self.ratio = ratio

    def augment(self, x, edge_index, edge_weights, cluster_labels, node_batch_id=None, **kwargs):
        # x, edge_index, edge_weights = g.unfold()
        if isinstance(cluster_labels, torch.Tensor):
            num_nodes = edge_index.max().item() + 1
            num_sample = int(edge_index.shape[1]*self.ratio)

            sampled_edge_index = sample_cross_cluster_edges(edge_index, num_sample, cluster_labels)
            sampled_edge_weights = torch.ones((sampled_edge_index.shape[1],), device=edge_index.device)
            edge_index = torch.cat([edge_index, sampled_edge_index], dim=1)
            if edge_weights!=None:
                edge_weights = torch.cat([edge_weights, sampled_edge_weights], dim=1)

            edge_index, edge_weights = coalesce(edge_index, edge_attr=edge_weights, num_nodes=num_nodes)
        elif isinstance(cluster_labels, list):
            assert len(cluster_labels)==(node_batch_id.max()+1), "cluster labels num not match graph num!"
            node_mask = torch.eq(node_batch_id, torch.arange(len(cluster_labels), device=edge_index.device).reshape(-1,1)) # [len(cluster_labels), n_nodes]
            """
            i_th graph: 
            x_single_graph = x[node_mask[i]] 
            edge_index_single_graph = edge_index[:, node_mask[i][edge_index[0]]]
            """
            all_edge_index = []
            all_edge_weights = []
            for i in range(len(cluster_labels)): # for each graph
                cluster_labels_single_graph = cluster_labels[i]
                edge_index_single_graph_mask = node_mask[i][edge_index[0]]
                edge_index_single_graph_mask += node_mask[i][edge_index[1]]
                edge_index_single_graph = edge_index[:, edge_index_single_graph_mask]
                edge_weights_single_graph = edge_weights[edge_index_single_graph_mask]
                # print(torch.cuda.memory_summary())

                num_sample = int(edge_index_single_graph.shape[1]*self.ratio)
                sampled_edge_index_single_graph = sample_cross_cluster_edges(edge_index_single_graph, num_sample, cluster_labels_single_graph)
                sampled_edge_weights_single_graph = torch.ones((sampled_edge_index_single_graph.shape[1],), device=edge_index.device)

                edge_index_single_graph = torch.cat([edge_index_single_graph, sampled_edge_index_single_graph], dim=1)
                edge_weights_single_graph = torch.cat([edge_weights_single_graph, sampled_edge_weights_single_graph], dim=0)

                all_edge_index.append(edge_index_single_graph)
                all_edge_weights.append(edge_weights_single_graph)
            edge_index = torch.concat(all_edge_index, dim=1)
            edge_weights = torch.concat(all_edge_weights, dim=0)
        return x, edge_index, edge_weights

class WithinClusterEdgeAdding(ManuallyAugmentor):
    def __init__(self, ratio):
        super(WithinClusterEdgeAdding, self).__init__()
        self.ratio = ratio

    def augment(self, x, edge_index, edge_weights, cluster_labels, node_batch_id, **kwargs):

        assert len(cluster_labels)==(node_batch_id.max()+1), "cluster labels num not match graph num!"
        node_mask = torch.eq(node_batch_id, torch.arange(len(cluster_labels), device=edge_index.device).reshape(-1,1)) # [len(cluster_labels), n_nodes]
        """
        i_th graph: 
        x_single_graph = x[node_mask[i]] 
        edge_index_single_graph = edge_index[:, node_mask[i][edge_index[0]]]
        """
        all_edge_index = []
        all_edge_weights = []
        for i in range(len(cluster_labels)): # for each graph
            cluster_labels_single_graph = cluster_labels[i]
            edge_index_single_graph_mask = node_mask[i][edge_index[0]] 
            edge_index_single_graph = edge_index[:, edge_index_single_graph_mask]
            edge_weights_single_graph = edge_weights[edge_index_single_graph_mask]


            num_sample = int(edge_index_single_graph.shape[1]*self.ratio)
            sampled_edge_index_single_graph = sample_within_cluster_edges(edge_index_single_graph, num_sample, cluster_labels_single_graph)
            sampled_edge_weights_single_graph = torch.ones((sampled_edge_index_single_graph.shape[1],), device=edge_index.device)

            edge_index_single_graph = torch.cat([edge_index_single_graph, sampled_edge_index_single_graph], dim=1)
            edge_weights_single_graph = torch.cat([edge_weights_single_graph, sampled_edge_weights_single_graph], dim=0)

            all_edge_index.append(edge_index_single_graph)
            all_edge_weights.append(edge_weights_single_graph)

        edge_index = torch.concat(all_edge_index, dim=1)
        edge_weights = torch.concat(all_edge_weights, dim=0)

        return x, edge_index, edge_weights


# manully drop within cluster edges to destroy clusters
class WithinClusterEdgeDropping(ManuallyAugmentor):
    def __init__(self, ratio):
        super(WithinClusterEdgeDropping, self).__init__()
        self.ratio = ratio

    def augment(self, x, edge_index, edge_weights, cluster_labels, node_batch_id, **kwargs):

        assert len(cluster_labels)==(node_batch_id.max()+1), "cluster labels num not match graph num!"
        node_mask = torch.eq(node_batch_id, torch.arange(len(cluster_labels), device=edge_index.device).reshape(-1,1)) # [len(cluster_labels), n_nodes]
        """
        i_th graph: 
        x_single_graph = x[node_mask[i]] 
        edge_index_single_graph = edge_index[:, node_mask[i][edge_index[0]]]
        """
        all_edge_index = []
        all_edge_weights = []
        for i in range(len(cluster_labels)): # for each graph
            cluster_labels_single_graph = cluster_labels[i]
            edge_index_single_graph_mask = node_mask[i][edge_index[0]] 
            edge_index_single_graph = edge_index[:, edge_index_single_graph_mask]
            if edge_weights!=None:
                edge_weights_single_graph = edge_weights[edge_index_single_graph_mask]

            edge_mask_single_graph = torch.ones((edge_index_single_graph.shape[1],), dtype=bool, device=edge_index.device)

            num_nodes_single_graph = cluster_labels_single_graph.shape[0]
            edge_index_idx_single_graph = edge_index_single_graph-(edge_index_single_graph.max()+1-num_nodes_single_graph)
            
            within_cluster_edge_id_single_graph = torch.eq(cluster_labels_single_graph[edge_index_idx_single_graph[0]], 
                                                           cluster_labels_single_graph[edge_index_idx_single_graph[1]]).nonzero().to(edge_index.device)[:,0]

            num_drop = int(edge_index_single_graph.shape[1]*self.ratio)
            sampled_edges_id_for_within_single_graph = torch.randperm(len(within_cluster_edge_id_single_graph), device=edge_index.device)[:num_drop]
            sampled_edges_id_single_graph = within_cluster_edge_id_single_graph[sampled_edges_id_for_within_single_graph]
            edge_mask_single_graph[sampled_edges_id_single_graph] = False
            edge_index_single_graph = edge_index_single_graph[:,edge_mask_single_graph]
            if edge_weights!=None:
                edge_weights_single_graph = edge_weights_single_graph[edge_mask_single_graph]

            all_edge_index.append(edge_index_single_graph)
            all_edge_weights.append(edge_weights_single_graph)
        edge_index = torch.concat(all_edge_index, dim=1)
        edge_weights = torch.concat(all_edge_weights, dim=0)

        return x, edge_index, edge_weights
    

# manully drop cross cluster edges to destroy clusters
class CrossClusterEdgeDropping(ManuallyAugmentor):
    def __init__(self, ratio):
        super(CrossClusterEdgeDropping, self).__init__()
        self.ratio = ratio

    def augment(self, x, edge_index, edge_weights, cluster_labels, node_batch_id, **kwargs):

        assert len(cluster_labels)==(node_batch_id.max()+1), "cluster labels num not match graph num!"
        node_mask = torch.eq(node_batch_id, torch.arange(len(cluster_labels), device=edge_index.device).reshape(-1,1)) # [len(cluster_labels), n_nodes]
        """
        i_th graph: 
        x_single_graph = x[node_mask[i]] 
        edge_index_single_graph = edge_index[:, node_mask[i][edge_index[0]]]
        """
        all_edge_index = []
        all_edge_weights = []
        for i in range(len(cluster_labels)): # for each graph
            cluster_labels_single_graph = cluster_labels[i]
            edge_index_single_graph_mask = node_mask[i][edge_index[0]] 
            edge_index_single_graph = edge_index[:, edge_index_single_graph_mask]
            if edge_weights!=None:
                edge_weights_single_graph = edge_weights[edge_index_single_graph_mask]

            edge_mask_single_graph = torch.zeros((edge_index_single_graph.shape[1],), dtype=bool, device=edge_index.device)

            num_nodes_single_graph = cluster_labels_single_graph.shape[0]
            edge_index_idx_single_graph = edge_index_single_graph-(edge_index_single_graph.max()+1-num_nodes_single_graph)
            
            within_cluster_edge_id_single_graph = torch.eq(cluster_labels_single_graph[edge_index_idx_single_graph[0]], 
                                                           cluster_labels_single_graph[edge_index_idx_single_graph[1]]).nonzero().to(edge_index.device)[:,0]
            edge_mask_single_graph[within_cluster_edge_id_single_graph] = True # whithin cluster edges do not drop

            num_sample = int(1-edge_index_single_graph.shape[1]*self.ratio) # sampled edges do not drop
            sampled_edges_id_for_single_graph = torch.randperm(len(edge_index_idx_single_graph), device=edge_index.device)[:num_sample]
            edge_mask_single_graph[sampled_edges_id_for_single_graph] = True 
            
            edge_index_single_graph = edge_index_single_graph[:,edge_mask_single_graph]
            if edge_weights!=None:
                edge_weights_single_graph = edge_weights_single_graph[edge_mask_single_graph]

            all_edge_index.append(edge_index_single_graph)
            all_edge_weights.append(edge_weights_single_graph)
        edge_index = torch.concat(all_edge_index, dim=1)
        edge_weights = torch.concat(all_edge_weights, dim=0)

        return x, edge_index, edge_weights