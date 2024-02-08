from cgi import test
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
import os
import os.path as osp
import scipy.sparse as sp
import re


from GCL.eval import get_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData, Data, Dataset, InMemoryDataset
from gensim.models import KeyedVectors
from torch_geometric.datasets import WikiCS, Planetoid, Coauthor, Amazon, TUDataset, MoleculeNet, GNNBenchmarkDataset, MNISTSuperpixels
from bio_data_loader import BioDataset
from chem_data_loader import MoleculeDataset
from torch.nn import Embedding
from collections import defaultdict
from random import choice
from torch.nn.utils.rnn import pad_sequence
from splitters import scaffold_split
from ogb.graphproppred import PygGraphPropPredDataset
from feature_expansion import FeatureExpander


def get_split_mask(num_samples, train_ratio=0.1, test_ratio=0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)

    # use scatter_() to fill mask matrix
    train_mask = torch.zeros(num_samples, dtype=bool)
    train_mask.scatter_(0, indices[:train_size], 1)
    test_mask = torch.zeros(num_samples, dtype=bool)
    test_mask.scatter_(0, indices[train_size: test_size + train_size], 1)
    val_mask = torch.zeros(num_samples, dtype=bool)
    val_mask.scatter_(0, indices[test_size + train_size:], 1)
    return train_mask, val_mask, test_mask

def get_subgraph_edge_attr(data, args, path, dataset_name):
    node_hashing_table_save_path = osp.join(path, dataset_name, f"node_hashing_table_{args.max_hash_hops}.pt")
    subgraph_structure_save_path = osp.join(path, dataset_name, f"subgraph_structure_{args.max_hash_hops}.pt")
    subgraph_features = subgraph_structure_features(data, data.edge_index, args, node_hashing_table_save_path, subgraph_structure_save_path)
    return subgraph_features

def get_subgraph_edge_attr_with_postfix(data, args, path, dataset_name, postfix, do_save=False):
    if do_save:
        node_hashing_table_save_path = osp.join(path, dataset_name, f"node_hashing_table_{args.max_hash_hops}_{postfix}.pt")
        subgraph_structure_save_path = osp.join(path, dataset_name, f"subgraph_structure_{args.max_hash_hops}_{postfix}.pt")
        subgraph_features = subgraph_structure_features(data, data.edge_index, args, node_hashing_table_save_path, subgraph_structure_save_path)
    else:
        subgraph_features = subgraph_structure_features(data, data.edge_index, args, None, None)
    return subgraph_features

def print_summary(data, dataset_name):
    # summary
    num_nodes = data.edge_index.max()+1
    num_node_features = data.num_node_features
    print(dataset_name, ", num_nodes: ", num_nodes, ", num_node_features: ", num_node_features)
    if isinstance(data, Data):
        print(dataset_name, ", edges shape: ", data.edge_index.shape, ", subgraph shape: ", data.subgraph_features.shape)
    elif isinstance(data, Dataset):
        print(dataset_name, ", edges shape: ", data.edge_index.shape)
    

def load_dataset_ncls(path, dataset_name, args):
    if dataset_name == 'WikiCS':
        data = WikiCS(osp.join(path, dataset_name), is_undirected=True)[0]

        # non public split
        # train_mask, val_mask, test_mask = get_split_mask(data.num_nodes, train_ratio=0.1, test_ratio=0.8)
        # data.train_mask = train_mask
        # data.val_mask = val_mask
        # data.test_mask = test_mask

        subgraph_features = get_subgraph_edge_attr(data, args, path, dataset_name)
        assert data.edge_index.shape[1]==subgraph_features.shape[0]
        data.subgraph_features = subgraph_features

    elif dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        data = Planetoid(root=path, name=dataset_name)[0]

        subgraph_features = get_subgraph_edge_attr(data, args, path, dataset_name)
        assert data.edge_index.shape[1]==subgraph_features.shape[0]
        data.subgraph_features = subgraph_features

    elif dataset_name in ["CS", "Physics"]: # ms_amazon_cs, ms_academic_phy
        data = Coauthor(root=path, name=dataset_name)[0]

        train_mask, val_mask, test_mask = get_split_mask(data.num_nodes, train_ratio=0.1, test_ratio=0.8)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        subgraph_features = get_subgraph_edge_attr(data, args, path, dataset_name)
        assert data.edge_index.shape[1]==subgraph_features.shape[0]
        data.subgraph_features = subgraph_features

    elif dataset_name in ["Computers", "Photo"]: # amazon_computers, amazon_photos
        data = Amazon(root=path, name=dataset_name)[0]

        train_mask, val_mask, test_mask = get_split_mask(data.num_nodes, train_ratio=0.1, test_ratio=0.8)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        subgraph_features = get_subgraph_edge_attr(data, args, path, dataset_name)
        assert data.edge_index.shape[1]==subgraph_features.shape[0]
        data.subgraph_features = subgraph_features

    else:
        raise TypeError(f"Dataset not available: '{type(dataset_name)}'")
    
    print_summary(data, dataset_name)
    if data.train_mask.dim()==1:
        train_num = data.train_mask.sum()
        val_num = data.val_mask.sum()
        test_num = data.test_mask.sum()
    else:
        train_num = data.train_mask[:,0].sum()
        val_num = data.val_mask[:,0].sum()
        test_num = data.test_mask.sum()
    print(dataset_name, ", train mask: ", train_num, ", val mask: ", val_num, ", test mask: ", test_num)
    return data

def load_dataset_graphcls(path, dataset_name, args=None, semisup=False, device="cpu"):
    if dataset_name in ["NCI1", "PROTEINS", "DD", "MUTAG", "COLLAB", "REDDIT-BINARY", "REDDIT-MULTI-5K", "IMDB-BINARY", "IMDB-MULTI", "github_stargazers"]:
        if semisup:
            dataset = get_semi_dataset(path, name=dataset_name, feat_str='deg+odeg100')
        else:
            dataset = TUDataset(path, name=dataset_name)
    elif dataset_name in ["pcba", "muv", "hiv", "bace", "bbbp", "tox21", "toxcast", "sider", "clintox", "lipo"]:
        # dataset = MoleculeNet(path, name=dataset_name)
        if dataset_name=="lipo":
            dataset = MoleculeDataset(os.path.join(path, 'chem_dataset', "lipophilicity"), dataset="lipophilicity", map_device="cpu")
        else:
            dataset = MoleculeDataset(os.path.join(path, 'chem_dataset', dataset_name), dataset=dataset_name, map_device="cpu")
    elif dataset_name in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider"]:
        # dataset = MoleculeNet(path, name=dataset_name)
        dataset = PygGraphPropPredDataset(name = dataset_name, root=path)
    elif dataset_name in ["ZINC"]:
        dataset = MoleculeDataset(os.path.join(path, 'chem_dataset', 'zinc_standard_agent'), dataset='zinc_standard_agent', map_device="cpu")
    elif dataset_name in ["PPI_sup", "PPI_unsup"]: # PPI supervised, PPI unsupervised
        is_sup = (dataset_name=="PPI_sup")
        dataset = BioDataset(osp.join(path, 'bio_dataset', 'supervised' if is_sup else 'unsupervised'), 
                             data_type='supervised' if is_sup else 'unsupervised', 
                             map_device=device)
    elif dataset_name in ["MNIST", "CIFAR10"]: 
        train_dataset = GNNBenchmarkDataset(path, name=dataset_name, split="train") # have "MNIST"
        val_dataset = GNNBenchmarkDataset(path, name=dataset_name, split="val")
        test_dataset = GNNBenchmarkDataset(path, name=dataset_name, split="test")
    # elif dataset_name in ["MNIST"]:
    #     dataset = MNISTSuperpixels(osp.join(path, dataset_name))
    else:
        raise ValueError('Not supported dataset!')
    
    if dataset_name in ["PPI_unsup", "ZINC", "PPI_sup"]:
        return dataset, None
    
    elif dataset_name in ["MNIST", "CIFAR10"]:
        num_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        mask = torch.zeros(num_samples, dtype=bool)
        train_mask = mask.scatter(0, torch.arange(len(train_dataset)), 1)
        val_mask = mask.scatter(0, torch.arange(len(train_dataset), len(train_dataset)+len(val_dataset)), 1)
        test_mask = mask.scatter(0, torch.arange(len(train_dataset)+len(val_dataset), num_samples), 1)
        dataset = train_dataset + val_dataset + test_dataset
        split = {"train": train_mask,
                "valid": val_mask,
                "test": test_mask}
        train_num = len(train_dataset)
        val_num = len(val_dataset)
        test_num = len(test_dataset)
    elif dataset_name in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider"]:
        split= dataset.get_idx_split()
        train_num = len(split['train'])
        val_num = len(split['valid'])
        test_num = len(split['test'])
        print_summary(dataset, dataset_name)

    elif dataset_name in ["pcba", "muv", "hiv", "bace", "bbbp", "tox21", "toxcast", "sider", "clintox", "lipo"]:
        num_samples = len(dataset)
        if dataset_name=="lipo":
            smiles_list = pd.read_csv(osp.join(path, 'chem_dataset', "lipophilicity", 'processed/smiles.csv'), header=None)[0].tolist()
        else:
            smiles_list = pd.read_csv(osp.join(path, 'chem_dataset', dataset_name, 'processed/smiles.csv'), header=None)[0].tolist()
        train_idx, val_idx, test_idx = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        split = {'train': torch.asarray(train_idx, device=device, dtype=torch.int64), 
                'valid': torch.asarray(val_idx, device=device, dtype=torch.int64),
                'test':  torch.asarray(test_idx, device=device, dtype=torch.int64)}
        print_summary(dataset, dataset_name)
        train_num = len(train_idx)
        val_num = len(val_idx)
        test_num = len(test_idx)
    else:
        num_samples = len(dataset)
        train_mask, val_mask, test_mask = get_split_mask(num_samples, train_ratio=0.1, test_ratio=0.8)
        split = {"train": train_mask,
                "valid": val_mask,
                "test": test_mask}
        
        print_summary(dataset, dataset_name)
        train_num = train_mask.sum()
        val_num = val_mask.sum()
        test_num = test_mask.sum()

    print(dataset[0].keys())
    print(dataset_name, ", train mask: ", train_num, ", val mask: ", val_num, ", test mask: ", test_num)
    return dataset, split


class CombinedDataset:
    def __init__(self, **kwargs):
        self.all_dataset = [values for values in zip(*kwargs.values())]
    
    def __getitem__(self, i):
        return self.all_dataset[i]
    
    def __len__(self):
        return len(self.all_dataset)
    

    
def get_semi_dataset(path, name, sparse=True, feat_str="deg+ak3+reall"):

    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall("an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    groupd = re.findall("groupd(\d+)", feat_str)
    groupd = int(groupd[0]) if groupd else 0
    remove_edges = re.findall("re(\w+)", feat_str)
    remove_edges = remove_edges[0] if remove_edges else 'none'
    edge_noises_add = re.findall("randa([\d\.]+)", feat_str)
    edge_noises_add = float(edge_noises_add[0]) if edge_noises_add else 0
    edge_noises_delete = re.findall("randd([\d\.]+)", feat_str)
    edge_noises_delete = float(
        edge_noises_delete[0]) if edge_noises_delete else 0
    centrality = feat_str.find("cent") >= 0
    coord = feat_str.find("coord") >= 0

    pre_transform = FeatureExpander(
        degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
        centrality=centrality, remove_edges=remove_edges,
        edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
        group_degree=groupd).transform

    dataset = TUDataset(path, name, pre_transform=pre_transform, use_node_attr=True)

    dataset.data.edge_attr = None

    return dataset