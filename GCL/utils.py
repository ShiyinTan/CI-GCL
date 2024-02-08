from typing import *
import os
import torch
import random
import warnings
import numpy as np
import torch.nn.functional as F
import joblib
import os.path as osp
import networkx as nx


from sklearn.cluster import KMeans, SpectralClustering
from torch_geometric.data.data import BaseData
from torch_geometric.data import Batch, Dataset, Data
from torch_geometric.data.datapipes import DatasetAdapter
from typing import Any, List, Optional, Sequence, Union, Tuple
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from sklearn.model_selection import KFold, PredefinedSplit, GridSearchCV, StratifiedKFold
from torch_sparse import SparseTensor
from torch_geometric.utils import to_dense_adj, get_laplacian, add_self_loops, to_networkx, coalesce, unbatch

def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs and 'test_ratio' in kwargs
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        return {
            'train': indices[:train_size],
            'val': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        }
    elif split_mode == 'ogb':
        return dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())



def batchify_dict(dicts: List[dict], aggr_func=lambda x: x):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)
    res = {k: aggr_func(v) for k, v in res.items()}
    return res


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

_dot = lambda x,y: x @ y.t()

def compute_infonce(anchor, sample, pos_mask, tau=0.2, *args, **kwargs):
    """
    @pos_mask: indicate postive samples, 默认为h1[i], h2[i]
        default: eye(auchor.shape[0])
        extra_pos: eye(auchor.shape[0]) + supervised pos, h1[i] and h2[j] j in same class.
    @example: 0.5*compute_infonce(h1, h2, pos_mask) + 0.5*compute_infonce(h2, h1, pos_mask) 
    """
    sim = _similarity(anchor, sample) / tau # [len(h1), len(h2)]
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True)+1e-8) # [len(h1), len(h2)] - [len(h1), 1]
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    return -loss.mean()



def find_optimal_clusters(X, max_k, cluster_func='kmeans', n_neighbors=10, ):
    inertia_values = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        if cluster_func=='kmeans':
            cluster = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=100, random_state=0)
        else:
            cluster = SpectralClustering(n_neighbors=n_neighbors, n_clusters=k, affinity='precomputed_nearest_neighbors', assign_labels='cluster_qr')
        
        cluster.fit(X)
        inertia_values.append(cluster.inertia_)

    delta_inertia = [inertia_values[i] - inertia_values[i + 1] for i in range(len(inertia_values) - 1)]
    optimal_k = delta_inertia.index(max(delta_inertia)) + 1
    optimal_k = max(2, optimal_k)
    if cluster_func=='kmeans':
        cluster = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=100, random_state=0)
    else:
        cluster = SpectralClustering(n_neighbors=n_neighbors, n_clusters=optimal_k, affinity='precomputed_nearest_neighbors', assign_labels='cluster_qr')

    return optimal_k, cluster.fit(X)


def cluster_get(dataset, n_cluster):
    all_graph_cluster_labels = []
    all_graph_centrics = []
    all_graph_cluster_num = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if isinstance(dataset, Dataset):
            with tqdm(total=len(dataset), desc='(C)') as pbar:
                for graph in dataset:
                    n_cluster = min(n_cluster, graph.num_nodes)
                    # kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto").fit(graph.x.detach().cpu().numpy())
                    optimal_k, kmeans = find_optimal_clusters(graph.x.detach().cpu().numpy(), max_k=n_cluster)
                    cluster_labels = kmeans.labels_
                    centrics = kmeans.cluster_centers_
                    cluster_labels = torch.asarray(cluster_labels).unsqueeze(-1) # [n_nodes, 1]
                    centrics = torch.asarray(centrics)

                    all_graph_cluster_labels.append(cluster_labels)
                    all_graph_centrics.append(centrics)
                    all_graph_cluster_num.append(optimal_k)
                    pbar.set_postfix({'optimal_k': optimal_k})
                    pbar.update()
        elif isinstance(dataset, Data):
            n_cluster = min(n_cluster, dataset.num_nodes)
            # kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto").fit(graph.x.detach().cpu().numpy())
            optimal_k, kmeans = find_optimal_clusters(dataset.x.detach().cpu().numpy(), max_k=n_cluster, cluster_func='kmeans')
            cluster_labels = kmeans.labels_
            centrics = kmeans.cluster_centers_
            cluster_labels = torch.asarray(cluster_labels).unsqueeze(-1) # [n_nodes, 1]
            centrics = torch.asarray(centrics)
            all_graph_cluster_labels.append(cluster_labels)
            all_graph_centrics.append(centrics)
            all_graph_cluster_num.append(optimal_k)
            
    return all_graph_cluster_labels, all_graph_centrics, all_graph_cluster_num

def topo_cluster_labels_get(dataset, n_cluster): # topological cluster
    all_graph_cluster_labels = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if isinstance(dataset, Dataset):
            with tqdm(total=len(dataset), desc='(TC)') as pbar:
                for graph in dataset:
                    n_cluster = min(n_cluster, graph.num_nodes-1) # n_cluster < N
                    n_neighbors = int(graph.edge_index.shape[1]/graph.num_nodes)+1
                    n_neighbors = min(n_neighbors, graph.num_nodes)
                    
                    adj = to_dense_adj(graph.edge_index).squeeze().detach().cpu().numpy()
                    assert adj.shape[0] == graph.edge_index.max()+1
                    cluster = SpectralClustering(n_neighbors=n_neighbors, n_clusters=n_cluster, affinity='precomputed_nearest_neighbors', assign_labels='cluster_qr')
                    cluster = cluster.fit(adj)
                    cluster_labels = cluster.labels_
                    cluster_labels = torch.asarray(cluster_labels).unsqueeze(-1) # [n_nodes, 1]

                    all_graph_cluster_labels.append(cluster_labels)
                    pbar.update()
        elif isinstance(dataset, Data):
            n_cluster = min(n_cluster, dataset.num_nodes)
            n_neighbors = int(dataset.edge_index.shape[1]/dataset.num_nodes)+1
            n_neighbors = min(n_neighbors, dataset.num_nodes)

            adj = to_dense_adj(dataset.edge_index).squeeze().detach().cpu().numpy()
            cluster = SpectralClustering(n_neighbors=n_neighbors, n_clusters=n_cluster, affinity='precomputed_nearest_neighbors', assign_labels='cluster_qr')
            cluster = cluster.fit(adj)
            cluster_labels = cluster.labels_
            cluster_labels = torch.asarray(cluster_labels).unsqueeze(-1) # [n_nodes, 1]
            all_graph_cluster_labels.append(cluster_labels)
            
    return all_graph_cluster_labels


def compute_cluster_constrain_loss(x, cluster_labels, centrics, cluster_size, sim_func=_dot, device='cpu'):
    """
    BPR loss of triplet (x, cluster of x, other clusters)
    x: [n_nodes, fea_dim]
    cluster_labels: [n_nodes, 1]
    centrics: [n_clusters, fea_dim]
    cluster_size: int
    """
    # cluster_size = cluster_labels.max()+1
    # cluster_labels = torch.asarray(np.expand_dims(cluster_labels,1), device=device)
    # centrics = torch.asarray(centrics, device=device)
    idx_for_cluster_calc = torch.tile(torch.arange(cluster_size, device=device), [x.shape[0],1])
    sim = sim_func(x, centrics)
    cluster_labels = cluster_labels.reshape(-1,1)
    bpr_pos = sim[idx_for_cluster_calc==cluster_labels]
    bpr_negs = torch.reshape(sim[idx_for_cluster_calc!=cluster_labels], (-1,cluster_size-1))
    bpr_loss = 0
    for i in range(cluster_size-1):
        bpr_loss = bpr_loss - torch.log(torch.sigmoid(bpr_pos - bpr_negs[:,i])+1e-10).sum()
    return bpr_loss/x.shape[0]


class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        # g, train_mask, degree, edge_weights, subgraph_features, cluster_labels, cluster_centrics, cluster_num, topo_cluster_labels = batch[0]
        # data_list, train_mask, degree, edge_weights, subgraph_features, cluster_labels, cluster_centrics, cluster_num, topo_cluster_labels = zip(*batch)
        data_list, graph_id, train_mask, degree, edge_weights, \
            num_component, eigen_values, eigen_vectors, \
            x_eigen_values, x_eigen_vectors_U, x_eigen_vectors_V = zip(*batch)
        
        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        x_eigen_values = [torch.asarray(elem) for elem in x_eigen_values]
        x_eigen_vectors_U = [torch.asarray(elem) for elem in x_eigen_vectors_U]
        x_eigen_vectors_V = [torch.asarray(elem) for elem in x_eigen_vectors_V]
        extra_dict = {"eigen_values": eigen_values,
                        "x_eigen_values": x_eigen_values, 
                        "x_eigen_vectors_U": x_eigen_vectors_U,
                        "x_eigen_vectors_V": x_eigen_vectors_V}

        # combine all graph infos
        for graph, g_id, train_m, d, e_w, num_c, eig_vec in \
            zip(data_list, graph_id, train_mask, degree, edge_weights, num_component, eigen_vectors):
            graph.graph_id = g_id
            graph.train_mask = train_m
            graph.degree = d
            graph.edge_weights = e_w
            graph.num_component = num_c
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict
        
    def collate_fn(self, batch: List[Any]) -> Any:
        return self(batch)
    

    def collate_fn_without_features(self, batch):
        data_list, graph_id, train_mask, degree, edge_weights, num_component, eigen_values, eigen_vectors = zip(*batch)
        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        extra_dict = {"eigen_values": eigen_values}
        # combine all graph infos
        for graph, g_id, train_m, d, e_w, num_c, eig_vec in \
            zip(data_list, graph_id, train_mask, degree, edge_weights, num_component, eigen_vectors):
            graph.graph_id = g_id
            graph.train_mask = train_m
            graph.degree = d
            graph.edge_weights = e_w
            graph.num_component = num_c
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict

    def transfer_pretrain_collate_fn(self, batch: List[Any]) -> Any:
        data_list, graph_id, degree, eigen_values, eigen_vectors, x_eigen_values, x_eigen_vectors_U, x_eigen_vectors_V = zip(*batch)

        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        x_eigen_values = [torch.asarray(elem) for elem in x_eigen_values]
        x_eigen_vectors_U = [torch.asarray(elem) for elem in x_eigen_vectors_U]
        x_eigen_vectors_V = [torch.asarray(elem) for elem in x_eigen_vectors_V]
        extra_dict = {"eigen_values": eigen_values,
                        "x_eigen_values": x_eigen_values, 
                        "x_eigen_vectors_U": x_eigen_vectors_U,
                        "x_eigen_vectors_V": x_eigen_vectors_V}

        # combine all graph infos
        for graph, g_id, d, eig_vec in \
            zip(data_list, graph_id, degree, eigen_vectors):
            graph.graph_id = g_id
            graph.degree = d
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict
    
    def transfer_pretrain_collate_fn_without_features(self, batch: List[Any]) -> Any:
        data_list, graph_id, degree, eigen_values, eigen_vectors = zip(*batch)

        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        extra_dict = {"eigen_values": eigen_values}

        # combine all graph infos
        for graph, g_id, d, eig_vec in \
            zip(data_list, graph_id, degree, eigen_vectors):
            graph.graph_id = g_id
            graph.degree = d
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict

    def transfer_finetune_collate_fn(self, batch: List[Any]) -> Any:
        data_list, graph_id, train_mask, degree, eigen_values, eigen_vectors, x_eigen_values, x_eigen_vectors_U, x_eigen_vectors_V = zip(*batch)

        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        x_eigen_values = [torch.asarray(elem) for elem in x_eigen_values]
        x_eigen_vectors_U = [torch.asarray(elem) for elem in x_eigen_vectors_U]
        x_eigen_vectors_V = [torch.asarray(elem) for elem in x_eigen_vectors_V]
        extra_dict = {"eigen_values": eigen_values,
                        "x_eigen_values": x_eigen_values, 
                        "x_eigen_vectors_U": x_eigen_vectors_U,
                        "x_eigen_vectors_V": x_eigen_vectors_V}

        # combine all graph infos
        for graph, g_id, t_m, d, eig_vec in \
            zip(data_list, graph_id, train_mask, degree, eigen_vectors):
            graph.graph_id = g_id
            graph.degree = d
            graph.train_mask = t_m
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict
    
    def transfer_finetune_collate_fn_without_features(self, batch: List[Any]) -> Any:
        data_list, graph_id, train_mask, degree, eigen_values, eigen_vectors = zip(*batch)

        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        extra_dict = {"eigen_values": eigen_values}

        # combine all graph infos
        for graph, g_id, t_m, d, eig_vec in \
            zip(data_list, graph_id, train_mask, degree, eigen_vectors):
            graph.graph_id = g_id
            graph.train_mask = t_m
            graph.degree = d
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict
    
    def semi_collate_fn(self, batch: List[Any]) -> Any:
        data_list, train_mask, num_component, eigen_values, eigen_vectors, \
            x_eigen_values, x_eigen_vectors_U, x_eigen_vectors_V = zip(*batch)

        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        x_eigen_values = [torch.asarray(elem) for elem in x_eigen_values]
        x_eigen_vectors_U = [torch.asarray(elem) for elem in x_eigen_vectors_U]
        x_eigen_vectors_V = [torch.asarray(elem) for elem in x_eigen_vectors_V]
        extra_dict = {"eigen_values": eigen_values,
                        "x_eigen_values": x_eigen_values, 
                        "x_eigen_vectors_U": x_eigen_vectors_U,
                        "x_eigen_vectors_V": x_eigen_vectors_V}

        # combine all graph infos
        for graph, t_m, n_c, eig_vec in \
            zip(data_list, train_mask, num_component, eigen_vectors):
            graph.train_mask = t_m
            graph.num_component = n_c
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict
    
    def semi_collate_fn_without_features(self, batch: List[Any]) -> Any:
        data_list, train_mask, num_component, eigen_values, eigen_vectors = zip(*batch)

        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        extra_dict = {"eigen_values": eigen_values}

        # combine all graph infos
        for graph, t_m, n_c, eig_vec in \
            zip(data_list, train_mask, num_component, eigen_vectors):
            graph.train_mask = t_m
            graph.num_component = n_c
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict

    def adgcl_collate_fn(self, batch: List[Any]) -> Any:
        data_list, num_component, eigen_values, eigen_vectors = zip(*batch)

        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        extra_dict = {"eigen_values": eigen_values}

        # combine all graph infos
        for graph, n_c, eig_vec in \
            zip(data_list, num_component, eigen_vectors):
            graph.num_component = n_c
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict
    
    def autogcl_collate_fn(self, batch: List[Any]) -> Any:
        data_list, num_component, eigen_values, eigen_vectors, x_eigen_values, x_eigen_vectors_U, x_eigen_vectors_V = zip(*batch)

        eigen_values = [torch.asarray(elem) for elem in eigen_values]
        x_eigen_values = [torch.asarray(elem) for elem in x_eigen_values]
        x_eigen_vectors_U = [torch.asarray(elem) for elem in x_eigen_vectors_U]
        x_eigen_vectors_V = [torch.asarray(elem) for elem in x_eigen_vectors_V]
        extra_dict = {"eigen_values": eigen_values,
                        "x_eigen_values": x_eigen_values, 
                        "x_eigen_vectors_U": x_eigen_vectors_U,
                        "x_eigen_vectors_V": x_eigen_vectors_V}

        # combine all graph infos
        for graph, n_c, eig_vec in \
            zip(data_list, num_component, eigen_vectors):
            graph.num_component = n_c
            graph.eigen_vectors = eig_vec
        
        return Batch.from_data_list(data_list, self.follow_batch, self.exclude_keys), extra_dict

    def my_collate_fn(self, collate_mode, has_node_features):
        if collate_mode in ["unsup"]:
            if has_node_features:
                return self.collate_fn
            else:
                return self.collate_fn_without_features
        elif collate_mode in  ["semi_pretrain", "semi_finetune"]:
            if has_node_features:
                return self.semi_collate_fn
            else:
                return self.semi_collate_fn_without_features
        elif collate_mode in ["trans_pretrain"]:
            if has_node_features:
                return self.transfer_pretrain_collate_fn
            else:
                return self.transfer_pretrain_collate_fn_without_features
        elif collate_mode in ["trans_finetune"]:
            if has_node_features:
                return self.transfer_finetune_collate_fn
            else:
                return self.transfer_finetune_collate_fn_without_features
        elif collate_mode in ["adgcl"]:
            return self.adgcl_collate_fn
        elif collate_mode in ["autogcl"]:
            if has_node_features:
                return self.autogcl_collate_fn
            else:
                return self.adgcl_collate_fn


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter, Tuple],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        collate_mode = "unsup", # ["unsup", "trans_pretrain", "trans_finetune", "semi_pretrain", "semi_finetune", "adgcl", "autogcl"]
        has_node_features = True, 
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.collator = Collater(follow_batch, exclude_keys)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            # collate_fn=self.collator.collate_fn,
            collate_fn=self.collator.my_collate_fn(collate_mode, has_node_features),
            **kwargs,
        )


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


def save_model_evaluator(checkpoints_path, encoder_model, best_evaluator=None, predictor=None, post_fix="best", par_id=0):
    if not osp.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    
    if best_evaluator != None:
        with joblib.parallel_backend('loky'):
            joblib.dump(best_evaluator, osp.join(checkpoints_path, f"best_evaluator_{post_fix}_{par_id}.pkl"))
    torch.save(encoder_model, osp.join(checkpoints_path, f"best_encoder_{post_fix}_{par_id}.pt"))
    if predictor!=None:
        torch.save(predictor, osp.join(checkpoints_path, f"best_predictor_{post_fix}_{par_id}.pt"))


def load_model_evaluator(checkpoints_path, post_fix="best", par_id=0, device="cpu", old_version=False):
    if old_version:
        if osp.exists(osp.join(checkpoints_path, f"best_evaluator_{par_id}.pkl")):
            best_evaluator = joblib.load(osp.join(checkpoints_path, f"best_evaluator_{par_id}.pkl"))
        else:
            best_evaluator = None
        if osp.exists(osp.join(checkpoints_path, f"best_encoder_{par_id}.pt")):
            best_encoder_model = torch.load(osp.join(checkpoints_path, f"best_encoder_{par_id}.pt"), map_location=device)
        else:
            best_encoder_model = None
        if osp.exists(osp.join(checkpoints_path, f"best_predictor_{par_id}.pt")):
            best_predictor = torch.load(osp.join(checkpoints_path, f"best_predictor_{par_id}.pt"), map_location=device)
        else:
            best_predictor = None
    else:
        if osp.exists(osp.join(checkpoints_path, f"best_evaluator_{post_fix}_{par_id}.pkl")):
            best_evaluator = joblib.load(osp.join(checkpoints_path, f"best_evaluator_{post_fix}_{par_id}.pkl"))
        else:
            best_evaluator = None

        if osp.exists(osp.join(checkpoints_path, f"best_encoder_{post_fix}_{par_id}.pt")):
            best_encoder_model = torch.load(osp.join(checkpoints_path, f"best_encoder_{post_fix}_{par_id}.pt"), map_location=device)
        else:
            best_encoder_model = None

        if osp.exists(osp.join(checkpoints_path, f"best_predictor_{post_fix}_{par_id}.pt")):
            best_predictor = torch.load(osp.join(checkpoints_path, f"best_predictor_{post_fix}_{par_id}.pt"), map_location=device)
        else:
            best_predictor = None

    return best_encoder_model, best_evaluator, best_predictor


def load_model_evaluator_old_version(checkpoints_path, par_id=0, device="cpu"):
    if osp.exists(osp.join(checkpoints_path, f"best_evaluator_{par_id}.pkl")):
        best_evaluator = joblib.load(osp.join(checkpoints_path, f"best_evaluator_{par_id}.pkl"))
    else:
        best_evaluator = None

    if osp.exists(osp.join(checkpoints_path, f"best_encoder_{par_id}.pt")):
        best_encoder_model = torch.load(osp.join(checkpoints_path, f"best_encoder_{par_id}.pt"), map_location=device)
    else:
        best_encoder_model = None

    if osp.exists(osp.join(checkpoints_path, f"best_predictor_{par_id}.pt")):
        best_predictor = torch.load(osp.join(checkpoints_path, f"best_predictor_{par_id}.pt"), map_location=device)
    else:
        best_predictor = None

    return best_encoder_model, best_evaluator, best_predictor


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def add_extra_pos_mask(pos_mask, data, predict_train_mask=None):
    # predict_train_mask: [n_sample, ]
    # predict_train_mask.unsqueeze(1): [n_sample, 1]
    extra_pos_mask = torch.eq(data.y, data.y.unsqueeze(dim=1))
    # construct extra supervision signals for only training samples
    if predict_train_mask!=None:
        # extra_pos_mask[~predict_train_mask][:, ~predict_train_mask] = False
        extra_pos_mask[~predict_train_mask] = False
        extra_pos_mask[:, ~predict_train_mask] = False
    extra_pos_mask.fill_diagonal_(False)
    pos_mask += extra_pos_mask

    # trues = data.y
    # labels = data.y.unique()
    # num_samples = data.y.shape[0]
    # for label in labels:
    #     label_idx_vector = torch.zeros((num_samples, 1), dtype=bool, device=pos_mask.device)
    #     if predict_train_mask!=None:
    #         indxs_for_label = torch.argwhere((trues==label)*predict_train_mask).squeeze()
    #     else:
    #         indxs_for_label = torch.argwhere(trues==label).squeeze()
    #     label_idx_vector[indxs_for_label] = True
    #     pos_mask += label_idx_vector.mul(label_idx_vector.T)
    return pos_mask

def degree(edge_index, edge_weight = None): # calculate the degree
    if edge_weight==None:
        edge_weight = torch.ones((edge_index.shape[1],), device=edge_index.device)

    # Sparse matrix to represent adjacency matrix
    adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight)

    # sum degree
    node_degrees = adj_matrix.sum(dim=1).to_dense()
    return node_degrees

def compute_eig_vals_changed(node_mask, num_nodes_single_graph, aug_edge_index, aug_edge_weights, num_component, num_eig_vals):
    edge_index_single_graph_mask = node_mask[aug_edge_index[0]] + node_mask[aug_edge_index[1]]
    edge_index_single_graph = aug_edge_index[:, edge_index_single_graph_mask]
    edge_weights_single_graph = aug_edge_weights[edge_index_single_graph_mask]
    if torch.isnan(edge_weights_single_graph).any():
        return None
    
    edge_index_idx_single_graph = edge_index_single_graph-(edge_index_single_graph.max()+1-num_nodes_single_graph)

    if edge_index_single_graph.shape[1]==0:
        return None
    
    assert edge_index_idx_single_graph.min()>=0
    assert edge_index_idx_single_graph.max()<=(num_nodes_single_graph-1)

    edge_index_idx_single_graph, edge_weights_single_graph = add_self_loops(edge_index_idx_single_graph, edge_weights_single_graph, fill_value=1.0, num_nodes=num_nodes_single_graph)
    lap_indicis, lap_values = get_laplacian(edge_index_idx_single_graph, edge_weights_single_graph, normalization='sym', num_nodes=num_nodes_single_graph)
    
    norm_L = torch.sparse_coo_tensor(lap_indicis, lap_values.squeeze(), size=(num_nodes_single_graph, num_nodes_single_graph)).to_dense()
    # if torch.isnan(lap_values).any() or torch.isnan(norm_L).any():
    #     print(torch.isnan(edge_index_idx_single_graph).any(), torch.isnan(edge_weights_single_graph).any(), torch.isnan(lap_values).any(), torch.isnan(norm_L).any())
    try:
        after_eig_vals = torch.linalg.eigvalsh(norm_L) # eigvalsh or eigvals
    except:
        after_eig_vals = torch.linalg.eigvals(norm_L)
    indices = after_eig_vals.real.sort().indices[num_component:num_component+num_eig_vals]
    after_eig_vals = after_eig_vals[indices].real
    return after_eig_vals

def compute_spectral_topo_loss(edge_index, edge_weights, aug_g, eigen_values, num_component, node_batch_id, aug_type):
    if len(aug_g)==3:
        aug_x, aug_edge_index, aug_edge_weights = aug_g
        edge_index_ed = edge_weights_ed = add_edge_index = add_edge_weights = None # only edge dropping or edge adding

    else:
        aug_x, aug_edge_index, aug_edge_weights, edge_index_ed, edge_weights_ed, add_edge_index, add_edge_weights = aug_g
    
    spectral_topo_loss = 0

    for i in range(node_batch_id.max()+1): # for each graph
        before_eig_vals = eigen_values[i]
        if isinstance(num_component, int):
            num_component_single_graph = num_component
        else:    
            num_component_single_graph = num_component[i]

        node_mask = (node_batch_id==i)
        num_nodes_single_graph = node_mask.sum() # num_nodes
        if num_nodes_single_graph > 5: # only for graph with number of nodes more than 5
            if edge_index_ed==None: # only edge dropping or edge adding
                after_eig_vals = compute_eig_vals_changed(node_mask, num_nodes_single_graph, aug_edge_index, aug_edge_weights, num_component_single_graph, len(before_eig_vals))
                if after_eig_vals != None:
                    assert before_eig_vals.dim()==1
                    assert after_eig_vals.dim()==1
                    k = min(len(before_eig_vals), len(after_eig_vals))

                    if aug_type=="LearnableEdgeDropping":
                        spectral_topo_loss -= torch.pow(after_eig_vals[:k]-before_eig_vals[:k], 2).sum()
                    else:
                        spectral_topo_loss += torch.pow(after_eig_vals[:k]-before_eig_vals[:k], 2).sum()
            else: # edge perturbation
                # edge dropping 
                aug_edge_index_ed = edge_index_ed
                aug_edge_weights_ed = edge_weights_ed
                after_eig_vals = compute_eig_vals_changed(node_mask, num_nodes_single_graph, aug_edge_index_ed, aug_edge_weights_ed, num_component_single_graph, len(before_eig_vals))
                if after_eig_vals==None:
                    return 0
                else: # single graph at least have one edge
                    assert before_eig_vals.dim()==1
                    assert after_eig_vals.dim()==1
                    k = min(len(before_eig_vals), len(after_eig_vals))

                    spectral_topo_loss -= torch.pow(after_eig_vals[:k]-before_eig_vals[:k], 2).sum()
                    # edge adding
                    aug_edge_index_ea = torch.concat([edge_index, add_edge_index], dim=1)
                    aug_edge_weights_ea = torch.concat([edge_weights, add_edge_weights])
                    aug_edge_index_ea, aug_edge_weights_ea = coalesce(aug_edge_index_ea, aug_edge_weights_ea, reduce='sum')

                    after_eig_vals = compute_eig_vals_changed(node_mask, num_nodes_single_graph, aug_edge_index_ea, aug_edge_weights_ea, num_component_single_graph, len(before_eig_vals))
                    if after_eig_vals==None:
                        return 0
                    else: # single graph at least have one edge
                        spectral_topo_loss += torch.pow(after_eig_vals[:k]-before_eig_vals[:k], 2).sum()
    
    return spectral_topo_loss


def compute_spectral_feature_loss(aug_g, eigen_values, node_batch_id):
    aug_x = aug_g[0]
    spectral_loss = 0
    ill_graph_num = 0
    aug_x_list = unbatch(aug_x, node_batch_id)
    for i, x_single_graph in enumerate(aug_x_list): # for each graph
        before_eig_vals = eigen_values[i]
        if x_single_graph.shape[0] > 5: # cal loss for grpah with nodes more than 5
            try:
                after_eig_vals = torch.linalg.svdvals(x_single_graph)
                k = min(before_eig_vals.shape[0], after_eig_vals.shape[0])
                single_graph_loss = torch.pow(after_eig_vals[:k]-before_eig_vals[:k], 2).sum()
            except:
                single_graph_loss = 0
                ill_graph_num += 1
            spectral_loss -= single_graph_loss

    # if ill_graph_num > 0:
    #     print("ill_conditioned_graphs: ", ill_graph_num)

    # for i in range(node_batch_id.max()+1): # for each graph
    #     before_eig_vals = eigen_values[i]
    #     node_mask = (node_batch_id==i)
    #     num_nodes_single_graph = node_mask.sum()
    #     if num_nodes_single_graph > 5: # cal loss for grpah with nodes more than 5
    #         x_single_graph = aug_x[node_mask]
    #         after_eig_vals = torch.linalg.svdvals(x_single_graph).real
    #         # expand_X = F.pad(x_single_graph, (x_single_graph.shape[0], 0, 0, x_single_graph.shape[1]), mode='constant', value=0)
    #         # A_X = expand_X+expand_X.T
    #         # after_eig_vals = torch.linalg.eigvalsh(A_X)

    #         k = min(before_eig_vals.shape[0], after_eig_vals.shape[0])

    #         # indices = after_eig_vals.real.sort().indices[1:1+k]
    #         # after_eig_vals = after_eig_vals[indices].real

    #         spectral_loss -= torch.pow(after_eig_vals[:k]-before_eig_vals[:k], 2).sum()
    
    return spectral_loss


def print_memory_usage():
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


def unravel_index(index, shape):
    indices_2d = torch.stack((index // shape[1], 
                              index % shape[1]), dim=1)
    return indices_2d


# for Parallel
def process_i_degree_edge_weight(i,data):
    return process_i(i), process_degree(data), process_edge_weight(data)

def process_i(i):
    return i

def process_degree(data):
    node_degree = degree(data.edge_index)
    graph_degree = node_degree.sum()
    node_degree_features = (node_degree/graph_degree).reshape(-1, 1)
    return node_degree_features

def process_edge_weight(data):
    edge_weights = torch.ones((data.edge_index.shape[1],))
    return edge_weights

def process_topo_eigens(data, EIGEN_VEC_NUM=50, num_component=1):
    # Normalized laplacian
    lap_indicis, lap_values = get_laplacian(data.edge_index, normalization='sym')
    norm_L = torch.sparse_coo_tensor(lap_indicis, lap_values, size=(data.num_nodes, data.num_nodes)).to_dense()

    eig_vals, eig_vecs = torch.linalg.eig(norm_L)
    indices = eig_vals.real.sort().indices[num_component:num_component+EIGEN_VEC_NUM]
    eig_vals = eig_vals[indices].real
    eig_vecs = eig_vecs[:, indices].real
    if eig_vecs.shape[1] < EIGEN_VEC_NUM:
        zero_columns = torch.zeros(eig_vecs.shape[0], EIGEN_VEC_NUM - eig_vecs.shape[1], device=eig_vecs.device)
        eig_vecs = torch.cat((eig_vecs, zero_columns), dim=1)
    return eig_vals, eig_vecs

def process_xtopo_eigens(data, X_EIGEN_VEC_NUM=50, x_norm_part=1):
    x = data.x.float()/x_norm_part
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    U = U[:, :X_EIGEN_VEC_NUM]
    V = Vh.T[:, :X_EIGEN_VEC_NUM]
    S = S[:X_EIGEN_VEC_NUM]
    
    if U.shape[1] < X_EIGEN_VEC_NUM:
        zero_columns = torch.zeros(U.shape[0], X_EIGEN_VEC_NUM - U.shape[1], device=U.device)
        U = torch.cat((U, zero_columns), dim=1)
    if V.shape[1] < X_EIGEN_VEC_NUM:
        zero_columns = torch.zeros(V.shape[0], X_EIGEN_VEC_NUM - V.shape[1], device=V.device)
        V = torch.cat((V, zero_columns), dim=1)

    return S, U, V


def k_fold(num_samples, y=None, fold_num=10, epoch_select="test_max", semi_sup_rate=1.0, seed=None):
    """
    Fold generator: use KFold when y not provided, use StratifiedKFold when y is provided.

    test set size: 1.0/fold_num, where default is 10%
    validation set size: 1.0/fold_num, where default is 10%
    train set size: min(1.0/fold_num, semi_sup_rate), where default is 80%, 
                    in semi-supervise setting, is 1% or 10%
    """
    if y==None:
        kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
    else:
        kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    
    # all_folds = []
    # for _, test_idx in kf.split(range(num_samples), y=y):
    #     all_folds.append(test_idx)

    num_semi_train = int(num_samples * semi_sup_rate)
    for train_idx, test_idx in kf.split(range(num_samples), y=y):
        val_idx = np.array([])
        trian_idx_idx = np.random.permutation(len(train_idx))
        train_idx = train_idx[trian_idx_idx]
        if epoch_select=="val_max":
            num_val = len(test_idx)
            val_idx = train_idx[:num_val]
            train_idx = train_idx[num_val:num_val+num_semi_train]
        else:
            val_idx = test_idx
            train_idx = train_idx[:num_semi_train]
        yield train_idx, val_idx, test_idx


def split_xy(x, y, split):
    """
    return: 
    x_train, x_test, x_valid, y_train, y_test, y_valid
    """
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]] for obj in objs for key in keys]


def print_args(args):
    args_dict = vars(args)

    for i, (key, value) in enumerate(args_dict.items()):
        print(f"{key}={value}", end=', ')
        if i % 8 == 0:
            print()