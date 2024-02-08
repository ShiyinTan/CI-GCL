import socket
from sympy import capture
import torch
import os
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import GCL.augmentors.learnable_augs as LeA
import GCL.augmentors.manually_augs as MaA
import logging
import joblib
import argparse
import numpy as np
import networkx as nx
import sys
import time

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader, Dataset
from GCL.models.mymodels import GraphNodeEncoder, GraphEncoder, Predictor
from GCL.utils import compute_infonce, cluster_get, CustomDataLoader, compute_cluster_constrain_loss, process_edge_weight, save_model_evaluator, \
    load_model_evaluator, str_to_bool, add_extra_pos_mask, degree, topo_cluster_labels_get, compute_spectral_topo_loss, compute_spectral_feature_loss, \
    process_degree, process_i_degree_edge_weight, process_topo_eigens, process_xtopo_eigens
from sklearn.metrics import f1_score
from datetime import datetime
from general_data_loader import load_dataset_graphcls, CombinedDataset, get_subgraph_edge_attr_with_postfix
from torch_geometric.utils import scatter, to_networkx, to_dense_adj, get_laplacian
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, parallel_backend

os.environ['TORCH_USE_CUDA_DSA'] = '1'

def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--load_subgraph", type=str_to_bool, default=True)
    parser.add_argument("--use_cluster_reg", type=str_to_bool, default=False)
    parser.add_argument("--use_degree", type=str_to_bool, default=False)
    parser.add_argument("--use_spectral_fea", type=str_to_bool, default=True)
    parser.add_argument("--use_spectral_topo", type=str_to_bool, default=True)

    parser.add_argument("--cluster_reg_weight", type=float, default=0.1)

    parser.add_argument("--mode", type=str, default="unsup", choices=["unsup", "semisup"])
    parser.add_argument("--dataset_name", type=str, default="ZINC")
    
    # for subgraph structure
    parser.add_argument("--hll_p", type=int, default=8)
    parser.add_argument("--minhash_num_perm", type=int, default=128)
    parser.add_argument("--max_hash_hops", type=int, default=3)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512) # 512
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--par_id", type=int, default=0)
    parser.add_argument("--preprocess_part", type=str, default="degree", choices=["degree", "topo", "xtopo", "load"])

    args = parser.parse_args()
    return args


def main():
    hostname = socket.gethostname()
    print("Current hostname:", hostname)

    EIGEN_VEC_NUM = 50
    x_eigen_load_dataset = True
    num_jobs = 4
    cpu_cores_num = os.cpu_count()
    print("cpu_cores_num: ", cpu_cores_num)
    num_thread_each_core = 2
    args = get_config()
    is_parallel = False

    if args.dataset_name in ["ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox"]:
        dataset_name = args.dataset_name.lower()
    else:
        dataset_name = args.dataset_name

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    current_time = datetime.now()
    print("Current time:", current_time)
    # print settings
    print("dataset: ", args.dataset_name, ", mode: ", args.mode, ", cluster_reg: ", args.use_cluster_reg, 
          ", cluster_reg_weight: ", args.cluster_reg_weight, ", n_clusters: ", args.n_clusters)
    
    if not osp.exists('./log'):
        os.mkdir('./log')
    if not osp.exists(f'./log/{dataset_name}'):
        os.mkdir(f'./log/{dataset_name}')
    logging_path = "./log/{}/results.log".format(dataset_name)
    logging.basicConfig(filename=logging_path, level=logging.DEBUG, format='%(asctime)s %(message)s')


    if not osp.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not osp.exists(f'./checkpoints/{dataset_name}'):
        os.mkdir(f'./checkpoints/{dataset_name}')

    if not osp.exists('./tmp'):
        os.mkdir('./tmp')
    if not osp.exists(f'./tmp/{dataset_name}'):
        os.mkdir(f'./tmp/{dataset_name}')
    
    ################################# Data load and description #################################
    # data load
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset, split = load_dataset_graphcls(path, dataset_name, args)
    
    if isinstance(dataset, Dataset):
        # description
        data_summary = dataset.get_summary()
        num_nodes = dataset.edge_index.max()+1
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
        print(data_summary)
        print(dataset_name, "num_nodes: ", num_nodes, "num_node_features: ", num_node_features, "num_classes: ", num_classes)
    else:
        num_node_features = dataset[0].x.shape[1]
        num_classes = dataset[0].y.shape[0]
        print(dataset_name, "num_node_features: ", num_node_features, "num_classes: ", num_classes)

    ################################# Data PreProcess and Custom Data loader #################################
    # degree calculating
    if dataset_name in ["ZINC"]:
        whole_processed_dataset_save_path = f"./tmp/{dataset_name}/whole_processed_dataset.pt"
        extra_eig_info_dataset_save_path = f"./tmp/{dataset_name}/eig_info.pt"
        extra_x_eig_info_dataset_save_path = f"./tmp/{dataset_name}/x_eig_info.pt"
    else:
        whole_processed_dataset_save_path = osp.join(path, dataset_name, "whole_processed_dataset.pt")
        extra_eig_info_dataset_save_path = osp.join(path, dataset_name, "eig_info.pt")
        extra_x_eig_info_dataset_save_path = osp.join(path, dataset_name, "x_eig_info.pt")
    if args.preprocess_part=="degree":
        all_degrees = []
        all_edge_weights = []
        all_graph_id = []
        all_num_components = []
        
        if dataset_name in ["ZINC", "PPI_unsup"]:
            if is_parallel:
                with parallel_backend(backend="loky", inner_max_num_threads=num_thread_each_core):
                    results_parallel = Parallel(n_jobs=num_jobs)(delayed(process_i_degree_edge_weight)(i, data) for i, data in tqdm(enumerate(dataset), total=len(dataset), desc='(PRE)'))
                all_graph_id, all_degrees, all_edge_weights = list(zip(*results_parallel))
            else:
                for i, data in tqdm(enumerate(dataset), total=len(dataset), desc='(PRE)'):
                    degree = process_degree(data)
                    all_graph_id.append(i)
                    all_degrees.append(degree)

            dataset_dict = {"graph_id": all_graph_id, 
                            "degree": all_degrees, 
                            }
        else:
            for i, data in tqdm(enumerate(dataset), total=len(dataset), desc='(PRE)'):
                degree = process_degree(data)
                if dataset_name in ["MNIST", "CIFAR10"]:
                    edge_weights = data.edge_attr
                else:
                    edge_weights = process_edge_weight(data)
                
                all_graph_id.append(i)
                all_degrees.append(degree)
                all_edge_weights.append(edge_weights)

            dataset_dict = {"dataset": dataset, 
                            "graph_id": all_graph_id, 
                            "train_mask": split['train'], 
                            "degree": all_degrees, 
                            "edge_weights":all_edge_weights, 
                            }

        torch.save(dataset_dict, whole_processed_dataset_save_path)
        # torch.save(dataset_dict, f"./tmp/{dataset_name}/whole_processed_dataset.pt")
    elif args.preprocess_part=="topo":
        all_eigen_values = [] # all non-zero eigen values
        all_eigen_vectors = [] # and corresponded eigen vectors
        all_num_components = []
        if dataset_name in ["ZINC", "PPI_unsup"]:
            if is_parallel:
                with parallel_backend(backend="loky", inner_max_num_threads=num_thread_each_core):
                    results_parallel = Parallel(n_jobs=num_jobs)(delayed(process_topo_eigens)(data, EIGEN_VEC_NUM) for data in tqdm(dataset, total=len(dataset), desc='(Eig)'))
                all_eigen_values, all_eigen_vectors = list(zip(*results_parallel))
            else:
                for i, data in tqdm(enumerate(dataset), total=len(dataset), desc='(Eig)'):
                    eig_value, eig_vector = process_topo_eigens(data)
                    all_eigen_values.append(eig_value)
                    all_eigen_vectors.append(eig_vector)

            eig_info_dict = {"eigen_values": all_eigen_values, 
                            "eigen_vectors": all_eigen_vectors, 
                            }
        else:
            for i, data in tqdm(enumerate(dataset), total=len(dataset), desc='(Eig)'):
                g = to_networkx(data, to_undirected=True)
                num_components = nx.number_connected_components(g)
                all_num_components.append(num_components)

                eig_value, eig_vector = process_topo_eigens(data)
                all_eigen_values.append(eig_value)
                all_eigen_vectors.append(eig_vector)
            
            eig_info_dict = {"eigen_values": all_eigen_values, 
                            "eigen_vectors": all_eigen_vectors, 
                            "num_component": all_num_components,
                            }
        torch.save(eig_info_dict, extra_eig_info_dataset_save_path)
        # torch.save(eig_info_dict, f"./tmp/{dataset_name}/eig_info.pt")

    elif args.preprocess_part=="xtopo":
        X_EIGEN_VEC_NUM = min(num_node_features, EIGEN_VEC_NUM)
        all_x_eigen_values = []
        all_x_eigen_vectors_U = []
        all_x_eigen_vectors_V = []
        if is_parallel:
            with parallel_backend(backend="loky", inner_max_num_threads=num_thread_each_core):
                results_parallel = Parallel(n_jobs=num_jobs)(delayed(process_xtopo_eigens)(data, X_EIGEN_VEC_NUM) for data in tqdm(dataset, total=len(dataset), desc='(X_Eig)'))
            all_x_eigen_values, all_x_eigen_vectors_U, all_x_eigen_vectors_V = list(zip(*results_parallel))
        else:
            for i, data in tqdm(enumerate(dataset), total=len(dataset), desc='(X_Eig)'):
                x_eig_value, x_eig_vector_U, x_eig_vector_V = process_xtopo_eigens(data, X_EIGEN_VEC_NUM)
                all_x_eigen_values.append(x_eig_value)
                all_x_eigen_vectors_U.append(x_eig_vector_U)
                all_x_eigen_vectors_V.append(x_eig_vector_V)
        x_eig_info_dict = {"x_eigen_values": all_x_eigen_values,
                            "x_eigen_vectors_U": all_x_eigen_vectors_U, 
                            "x_eigen_vectors_V": all_x_eigen_vectors_V,
                            }
        torch.save(x_eig_info_dict, extra_x_eig_info_dataset_save_path)
        # torch.save(x_eig_info_dict, f"./tmp/{dataset_name}/x_eig_info.pt") 
    elif args.preprocess_part=="load":
        dataset_dict = torch.load(whole_processed_dataset_save_path)
        eig_info_dict = torch.load(extra_eig_info_dataset_save_path)
        x_eig_info_dict = torch.load(extra_x_eig_info_dataset_save_path)
        for k,v in dataset_dict.items():
            print(k, len(v))
        for k,v in eig_info_dict.items():
            print(k, len(v), v[0].shape)
        for k,v in x_eig_info_dict.items():
            print(k, len(v), v[0].shape)
    
if __name__ == '__main__':
    main()
    print("end!")