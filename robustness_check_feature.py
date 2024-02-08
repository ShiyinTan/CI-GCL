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
import warnings
import socket

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader, Dataset
from GCL.models.mymodels import GraphNodeEncoder, GraphEncoder, Predictor
from GCL.utils import compute_infonce, cluster_get, CustomDataLoader, compute_cluster_constrain_loss, k_fold, process_degree, process_topo_eigens, process_xtopo_eigens, save_model_evaluator, \
    load_model_evaluator, str_to_bool, add_extra_pos_mask, degree, topo_cluster_labels_get, compute_spectral_topo_loss, compute_spectral_feature_loss, print_memory_usage
from sklearn.metrics import f1_score
from datetime import datetime
from general_data_loader import load_dataset_graphcls, get_split_mask, CombinedDataset
from torch_geometric.utils import scatter, to_networkx, to_dense_adj, get_laplacian
from sklearn.model_selection import KFold
from torch_geometric.utils import coalesce, batched_negative_sampling, to_undirected, dropout_edge
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_degree", type=str_to_bool, default=True)
    parser.add_argument("--use_spectral_fea", type=str_to_bool, default=True)
    parser.add_argument("--use_spectral_topo", type=str_to_bool, default=True)
    parser.add_argument("--linear", type=str_to_bool, default=False)
    parser.add_argument("--old_version", type=str_to_bool, default=False)

    parser.add_argument("--mode", type=str, default="robust_fea", choices=["unsup", "semisup", "robust"])
    parser.add_argument("--semi_sup_rate", type=float, default=0.1) # 0.1, or 0.01
    parser.add_argument("--dataset_name", type=str, default="PROTEINS")
    parser.add_argument("--epoch_select", type=str, default="test_max")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128) # 512
    parser.add_argument("--epochs", type=int, default=200) # 200
    parser.add_argument("--finetune_epochs", type=int, default=40) # 200
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--par_id", type=int, default=6)

    args = parser.parse_args()
    return args


def add_white_noise(X, noise_rate, noise_level=1):
    m, n = X.shape
    X_with_noise = X.clone()
    k = int(m*n*noise_rate)

    mask = torch.zeros(m,n, device=X.device)
    mask[torch.randint(0, m, (k,), device=X.device), torch.randint(0, n, (k,), device=X.device)] = 1
    noise = torch.randn((m,n), device=X.device) * noise_level * mask
    X_with_noise = X_with_noise + noise
    
    return X_with_noise

def robust_test(encoder_model, dataloader, args, split, noise_rate=0.04, best_evaluator=None, device='cpu'):
    encoder_model.eval()
    x = []
    y = []
    for data, _ in dataloader:
        data = data.to(device)
        num_nodes = data.batch.size(0)
        if args.use_degree:
            if data.x is None:
                data.x = data.degree
            # else:
            #     data.x = torch.concat([data.x, data.degree], dim=1)
        else:
            if data.x is None:
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        new_x = add_white_noise(data.x, noise_rate)

        edge_weights = torch.ones((data.edge_index.shape[1], 1), device=device)

        z, g = encoder_model(new_x, data.edge_index, data.batch, edge_weights=edge_weights, mode="eval")
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if best_evaluator==None:
            # result, best_evaluator = SVMEvaluator(linear=False, setting=setting)(x, y, split)
            result, best_evaluator = SVMEvaluator(linear=args.linear, epoch_select=args.epoch_select)(x, y, split)
        else:
            x_test = x[split['test']].detach().cpu().numpy()
            y_test = y[split['test']].detach().cpu().numpy()
            test_macro = f1_score(y_test, best_evaluator.predict(x_test), average='macro')
            test_micro = f1_score(y_test, best_evaluator.predict(x_test), average='micro')
            result = {
                'micro_f1': test_micro,
                'macro_f1': test_macro,
            }
    return result, best_evaluator



def main():

    hostname = socket.gethostname()
    print("Current hostname:", hostname)

    EIGEN_VEC_NUM = 4
    load_dataset = True
    x_eigen_load_dataset = True
    not_load_exist_model = False
    val_gap=10
    save_checkpoint_gap=50
    args = get_config()

    batch_size = args.batch_size
    epochs = args.epochs
    dataset_name = args.dataset_name
    finetune_epochs = args.finetune_epochs

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    current_time = datetime.now()
    print("Current time:", current_time)
    # print settings
    print("dataset: ", args.dataset_name, "device: ", device, ", mode: ", args.mode, args.linear)
    
    if not osp.exists('./log'):
        os.mkdir('./log')
    if not osp.exists(f'./log/{dataset_name}'):
        os.mkdir(f'./log/{dataset_name}')
    logging_path = "./log/{}/results_{}.log".format(dataset_name, args.mode)
    logging.basicConfig(filename=logging_path, level=logging.DEBUG, format='%(asctime)s %(message)s')

    if not osp.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not osp.exists(f'./checkpoints/{dataset_name}'):
        os.mkdir(f'./checkpoints/{dataset_name}')
    
    # data load
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset, split = load_dataset_graphcls(path, dataset_name, args)

    # description
    if isinstance(dataset, Dataset):
        # description
        data_summary = dataset.get_summary()
        num_nodes = dataset.edge_index.max()+1
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
        y = dataset.y
        print(data_summary)
        print(dataset_name, "num_nodes: ", num_nodes, "num_node_features: ", num_node_features, "num_classes: ", num_classes)
    else:
        num_node_features = dataset[0].x.shape[1]
        num_classes = dataset[0].y.shape[0]
        y = torch.asarray([data.y for data in dataset])
        print(dataset_name, "num_node_features: ", num_node_features, "num_classes: ", num_classes)  
    
    # degree calculating
    whole_processed_dataset_save_path = osp.join(path, dataset_name, "whole_processed_dataset.pt")
    if osp.exists(whole_processed_dataset_save_path) and load_dataset:
        dataset_dict = torch.load(whole_processed_dataset_save_path)
    else:
        all_degrees = []
        all_edge_weights = []
        all_graph_id = []
        all_num_components = []
        all_eigen_values = [] # all non-zero eigen values
        all_eigen_vectors = [] # and corresponded eigen vectors
        with tqdm(total=len(dataset), desc='(PRE)') as pbar:
            for i, data in enumerate(dataset):
                node_degree = process_degree(data)
                all_degrees.append(node_degree)
                pbar.update()

                edge_weights = torch.ones((data.edge_index.shape[1],))
                all_edge_weights.append(edge_weights)

                all_graph_id.append(i)
        # combine dataset
        dataset_dict = {"dataset": dataset, 
                        "graph_id": all_graph_id, 
                        "train_mask": torch.zeros(len(dataset), dtype=bool), 
                        "degree": all_degrees, 
                        "edge_weights":all_edge_weights, 
                        }
        torch.save(dataset_dict, whole_processed_dataset_save_path)
    
    # eigen info calculate
    if not osp.exists(osp.join(path, dataset_name)):
        os.mkdir(osp.join(path, dataset_name))
    extra_eig_info_dataset_save_path = osp.join(path, dataset_name, "eig_info.pt")
    if osp.exists(extra_eig_info_dataset_save_path) and load_dataset:
        eig_info_dict = torch.load(extra_eig_info_dataset_save_path)
    else:
        all_eigen_values = [] # all non-zero eigen values
        all_eigen_vectors = [] # and corresponded eigen vectors
        with tqdm(total=len(dataset), desc='(Eig)') as pbar:
            for i, data in enumerate(dataset):
                g = to_networkx(data, to_undirected=True)
                num_components = nx.number_connected_components(g)
                all_num_components.append(num_components)
                del g

                eig_value, eig_vector = process_topo_eigens(data, EIGEN_VEC_NUM, num_components)
                all_eigen_values.append(eig_value)
                all_eigen_vectors.append(eig_vector)
                
                pbar.update()

        eig_info_dict = {"eigen_values": all_eigen_values, 
                         "eigen_vectors": all_eigen_vectors, 
                         "num_component": all_num_components,
                        }
        torch.save(eig_info_dict, extra_eig_info_dataset_save_path)
    dataset_dict.update({'num_component': eig_info_dict["num_component"]})
    dataset_dict.update({'eigen_values': eig_info_dict["eigen_values"]})
    dataset_dict.update({'eigen_vectors': eig_info_dict["eigen_vectors"]})


    if num_node_features >= 1:
        extra_x_eig_info_dataset_save_path = osp.join(path, dataset_name, "x_eig_info.pt")
        X_EIGEN_VEC_NUM = min(num_node_features, EIGEN_VEC_NUM)
        if osp.exists(extra_x_eig_info_dataset_save_path) and x_eigen_load_dataset:
            x_eig_info_dict = torch.load(extra_x_eig_info_dataset_save_path)
        else:
            # feature x's sigular vector: U, S, V
            all_x_eigen_values = [] # all non-zero eigen values
            all_x_eigen_vectors_U = []
            all_x_eigen_vectors_V = [] # and corresponded eigen vectors
            with tqdm(total=len(dataset), desc='(X_Eig)') as pbar:
                for i, data in enumerate(dataset):
                    x_eig_value, x_eig_vector_U, x_eig_vector_V = process_xtopo_eigens(data, X_EIGEN_VEC_NUM)
                    all_x_eigen_values.append(x_eig_value)
                    all_x_eigen_vectors_U.append(x_eig_vector_U)
                    all_x_eigen_vectors_V.append(x_eig_vector_U)

                    pbar.update()
            x_eig_info_dict = {"x_eigen_values": all_x_eigen_values,
                               "x_eigen_vectors_U": all_x_eigen_vectors_U, 
                               "x_eigen_vectors_V": all_x_eigen_vectors_V,
                               }
            torch.save(x_eig_info_dict, extra_x_eig_info_dataset_save_path)
        dataset_dict.update({'x_eigen_values': x_eig_info_dict["x_eigen_values"]})
        dataset_dict.update({'x_eigen_vectors_U': x_eig_info_dict["x_eigen_vectors_U"]})
        dataset_dict.update({'x_eigen_vectors_V': x_eig_info_dict["x_eigen_vectors_V"]})
    
    if (num_node_features>=1):
        dataset_dict_keys = ["dataset", "graph_id", "train_mask", "degree", "edge_weights", 
                            'num_component', "eigen_values", "eigen_vectors", 
                            "x_eigen_values", "x_eigen_vectors_U", "x_eigen_vectors_V"]
    else:
        dataset_dict_keys = ["dataset", "graph_id", "train_mask", "degree", "edge_weights", 
                            'num_component', "eigen_values", "eigen_vectors"]
    dataset_dict = {key: dataset_dict[key] for key in dataset_dict_keys}

    combined_dataset = CombinedDataset(**dataset_dict)
    dataloader = CustomDataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=True, has_node_features=(num_node_features>=1))


    if args.use_degree:
        input_fea_dim = max(num_node_features, 1)
    else:
        input_fea_dim = max(num_node_features, 1)

    # define augmentations
    aug1 = A.Identity()
    if num_node_features >= 1:
        x_eigen_distance_input_dim = X_EIGEN_VEC_NUM
        leA_FD = LeA.LearnableFeatureDroppingBySpectral(input_dim=x_eigen_distance_input_dim, hidden_dim=128).to(device)
        aug1 = leA_FD
    rand_FM = A.FeatureMasking(pf=0.1)
    rand_EA = A.EdgeAdding(pe=0.3)
    rand_ED = A.EdgeRemoving(pe=0.3)

    edge_attr_input_dim = input_fea_dim*2 + dataset_dict['eigen_vectors'][0].shape[1]
    leA_ED = LeA.LearnableEdgeDropping(input_dim=edge_attr_input_dim, hidden_dim=128, temp=1.0).to(device) # edge_attr: subg feas, eigen_vecs
    leA_EA = LeA.LearnableEdgeAdding(input_dim=edge_attr_input_dim, hidden_dim=128, sample_edges_ratio=0.2).to(device)
    leA_EP = LeA.LearnableEdgePerturbation(input_dim_drop=edge_attr_input_dim, input_dim_add=edge_attr_input_dim, hidden_dim=128, sample_edges_ratio=0.2).to(device)
    aug2 = leA_EP
    augs_type = [type(aug1).__name__, type(aug2).__name__]
    aug1_no_spec = aug2_no_spec = False
    if augs_type[0] == "LearnableFeatureDroppingBySpectral":
        if args.use_spectral_fea==False:
            augs_type[0] += "WithoutSpectral"
            aug1_no_spec = True
            val_gap=epochs//2
    elif augs_type[0] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
        if args.use_spectral_topo==False:
            augs_type[0] += "WithoutSpectral"
            aug1_no_spec = True
            val_gap=epochs
    
    if augs_type[1] == "LearnableFeatureDroppingBySpectral":
        if args.use_spectral_fea==False:
            augs_type[1] += "WithoutSpectral"
            aug2_no_spec = True
            val_gap=epochs//2
    elif augs_type[1] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
        if args.use_spectral_topo==False:
            augs_type[1] += "WithoutSpectral"
            aug2_no_spec = True
            val_gap=epochs
    not_load_exist_model = (aug1_no_spec and aug2_no_spec)
    if not_load_exist_model:
        val_gap=epochs

    print(augs_type)

    checkpoints_path = "./checkpoints/{}/{}_{}".format(dataset_name, augs_type[0], augs_type[1])
    checkpoints_path_2 = "./checkpoints/{}".format(dataset_name) # for some reason, we have backup checkpoints, in case the checkpoints_path doesn't exist
    
    # print_memory_usage()
    ################################# FineTune #################################
    best_acc = 0
    print("-"*40+f"Starting"+"-"*40)
    test_acc = "nan"

    # load encoder model
    # if osp.exists(osp.join(checkpoints_path, f"best_encoder_best_0.pt")):
    #     encoder_model = torch.load(osp.join(checkpoints_path, f"best_encoder_best_0.pt"), map_location=device)
    # elif osp.exists(osp.join(checkpoints_path_2, f"best_encoder_0.pt")):
    #     encoder_model = torch.load(osp.join(checkpoints_path_2, f"best_encoder_0.pt"), map_location=device)
    # else:
    #     raise FileNotFoundError

    encoder_model, evaluator, predictor = load_model_evaluator(checkpoints_path, par_id=args.par_id, device=device, old_version=False)
    # if encoder_model==None:
    # if dataset_name in ["IMDB-BINARY", "COLLAB"]:
        # encoder_model, evaluator, predictor = load_model_evaluator(checkpoints_path_2, par_id=args.par_id, device=device, old_version=True)
        # print("load_old_checkpoint")
    print(type(evaluator))
    encoder_model = encoder_model.to(device)
    
    all_results = {}

    for noise_rate in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        
        all_trail_results = {'micro_f1': [],
                             'macro_f1': []}
        for trial in range(5):
            fold_all_results = {'micro_f1': [], 
                                'macro_f1': []}
            for fold_i, (train_idx, val_idx, test_idx) in enumerate(k_fold(len(dataset), epoch_select=args.epoch_select)):
                split = {'train': torch.asarray(train_idx, device=device, dtype=torch.int64), 
                        'valid': torch.asarray(val_idx, device=device, dtype=torch.int64),
                        'test':  torch.asarray(test_idx, device=device, dtype=torch.int64)}

                if noise_rate==0:
                    test_result, best_evaluator = robust_test(encoder_model, dataloader, args, split, 
                                                              noise_rate=noise_rate, 
                                                              best_evaluator=None, device=device)
                else:
                    test_result, best_evaluator = robust_test(encoder_model, dataloader, args, split, 
                                                              noise_rate=noise_rate, 
                                                              best_evaluator=None, device=device)

                fold_all_results['micro_f1'].append(test_result['micro_f1'])
                fold_all_results['macro_f1'].append(test_result['macro_f1'])
            all_trail_results['micro_f1'].append(np.mean(fold_all_results['micro_f1']))
            all_trail_results['macro_f1'].append(np.mean(fold_all_results['macro_f1']))
        
        test_result = {'micro_f1': np.mean(all_trail_results['micro_f1']),
                       'macro_f1': np.mean(all_trail_results['macro_f1'])}
        test_max_result = {'micro_f1': np.max(all_trail_results['micro_f1']),
                       'macro_f1': np.max(all_trail_results['macro_f1'])}
        test_min_result = {'micro_f1': np.min(all_trail_results['micro_f1']),
                       'macro_f1': np.min(all_trail_results['macro_f1'])}
        
        all_results[noise_rate] = test_result
        
        print(f"noise_rate {noise_rate}: {test_result}, max: {test_max_result}, min: {test_min_result}")
        logging.info("Aug_1: {}, Aug_2: {}, noise_rate: {}: avg {}, max {}, min {}".format(augs_type[0], augs_type[1], noise_rate, test_result, test_max_result, test_min_result))
    # results
    # print(f'(E): Avg all folds test F1Mi={avg_results["micro_f1"]:.4f}, F1Ma={avg_results["macro_f1"]:.4f}')
    logging.info("Aug_1: {}, Aug_2: {}, all_results: {}".format(augs_type[0], augs_type[1], all_results))
    print("-"*90)

if __name__ == '__main__':
    main()