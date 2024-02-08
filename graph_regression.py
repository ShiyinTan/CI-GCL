from pyparsing import NoMatch
import torch
import os
import os.path as osp
from GCL.eval.svm import SVREvaluator
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
from GCL.utils import (compute_infonce, cluster_get, CustomDataLoader, compute_cluster_constrain_loss, k_fold, process_degree, process_topo_eigens, 
                       process_xtopo_eigens, save_model_evaluator, load_model_evaluator, str_to_bool, add_extra_pos_mask, degree, 
                       topo_cluster_labels_get, compute_spectral_topo_loss, compute_spectral_feature_loss, print_memory_usage, print_args)
from sklearn.metrics import f1_score
from datetime import datetime
from general_data_loader import load_dataset_graphcls, get_split_mask, CombinedDataset
from torch_geometric.utils import scatter, to_networkx, to_dense_adj, get_laplacian
from sklearn.model_selection import KFold


def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_degree", type=str_to_bool, default=True)
    parser.add_argument("--use_spectral_fea", type=str_to_bool, default=True)
    parser.add_argument("--use_spectral_topo", type=str_to_bool, default=True)

    parser.add_argument("--mode", type=str, default="regression", choices=["unsup", "semisup", "regression"])
    parser.add_argument("--semi_sup_rate", type=float, default=0.1) # 0.1, or 0.01
    parser.add_argument("--dataset_name", type=str, default="esol")
    parser.add_argument("--epoch_select", type=str, default="val_max")
    parser.add_argument("--linear", type=str_to_bool, default=True)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512) # 512
    parser.add_argument("--epochs", type=int, default=200) # 200
    parser.add_argument("--finetune_epochs", type=int, default=40) # 200
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--par_id", type=int, default=0)

    args = parser.parse_args()
    return args


def train(encoder_model, predictor, dataloader, optimizer, args, pretrain=True, augs_type=None, device='cpu'):
    encoder_model.train()
    epoch_loss = 0
    epoch_predict_loss = 0
    epoch_spectral_topo_loss = 0
    epoch_spectral_feature_loss = 0
    encoder_exec_time = 0
    spectral_exec_time = 0
    for data, extra_info_dict in dataloader:
        """
        data.cluster_labels: list of tensor: [n_nodes, 1]
        data.cluster_centrics: list of tensor: [n_cluster, fea_dim]
        data.cluster_num: list of int
        """
        data = data.to(device)

        eigen_values = extra_info_dict["eigen_values"]
        eigen_values = [ev.to(device) for ev in eigen_values]
        if "x_eigen_values" in extra_info_dict:
            x_eigen_values = extra_info_dict["x_eigen_values"]
            x_eigen_vectors_U = extra_info_dict["x_eigen_vectors_U"]
            x_eigen_vectors_V = extra_info_dict["x_eigen_vectors_V"]

            x_eigen_values = [ev.to(device) for ev in x_eigen_values]
            x_eigen_vectors_U = [elem.to(device) for elem in x_eigen_vectors_U]
            x_eigen_vectors_V = [elem.to(device) for elem in x_eigen_vectors_V]

        else:
            x_eigen_values = None
            x_eigen_vectors_U = None
            x_eigen_vectors_V = None
        
        # eigen_vectors = [ev.to(device) for ev in eigen_vectors]
        assert len(eigen_values)==(data.batch.max()+1)
        assert data.num_component.shape[0]==(data.batch.max()+1)
        assert data.num_nodes==data.eigen_vectors.shape[0]
        # make sure edge_weights [num_edges, 1]
        if 'edge_weights' in data.keys():
            edge_weights = data.edge_weights.reshape(-1,1)
        else:
            edge_weights = torch.ones((data.edge_index.shape[1], 1), device=device)

        if 'edge_attr' in data.keys():
            gnn_edge_attr = data.edge_attr.float()
        else:
            gnn_edge_attr = torch.zeros((data.edge_index.shape[1], 1), dtype=float, device=device)

        batch_size = len(data)
        num_nodes = data.batch.size(0)
        optimizer.zero_grad()

        if args.use_degree:
            if data.x is None:
                data.x = data.degree
            # else:
            #     data.x = torch.concat([data.x, data.degree], dim=1)
        else:
            if data.x is None:
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        
        # edge attr construct
        edge_attr = torch.pow(data.eigen_vectors[data.edge_index[0]] - data.eigen_vectors[data.edge_index[1]], 2)
        edge_attr = torch.cat([edge_attr, torch.concat([data.x[data.edge_index[0]], data.x[data.edge_index[1]]], dim=1)], dim=1)

        if x_eigen_values!=None:
            all_x_eigen_distance = []
            for i in range(len(x_eigen_vectors_U)): # for each single graph
                U_expanded = x_eigen_vectors_U[i].unsqueeze(1) # [n_nodes, 1, n_eigen_vecs]
                V_expanded = x_eigen_vectors_V[i].unsqueeze(0) # [1, n_feas, n_eigen_vecs]

                all_x_eigen_distance.append(torch.pow(U_expanded - V_expanded, 2)) # [n_nodes, n_feas, n_eigen_vecs]
            x_eigen_distance = torch.concat(all_x_eigen_distance)
            # print(U_expanded.shape, V_expanded.shape, x_eigen_distance.shape, data.num_nodes, data.x.shape)
        else:
            x_eigen_distance = None
        
        # print(encoder_model.device, encoder_model.aug2.device)
        start_time = time.time()
        _, g, _, g1, _, g2, augs_g = encoder_model(data.x.float(), data.edge_index, data.batch, edge_weights=edge_weights, 
                                                   edge_attr=edge_attr, gnn_edge_attr=gnn_edge_attr,
                                                   eigen_values=eigen_values, eigen_vectors=data.eigen_vectors, x_eigen_distance=x_eigen_distance)
        g1, g2 = [encoder_model.graph_node_encoder.project(g) for g in [g1, g2]]
        # loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        encoder_exec_time += time.time() - start_time

        # denote positive samples, only for semi-supervise
        num_samples = len(data)
        pos_mask = torch.eye(num_samples, dtype=bool, device=device)
        if pretrain == False:
            pos_mask = add_extra_pos_mask(pos_mask, data, data.train_mask)
        
        ssl_loss = compute_infonce(g1, g2, pos_mask)
        epoch_loss += ssl_loss.item()

        if pretrain == False:
            trues = data.y[data.train_mask]
            preds1 = predictor(g1)[data.train_mask]
            preds2 = predictor(g2)[data.train_mask]
            predict_loss1 = F.nll_loss(preds1, trues, reduction='sum')/(trues.shape[0])
            predict_loss2 = F.nll_loss(preds2, trues, reduction='sum')/(trues.shape[0])
            predict_loss = (predict_loss1 + predict_loss2)/2
            epoch_predict_loss += predict_loss.item()
        else:
            predict_loss = 0

        start_time = time.time()
        # spectral topological regularization
        aug_g1, aug_g2 = augs_g
        spectral_topo_loss = 0
        if args.use_spectral_topo:
            if augs_type[0] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
                spectral_topo_loss += compute_spectral_topo_loss(data.edge_index, edge_weights, aug_g1, eigen_values, data.num_component, data.batch, augs_type[0])
            if augs_type[1] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
                spectral_topo_loss += compute_spectral_topo_loss(data.edge_index, edge_weights, aug_g2, eigen_values, data.num_component, data.batch, augs_type[1])
            spectral_topo_loss = spectral_topo_loss/(data.batch.max()+1)
            epoch_spectral_topo_loss += spectral_topo_loss.item()

        # spectral feature regularization
        spectral_feature_loss = 0
        if args.use_spectral_fea:
            if x_eigen_values!=None:
                if augs_type[0] in ["LearnableFeatureDroppingBySpectral"]:
                    spectral_feature_loss += compute_spectral_feature_loss(aug_g1, x_eigen_values, data.batch)/(data.batch.max()+1)
                    epoch_spectral_feature_loss += spectral_feature_loss.item()
                if augs_type[1] in ["LearnableFeatureDroppingBySpectral"]:
                    spectral_feature_loss += compute_spectral_feature_loss(aug_g2, x_eigen_values, data.batch)/(data.batch.max()+1)
                    epoch_spectral_feature_loss += spectral_feature_loss.item()
        spectral_exec_time += time.time() - start_time

        loss = 0.2 * (ssl_loss + 0.8*spectral_topo_loss + 0.3*spectral_feature_loss) + predict_loss

        loss.backward() # on


        optimizer.step()

    return epoch_loss, epoch_predict_loss, epoch_spectral_topo_loss, epoch_spectral_feature_loss


def test(encoder_model, dataloader, args, split, epoch_select='val_max', best_evaluator=None, device='cpu'):
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
        if 'edge_weights' in data.keys():
            edge_weights = data.edge_weights.reshape(-1,1)
        else:
            edge_weights = torch.ones((data.edge_index.shape[1], 1), device=device)
        
        if 'edge_attr' in data.keys():
            gnn_edge_attr = data.edge_attr.float()
        else:
            gnn_edge_attr = torch.zeros((data.edge_index.shape[1], 1), dtype=float, device=device)

        z, g = encoder_model(data.x.float(), data.edge_index, data.batch, edge_weights=edge_weights, gnn_edge_attr=gnn_edge_attr, mode="eval")
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0).squeeze()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 忽略特定类型的警告
        if best_evaluator==None:
            # result, best_evaluator = SVREvaluator(linear=False, epoch_select=epoch_select)(x, y, split) # "lipo"
            result, best_evaluator = SVREvaluator(linear=args.linear, max_iter=10000, epoch_select=epoch_select)(x, y, split)
        else:
            x_test = x[split['test']].detach().cpu().numpy()
            y_test = y[split['test']].detach().cpu().numpy()
            test_rmse = f1_score(y_test, best_evaluator.predict(x_test), squared=False)
            result = {
                'rmse': test_rmse,
            }
    return result, best_evaluator



def main():

    hostname = socket.gethostname()
    print("Current hostname:", hostname)

    EIGEN_VEC_NUM = 50
    load_dataset = True
    x_eigen_load_dataset = True
    not_load_exist_model = False
    val_gap=10
    save_checkpoint_gap=50
    args = get_config()
    if args.dataset_name=="lipo":
        val_gap=2

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
    print_args(args)
    print("dataset: ", args.dataset_name, "device: ", device, ", mode: ", args.mode, ", linear: ", args.linear)
    
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
    if not osp.exists(f'./checkpoints/{dataset_name}/{args.mode}'):
        os.mkdir(f'./checkpoints/{dataset_name}/{args.mode}')
    # checkpoints_path = "./checkpoints/{}".format(dataset_name)
    
    # data load
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset, split = load_dataset_graphcls(path, dataset_name, args)

    # description
    if isinstance(dataset, Dataset):
        # description
        data_summary = dataset.get_summary()
        num_nodes = dataset.edge_index.max()+1
        num_node_features = dataset.num_node_features
        
        if dataset.y.dim()==2:
            num_classes = dataset.y.shape[1]
        elif dataset.y.dim()==1:
            num_classes = dataset.y.shape[0]
        
        if "edge_attr" in dataset[0].keys() and dataset_name!="PPI_sup":
            num_edge_features = dataset[0].edge_attr.shape[1]
        else:
            num_edge_features = 0
        
        print(data_summary)
        print(dataset_name, "num_nodes: ", num_nodes, "num_node_features: ", num_node_features, 
              "num_edge_features: ", num_edge_features, "num_classes: ", num_classes)
    else:
        num_node_features = dataset[0].x.shape[1]
        num_classes = dataset[0].y.shape[0]
        if "edge_attr" in dataset[0].keys() and dataset_name!="PPI_sup":
            num_edge_features = dataset[0].edge_attr.shape[1]
        else:
            num_edge_features = 0
        print(dataset_name, "num_nodes: ", num_nodes, "num_node_features: ", num_node_features, 
              "num_edge_features: ", num_edge_features, "num_classes: ", num_classes)
    
    # if num_node_features>=1:
    #     x_norm_part = torch.linalg.norm(dataset.x.float(), dim=0).to(device)
    #     x_norm_part = torch.clip(x_norm_part, min=1e-6)
    # else:
    #     x_norm_part = 1
    x_norm_part = 1


    # degree calculating
    if not osp.exists(osp.join(path, dataset_name)):
        os.mkdir(osp.join(path, dataset_name))
    dataset_dict = {"dataset": dataset}
    if EIGEN_VEC_NUM==50:
        whole_processed_dataset_save_path = osp.join(path, dataset_name, "whole_processed_dataset.pt")
    else:
        whole_processed_dataset_save_path = osp.join(path, dataset_name, f"whole_processed_dataset_{EIGEN_VEC_NUM}.pt")
    if osp.exists(whole_processed_dataset_save_path) and load_dataset:
        id_degree_dict = torch.load(whole_processed_dataset_save_path)
    else:
        all_degrees = []
        all_edge_weights = []
        all_graph_id = []
        all_eigen_values = [] # all non-zero eigen values
        all_eigen_vectors = [] # and corresponded eigen vectors
        with tqdm(total=len(dataset), desc='(PRE)') as pbar:
            for i, data in enumerate(dataset):
                node_degree = process_degree(data)
                if node_degree.shape[0]==0:
                    node_degree = torch.zeros((data.num_nodes, 1))
                assert node_degree.shape[0] == data.num_nodes, f"{node_degree.shape}, {data.num_nodes}"
                all_degrees.append(node_degree)

                edge_weights = torch.ones((data.edge_index.shape[1],))
                all_edge_weights.append(edge_weights)

                all_graph_id.append(i)

                pbar.update()
        # combine dataset
        id_degree_dict = {"graph_id": all_graph_id, 
                          "degree": all_degrees, 
                          "edge_weights":all_edge_weights, 
                        }
        torch.save(id_degree_dict, whole_processed_dataset_save_path)
    dataset_dict.update({'graph_id': id_degree_dict["graph_id"]})
    dataset_dict.update({'degree': id_degree_dict["degree"]})
    dataset_dict.update({'train_mask': torch.zeros(len(dataset), dtype=bool)})
    dataset_dict.update({'edge_weights': id_degree_dict['edge_weights']})


    # eigen info calculate
    if not osp.exists(osp.join(path, dataset_name)):
        os.mkdir(osp.join(path, dataset_name))
    if EIGEN_VEC_NUM==50:
        extra_eig_info_dataset_save_path = osp.join(path, dataset_name, "eig_info.pt")
    else:
        extra_eig_info_dataset_save_path = osp.join(path, dataset_name, f"eig_info_{EIGEN_VEC_NUM}.pt")
    if osp.exists(extra_eig_info_dataset_save_path) and load_dataset:
        eig_info_dict = torch.load(extra_eig_info_dataset_save_path)
    else:
        all_eigen_values = [] # all non-zero eigen values
        all_eigen_vectors = [] # and corresponded eigen vectors
        all_num_components = []
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
        if EIGEN_VEC_NUM==50:
            extra_x_eig_info_dataset_save_path = osp.join(path, dataset_name, "x_eig_info.pt")
        else:
            extra_x_eig_info_dataset_save_path = osp.join(path, dataset_name, f"x_eig_info_{EIGEN_VEC_NUM}.pt")
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
                    x_eig_value, x_eig_vector_U, x_eig_vector_V = process_xtopo_eigens(data, X_EIGEN_VEC_NUM, x_norm_part=x_norm_part.cpu())
                    all_x_eigen_values.append(x_eig_value)
                    all_x_eigen_vectors_U.append(x_eig_vector_U)
                    all_x_eigen_vectors_V.append(x_eig_vector_V)

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

    checkpoints_path = "./checkpoints/{}/{}/{}_{}".format(dataset_name, args.mode, augs_type[0], augs_type[1])
    if not osp.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    
    # define encoder
    gconv = GraphNodeEncoder(input_channels=input_fea_dim, hidden_channels=256, num_layers=2, edge_dim=num_edge_features).to(device)
    encoder_model = GraphEncoder(graph_node_encoder=gconv, augmentor=(aug1, aug2), x_norm_part=x_norm_part).to(device)
    predictor = Predictor(gconv.project_dim, num_classes).to(device) # num_classes = len(dataset.y.unique())
    
    # define optimizer
    params_for_optimizer = [{'params': encoder_model.parameters()}]
    if isinstance(aug1, LeA.Compose):
        for aug_elem in aug1.augmentors:
            if isinstance(aug_elem, LeA.LearnableAugmentor):
                params_for_optimizer.append({"params": aug_elem.parameters()})
    elif isinstance(aug1, LeA.LearnableAugmentor):
        print(augs_type[0], "learnable")
        params_for_optimizer.append({"params": aug1.parameters()})
    if isinstance(aug2, LeA.Compose):
        for aug_elem in aug2.augmentors:
            if isinstance(aug_elem, LeA.LearnableAugmentor):
                params_for_optimizer.append({"params": aug_elem.parameters()})
    elif isinstance(aug2, LeA.LearnableAugmentor):
        print(augs_type[1], "learnable")
        params_for_optimizer.append({"params": aug2.parameters()})
    optimizer = Adam(params_for_optimizer, lr=0.01)

    # training
    split = get_split(num_samples=len(dataset), train_ratio=0.8, test_ratio=0.1)
    best_rmse = 10000
    print("-"*40+f"Starting"+"-"*40)
    all_results = {'rmse': []}
    test_rmse = "nan"
    # print_memory_usage()
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1, epochs+1):
            # print_memory_usage()
            loss, predict_loss, spectral_topo_loss, spectral_feature_loss = train(encoder_model, predictor, dataloader, optimizer, args, augs_type=augs_type, device=device)
            if epoch % save_checkpoint_gap == 0:
                save_model_evaluator(checkpoints_path, encoder_model, post_fix=epoch, par_id=args.par_id)
            
            if epoch % val_gap == 0:
                test_result, best_evaluator = test(encoder_model, dataloader, args, split, epoch_select=args.epoch_select, best_evaluator=None, device=device)
                test_rmse = test_result['rmse']
                all_results['rmse'].append(test_result['rmse'])

                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    save_model_evaluator(checkpoints_path, encoder_model, best_evaluator, par_id=args.par_id)

            pbar.set_postfix({'loss': loss, 
                            'spec_topo': spectral_topo_loss, 
                            'spec_feas': spectral_feature_loss,
                            'val_rmse': test_rmse})
            pbar.update()

    best_idx = np.argmin(all_results['rmse'])
    best_results = {'rmse': all_results['rmse'][best_idx]}
    best_max_result = {'rmse': np.max(all_results['rmse'])}
    best_min_result = {'rmse': np.min(all_results['rmse'])}
    # results
    print(f'(E): Best test RMSE={best_results["rmse"]:.4f}')
    logging.info("Aug_1: {}, Aug_2: {}, avg_results: {}, max: {}, min: {}"\
                 .format(augs_type[0], augs_type[1], best_results, best_max_result, best_min_result))
    print("-"*90)

if __name__ == '__main__':
    main()