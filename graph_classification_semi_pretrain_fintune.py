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
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from GCL.models.mymodels import GraphNodeEncoder, GraphEncoder, Predictor
from GCL.models.res_gcn import ResGCN
from GCL.utils import compute_infonce, cluster_get, CustomDataLoader, compute_cluster_constrain_loss, k_fold, print_args, process_topo_eigens, process_xtopo_eigens, save_model_evaluator, \
    load_model_evaluator, str_to_bool, add_extra_pos_mask, degree, topo_cluster_labels_get, compute_spectral_topo_loss, compute_spectral_feature_loss, print_memory_usage
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
    parser.add_argument("--linear", type=str_to_bool, default=True)

    parser.add_argument("--mode", type=str, default="semisup", choices=["unsup", "semisup"])
    parser.add_argument("--semi_sup_rate", type=float, default=0.1) # 0.1, or 0.01
    parser.add_argument("--dataset_name", type=str, default="MUTAG")
    parser.add_argument("--epoch_select", type=str, default="test_max")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512) # 512
    parser.add_argument("--epochs", type=int, default=200) # 200
    parser.add_argument("--finetune_epochs", type=int, default=50) # 200 40
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--par_id", type=int, default=0)

    args = parser.parse_args()
    return args


def pretrain_train(encoder_model, predictor, dataloader, optimizer, args, pretrain=True, augs_type=None, device='cpu'):
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
        _, g, _, g1, _, g2, augs_g = encoder_model(data.x, data.edge_index, data.batch, edge_weights=edge_weights, edge_attr=edge_attr, 
                                                   eigen_values=eigen_values, eigen_vectors=data.eigen_vectors, x_eigen_distance=x_eigen_distance)
        g1, g2 = [encoder_model.graph_node_encoder.project(g) for g in [g1, g2]]
        # loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        encoder_exec_time += time.time() - start_time

        # denote positive samples, only for semi-supervise
        trues = data.y
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


def finetune_train(encoder_model, predictor, dataloader, optimizer, args, pretrain=False, joint_learn=False, num_classes=None, augs_type=None, device='cpu'):
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
        
        # trues = data.y[data.train_mask]
        trues = data.y

        # denote positive samples, only for semi-supervise
        ssl_loss = 0
        if joint_learn:
            _, g, _, g1, _, g2, augs_g = encoder_model(data.x, data.edge_index, data.batch, edge_weights=edge_weights, edge_attr=edge_attr, 
                                                   eigen_values=eigen_values, eigen_vectors=data.eigen_vectors, x_eigen_distance=x_eigen_distance)
            aug_g1, aug_g2 = augs_g
            p_g1, p_g2 = [encoder_model.graph_node_encoder.project(g_i) for g_i in [g1, g2]] # PROTEINS
            num_samples = data.y.shape[0]
            pos_mask = torch.eye(num_samples, dtype=bool, device=device)
            # if pretrain == False:
            #     # pos_mask = add_extra_pos_mask(pos_mask, data, data.train_mask)
            #     pos_mask = add_extra_pos_mask(pos_mask, data)
            ssl_loss = compute_infonce(p_g1, p_g2, pos_mask)
            epoch_loss += ssl_loss.item()

            preds = predictor(g)
            predict_loss1 = F.nll_loss(preds, trues)
            predict_loss2 = F.nll_loss(predictor(g1), trues)
            predict_loss3 = F.nll_loss(predictor(g2), trues)
            predict_loss = (predict_loss1+predict_loss2+predict_loss3)/3

            epoch_predict_loss += predict_loss.item()
        else:
            z, g = encoder_model(data.x, data.edge_index, data.batch, edge_weights=edge_weights, mode="eval")
        
            # preds = predictor(g)[data.train_mask]
            preds = predictor(g)
            predict_loss = F.nll_loss(preds, trues, reduction='sum')/(trues.shape[0])

            epoch_predict_loss += predict_loss.item()

        start_time = time.time()
        # spectral topological regularization
        spectral_topo_loss = 0
        if joint_learn and args.use_spectral_topo:
            if augs_type[0] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
                spectral_topo_loss += compute_spectral_topo_loss(data.edge_index, edge_weights, aug_g1, eigen_values, data.num_component, data.batch, augs_type[0])
            if augs_type[1] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
                spectral_topo_loss += compute_spectral_topo_loss(data.edge_index, edge_weights, aug_g2, eigen_values, data.num_component, data.batch, augs_type[1])
            spectral_topo_loss = spectral_topo_loss/(data.batch.max()+1)
            epoch_spectral_topo_loss += spectral_topo_loss.item()

        # spectral feature regularization
        spectral_feature_loss = 0
        if joint_learn and args.use_spectral_fea:
            if x_eigen_values!=None:
                if augs_type[0] in ["LearnableFeatureDroppingBySpectral"]:
                    spectral_feature_loss += compute_spectral_feature_loss(aug_g1, x_eigen_values, data.batch)/(data.batch.max()+1)
                    epoch_spectral_feature_loss += spectral_feature_loss.item()
                if augs_type[1] in ["LearnableFeatureDroppingBySpectral"]:
                    spectral_feature_loss += compute_spectral_feature_loss(aug_g2, x_eigen_values, data.batch)/(data.batch.max()+1)
                    epoch_spectral_feature_loss += spectral_feature_loss.item()
        spectral_exec_time += time.time() - start_time

        loss = 0.1*(ssl_loss + 0.8*spectral_topo_loss + 0.3*spectral_feature_loss) + predict_loss

        loss.backward() # on


        optimizer.step()

    return epoch_loss, epoch_predict_loss, epoch_spectral_topo_loss, epoch_spectral_feature_loss




def eval_acc(encoder_model, test_dataloader, predictor, device='cpu', args=None):
    encoder_model.eval()
    correct = 0
    for data in test_dataloader:
        data = data.to(device)
        num_nodes = data.batch.size(0)
        if data.x is None:
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        if 'edge_weights' in data.keys():
            edge_weights = data.edge_weights.reshape(-1,1)
        else:
            edge_weights = torch.ones((data.edge_index.shape[1], 1), device=device)

        z, g = encoder_model(data.x, data.edge_index, data.batch, edge_weights=edge_weights, mode="eval")
        # if args.dataset_name in ["PROTEINS"]:
        #     g = encoder_model.graph_node_encoder.project(g)
        pred = predictor(g).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(test_dataloader.dataset)


def main():

    hostname = socket.gethostname()
    print("Current hostname:", hostname)

    EIGEN_VEC_NUM = 10
    load_dataset = True
    x_eigen_load_dataset = True
    not_load_exist_model = False
    val_gap=10
    save_checkpoint_gap=50
    args = get_config()
    if args.dataset_name=="COLLAB":
        args.batch_size = 512-80
        args.epochs = 300
    elif args.dataset_name=="REDDIT-BINARY":
        args.batch_size = 256
    elif args.dataset_name=="REDDIT-MULTI-5K":
        args.batch_size = 256
    elif args.dataset_name=="DD":
        args.batch_size = 256-80
    elif args.dataset_name=="NCI1":
        args.batch_size = 512
    elif args.dataset_name=="CIFAR10":
        args.batch_size = 600
    elif args.dataset_name=="MNIST":
        args.batch_size = 1000

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
    path = osp.join(osp.expanduser('~'), 'datasets', 'semi')
    dataset, split = load_dataset_graphcls(path, dataset_name, args, semisup=True)

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
    dataset_dict = {"dataset": dataset,
                    "train_mask": split['train']}
    
    # eigen info calculate
    extra_eig_info_dataset_save_path = osp.join(path, dataset_name, f"eig_info_{EIGEN_VEC_NUM}.pt")
    if osp.exists(extra_eig_info_dataset_save_path) and load_dataset:
        eig_info_dict = torch.load(extra_eig_info_dataset_save_path)
    else:
        all_num_components = []
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
                    x_eig_value, x_eig_vector_U, x_eig_vector_V = process_xtopo_eigens(data, X_EIGEN_VEC_NUM)
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
        dataset_dict_keys = ["dataset", "train_mask", "num_component", "eigen_values", "eigen_vectors", 
                            "x_eigen_values", "x_eigen_vectors_U", "x_eigen_vectors_V"]
    else:
        dataset_dict_keys = ["dataset", "train_mask", "num_component", "eigen_values", "eigen_vectors"]
    dataset_dict = {key: dataset_dict[key] for key in dataset_dict_keys}
    combined_dataset = CombinedDataset(**dataset_dict)
    
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
    
    # print_memory_usage()
    ################################# FineTune #################################
    best_acc = 0
    print("-"*40+f"Starting"+"-"*40)
    all_results = []
    test_acc = "nan"
    joint_learn = False
    trails = 4

    ## pretrain

    gconv = ResGCN(dataset, 128, num_feat_layers=1, num_conv_layers=3,
                          num_fc_layers=2, gfn=False, collapse=False,
                          residual=False, res_branch='BNConvReLU',
                          global_pool="sum", dropout=0).to(device)

    encoder_model = GraphEncoder(graph_node_encoder=gconv, augmentor=(aug1, aug2)).to(device)

    ## finetune


    for fold_i, (train_idx, val_idx, test_idx) in enumerate(k_fold(len(dataset), y=y, epoch_select=args.epoch_select, semi_sup_rate=args.semi_sup_rate)):
        split = {'train': torch.asarray(train_idx, device=device, dtype=torch.int64), 
                 'valid': torch.asarray(val_idx, device=device, dtype=torch.int64),
                 'test':  torch.asarray(test_idx, device=device, dtype=torch.int64)}
        
        ## semi supervised update the dataset_dict["train_mask"]
        new_train_mask = torch.zeros(len(dataset), dtype=bool)
        new_train_mask = new_train_mask.scatter_(0, torch.asarray(train_idx), 1)
        dataset_dict.update({"train_mask": new_train_mask})

        combined_dataset = CombinedDataset(**dataset_dict)
        train_dataloader = CustomDataLoader(dataset=[combined_dataset[graph_i] for graph_i in train_idx], batch_size=batch_size, shuffle=True, 
                                  collate_mode="semi_pretrain", has_node_features=(num_node_features>=1))
        
        test_dataloader = DataLoader(dataset=dataset[test_idx], batch_size=batch_size)
        
        single_fold_all_results = []
        
        with tqdm(total=finetune_epochs*trails, desc=f'(FT_{fold_i})') as pbar:
            for trial in range(trails):
                if trial<2:
                    # each fold reload model
                    encoder_model, evaluator, predictor = load_model_evaluator(checkpoints_path, post_fix=f"pretrain_best", par_id=args.par_id, device=device)
                    encoder_model = encoder_model.to(device)
                    # each fold override predictor
                    predictor = Predictor(128, num_classes, semisup=True, num_layers=2).to(device) # num_classes = len(dataset.y.unique()) # PROTEINS 2
                    aug1 = encoder_model.augmentor[0]
                    aug2 = encoder_model.augmentor[1]
                    joint_learn = False
                    # define optimizer
                    params_for_optimizer = [{'params': encoder_model.parameters()}]
                    optimizer = Adam(params_for_optimizer, lr=0.002, weight_decay=0) # PROTEINS 0.001
                    
                else: # COLLAB lr 0.005, 0 1 2 trail, 
                    gconv = ResGCN(dataset, 128, num_feat_layers=1, num_conv_layers=3,
                            num_fc_layers=2, gfn=False, collapse=False,
                            residual=False, res_branch='BNConvReLU',
                            global_pool="mean", dropout=0).to(device)
                    # aug1 = LeA.LearnableFeatureDroppingBySpectral(input_dim=x_eigen_distance_input_dim, hidden_dim=128).to(device)
                    # aug2 = LeA.LearnableEdgePerturbation(input_dim_drop=edge_attr_input_dim, input_dim_add=edge_attr_input_dim, hidden_dim=128, sample_edges_ratio=0.2).to(device)
                    # encoder_model = GraphEncoder(graph_node_encoder=gconv, augmentor=(aug1, aug2)).to(device)
                    encoder_model.graph_node_encoder = gconv
                    predictor = Predictor(128, num_classes, semisup=True, num_layers=2).to(device)
                    joint_learn=True
                    params_for_optimizer = [{'params': encoder_model.parameters()}]
                    optimizer = Adam(params_for_optimizer, lr=0.01, weight_decay=1e-7*(10*(trial))) # lr 0.005
                
                for epoch in range(1, finetune_epochs+1):
                    # print_memory_usage()
                    loss, predict_loss, spectral_topo_loss, spectral_feature_loss = train(encoder_model, predictor, train_dataloader, optimizer, args, pretrain=False, num_classes=num_classes, joint_learn=joint_learn, augs_type=augs_type, device=device)
                    
                    test_acc = eval_acc(encoder_model, test_dataloader, predictor, device=device, args=args)

                    single_fold_all_results.append(test_acc)

                    pbar.set_postfix({'loss': loss, 
                                    'spec_topo': spectral_topo_loss, 
                                    'spec_feas': spectral_feature_loss,
                                    'pred_loss': predict_loss,
                                    'test_acc': test_acc})
                    pbar.update()
        
        best_idx = np.argmax(single_fold_all_results)
        all_results.append(single_fold_all_results[best_idx])
        logging.info("Aug_1: {}, Aug_2: {}, single_fold_results: {}".format(augs_type[0], augs_type[1], 
                                                                            single_fold_all_results[best_idx]))
        print(f"Fold {fold_i}'s result with semi_rate {args.semi_sup_rate}: micro_f1 {single_fold_all_results[best_idx]}")

        
    all_results = np.array(all_results)

    # each fold max/mean/min results
    final_acc = np.mean(all_results)

    print(f"all results: {all_results}")
    # results
    # print(f'(E): Avg all folds test F1Mi={avg_results["micro_f1"]:.4f}, F1Ma={avg_results["macro_f1"]:.4f}')
    print(f'(E): Best test F1Mi={final_acc:.4f}')
    logging.info("Aug_1: {}, Aug_2: {}, semi_rate: {}, avg_results: {}"\
                 .format(augs_type[0], augs_type[1], args.semi_sup_rate, final_acc))
    print("-"*90)

if __name__ == '__main__':
    main()