import pandas as pd
from sklearn.svm import SVC
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
from torch.utils.data import DataLoader, TensorDataset
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Dataset
from GCL.models.mymodels import GraphNodeEncoder, GraphEncoder, Predictor
from GCL.utils import compute_infonce, cluster_get, CustomDataLoader, compute_cluster_constrain_loss, k_fold, process_degree, process_topo_eigens, process_xtopo_eigens, save_model_evaluator, \
    load_model_evaluator, split_xy, str_to_bool, add_extra_pos_mask, degree, topo_cluster_labels_get, compute_spectral_topo_loss, compute_spectral_feature_loss, print_memory_usage
from sklearn.metrics import f1_score, roc_auc_score
from datetime import datetime
from general_data_loader import load_dataset_graphcls, get_split_mask, CombinedDataset
from torch_geometric.utils import scatter, to_networkx, to_dense_adj, get_laplacian
from sklearn.model_selection import KFold
from splitters import scaffold_split, random_split, species_split


def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_degree", type=str_to_bool, default=True)
    parser.add_argument("--use_spectral_fea", type=str_to_bool, default=True)
    parser.add_argument("--use_spectral_topo", type=str_to_bool, default=True)

    parser.add_argument("--mode", type=str, default="transfer", choices=["unsup", "semisup", "transfer"])
    parser.add_argument("--dataset_name", type=str, default="esol") # transfered dataset name
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


def train(encoder_model, predictor, dataloader, optimizer, args, num_classes=1, pretrain=True, augs_type=None, device='cpu'):
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

        # wether or not use degrees as node features when there is no given node features
        if args.use_degree:
            if data.x is None:
                data.x = data.degree
        else:
            if data.x is None:
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        
        # this edge attr construct for edge masking probability calculation
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
        
        ############# model ###################
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
        
        ssl_loss = compute_infonce(g1, g2, pos_mask)
        epoch_loss += ssl_loss.item()

        if pretrain == False:
            
            if args.dataset_name == "PPI_sup":
                trues = data.go_target_downstream.view(-1, num_classes)[data.train_mask].to(torch.float64)
                preds1 = predictor(g1)[data.train_mask].view(trues.shape)
                preds2 = predictor(g2)[data.train_mask].view(trues.shape)
                predict_loss1 = F.binary_cross_entropy_with_logits(preds1, trues)
                predict_loss2 = F.binary_cross_entropy_with_logits(preds2, trues)
            else:
                trues = data.y.view(-1, num_classes)
                # Whether y is non-null or not.
                trues = trues[data.train_mask]
                is_valid = trues**2 > 0
                trues = (trues+1)/2
                preds1 = predictor(g1)[data.train_mask].view(trues.shape)
                preds2 = predictor(g2)[data.train_mask].view(trues.shape)
                # Loss matrix
                predict_loss_mat1 = F.binary_cross_entropy_with_logits(preds1, trues, reduction="none")
                predict_loss_mat2 = F.binary_cross_entropy_with_logits(preds2, trues, reduction="none")
                # loss matrix after removing null target
                predict_loss_mat1 = torch.where(is_valid, predict_loss_mat1, torch.zeros(predict_loss_mat1.shape).to(predict_loss_mat1.device).to(predict_loss_mat1.dtype))
                predict_loss_mat2 = torch.where(is_valid, predict_loss_mat2, torch.zeros(predict_loss_mat2.shape).to(predict_loss_mat2.device).to(predict_loss_mat2.dtype))

                predict_loss1 = torch.sum(predict_loss_mat1)/torch.sum(is_valid)
                predict_loss2 = torch.sum(predict_loss_mat2)/torch.sum(is_valid)

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
                spectral_topo_loss += compute_spectral_topo_loss(data.edge_index, edge_weights, aug_g1, eigen_values, 1, data.batch, augs_type[0])
            if augs_type[1] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
                spectral_topo_loss += compute_spectral_topo_loss(data.edge_index, edge_weights, aug_g2, eigen_values, 1, data.batch, augs_type[1])
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

        loss.backward(retain_graph=True) # on


        optimizer.step()

    return epoch_loss, epoch_predict_loss, epoch_spectral_topo_loss, epoch_spectral_feature_loss


best_roc_score_val = 0
def train_and_test(predictor, x_train, x_val, x_test, y_train, y_val, y_test, dataset_name):
    global best_roc_score_val
    test_scores = []
    val_scores = []
    x_train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(x_train_data, batch_size=2000)
    optimizer = Adam(predictor.parameters(), lr=0.01)
    epochs = 100
    if dataset_name in ["hiv"]:
        epochs = 10
    if dataset_name in ["muv"]:
        epochs = 50
    for e in range(epochs):
        for i, (batch_x_train, batch_y_train) in enumerate(train_loader):
            y_preds = predictor(batch_x_train).view(batch_y_train.shape)
            y_trues = batch_y_train
            if dataset_name=="PPI_sup":
                y_trues = y_trues.to(torch.float64)
                predict_loss = F.binary_cross_entropy_with_logits(y_preds, y_trues)
            else:
                is_valid = y_trues**2 > 0
                y_trues = (y_trues+1)/2
                # Loss matrix
                predict_loss_mat = F.binary_cross_entropy_with_logits(y_preds, y_trues, reduction="none")
                predict_loss_mat = torch.where(is_valid, predict_loss_mat, torch.zeros(predict_loss_mat.shape).to(predict_loss_mat.device).to(predict_loss_mat.dtype))

                predict_loss = torch.sum(predict_loss_mat)/torch.sum(is_valid)

            # Backpropagation
            predict_loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        
        preds_val = predictor(x_val).view(y_val.shape)
        roc_score_val = evaluate(preds_val, y_val, dataset_name)
        val_scores.append(roc_score_val)

        preds_test = predictor(x_test).view(y_test.shape)
        roc_score_test = evaluate(preds_test, y_test, dataset_name)
        test_scores.append(roc_score_test)
    return val_scores, test_scores

def evaluate(preds, trues, dataset_name):
    roc_list_val = []
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.detach().cpu().numpy()
    if dataset_name=="PPI_sup":
        for i in range(trues.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(trues[:,i] == 1) > 0 and np.sum(trues[:,i] == 0) > 0:
                roc_list_val.append(roc_auc_score(trues[:,i], preds[:,i]))
            else:
                roc_list_val.append(np.nan)
        roc_score = np.mean(np.array(roc_list_val))
    else:
        for i in range(trues.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(trues[:,i] == 1) > 0 and np.sum(trues[:,i] == -1) > 0:
                is_valid_val = trues[:,i]**2 > 0
                roc_list_val.append(roc_auc_score((trues[is_valid_val,i] + 1)/2, preds[is_valid_val,i]))
        
        roc_score = sum(roc_list_val)/len(roc_list_val)
    return roc_score


def test(encoder_model, dataloader, args, split, predictor=None, num_classes=1, epoch_select='test_max', device='cpu'):
    encoder_model.eval()
    x = []
    y = []
    y_preds = []
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

        z, g = encoder_model(data.x, data.edge_index, data.batch, edge_weights=edge_weights, gnn_edge_attr=gnn_edge_attr, mode="eval")
        x.append(g)
        if args.dataset_name == "PPI_sup":
            y.append(data.go_target_downstream.view(-1, num_classes))
        else:
            y.append(data.y.view(-1, num_classes))
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    
    x_train, x_test, x_val, y_train, y_test, y_val = split_xy(x, y, split)
    y_train = y_train.view(-1, num_classes)
    y_val = y_val.view(-1, num_classes)
    y_test = y_test.view(-1, num_classes)

    val_scores, test_scores = train_and_test(predictor, x_train, x_val, x_test, y_train, y_val, y_test, dataset_name=args.dataset_name)

    return val_scores, test_scores



def main():

    hostname = socket.gethostname()
    print("Current hostname:", hostname)

    EIGEN_VEC_NUM = 50
    load_dataset = True
    x_eigen_load_dataset = True
    not_load_exist_model = False
    val_gap=1
    save_checkpoint_gap=50
    num_pred_layer = 1
    args = get_config()

    batch_size = args.batch_size
    epochs = args.epochs
    dataset_name = args.dataset_name
    if dataset_name == "PPI_sup":
        load_dataset_name = "PPI_unsup"
    else:
        load_dataset_name = "ZINC"
    finetune_epochs = args.finetune_epochs

    #Bunch of classification tasks
    if dataset_name == "tox21":
        num_classes = 12
        finetune_epochs = 100
    elif dataset_name == "hiv":
        num_classes = 1
        num_pred_layer = 2
        batch_size = 1024
        finetune_epochs = 100
    elif dataset_name == "pcba":
        num_classes = 128
    elif dataset_name == "muv":
        num_classes = 17
        num_pred_layer = 1
        batch_size = 1024
    elif dataset_name == "bace":
        num_classes = 1
    elif dataset_name == "bbbp":
        num_classes = 1
    elif dataset_name == "toxcast":
        num_classes = 617
    elif dataset_name == "sider":
        num_classes = 27
        finetune_epochs = 100
    elif dataset_name == "clintox":
        num_classes = 2
    elif dataset_name == "PPI_sup":
        num_classes = 40
    else:
        raise ValueError("Invalid dataset name.")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    current_time = datetime.now()
    print("Current time:", current_time)
    # print settings
    print("dataset: ", args.dataset_name, "load_dataset_name: ", load_dataset_name, "device: ", device, ", mode: ", args.mode)
    
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
        if "edge_attr" in dataset[0].keys() and dataset_name!="PPI_sup":
            num_edge_features = dataset[0].edge_attr.shape[1]
        else:
            num_edge_features = 0
        print(data_summary)
        print(dataset_name, "num_nodes: ", num_nodes, "num_node_features: ", num_node_features, 
              "num_edge_features: ", num_edge_features, "num_classes: ", num_classes)
    else:
        num_node_features = dataset[0].x.shape[1]
        if "edge_attr" in dataset[0].keys() and dataset_name!="PPI_sup":
            num_edge_features = dataset[0].edge_attr.shape[1]
        else:
            num_edge_features = 0
        print(dataset_name, "num_node_features: ", num_node_features, 
              "num_edge_features: ", num_edge_features, "num_classes: ", num_classes)  
    
    ################################# Data PreProcess and Custom Data loader #################################
    if not osp.exists(osp.join(path, dataset_name)):
        os.mkdir(osp.join(path, dataset_name))
    dataset_dict = {"dataset": dataset}
    # degree calculating
    whole_processed_dataset_save_path = osp.join(path, dataset_name, "whole_processed_dataset.pt")
    if osp.exists(whole_processed_dataset_save_path) and load_dataset:
        id_degree_dict = torch.load(whole_processed_dataset_save_path)
    else:
        all_degrees = []
        all_graph_id = []
        all_eigen_values = [] # all non-zero eigen values
        all_eigen_vectors = [] # and corresponded eigen vectors
        with tqdm(total=len(dataset), desc='(PRE)') as pbar:
            for i, data in enumerate(dataset):
                all_graph_id.append(i)

                node_degree = process_degree(data)
                all_degrees.append(node_degree)
                
                pbar.update()
        # combine dataset
        id_degree_dict = {"graph_id": all_graph_id, 
                          "degree": all_degrees,
                        }
        torch.save(id_degree_dict, whole_processed_dataset_save_path)
    dataset_dict.update({'graph_id': id_degree_dict["graph_id"]})
    dataset_dict.update({'degree': id_degree_dict["degree"]})
    dataset_dict.update({'train_mask': None})

    # eigen info calculate
    extra_eig_info_dataset_save_path = osp.join(path, dataset_name, "eig_info.pt")
    if osp.exists(extra_eig_info_dataset_save_path) and load_dataset:
        eig_info_dict = torch.load(extra_eig_info_dataset_save_path)
    else:
        all_eigen_values = [] # all non-zero eigen values
        all_eigen_vectors = [] # and corresponded eigen vectors
        with tqdm(total=len(dataset), desc='(Eig)') as pbar:
            for i, data in enumerate(dataset):

                eig_value, eig_vector = process_topo_eigens(data, EIGEN_VEC_NUM)
                all_eigen_values.append(eig_value)
                all_eigen_vectors.append(eig_vector)
                
                pbar.update()

        eig_info_dict = {"eigen_values": all_eigen_values, 
                         "eigen_vectors": all_eigen_vectors, 
                        }
        torch.save(eig_info_dict, extra_eig_info_dataset_save_path)
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
        dataset_dict_keys = ["dataset", "graph_id", "train_mask", "degree", 
                            "eigen_values", "eigen_vectors", 
                            "x_eigen_values", "x_eigen_vectors_U", "x_eigen_vectors_V"]
    else:
        dataset_dict_keys = ["dataset", "graph_id", "train_mask", "degree", 
                            "eigen_values", "eigen_vectors"]
    dataset_dict = {key: dataset_dict[key] for key in dataset_dict_keys}

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
    elif augs_type[0] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
        if args.use_spectral_topo==False:
            augs_type[0] += "WithoutSpectral"
            aug1_no_spec = True
    
    if augs_type[1] == "LearnableFeatureDroppingBySpectral":
        if args.use_spectral_fea==False:
            augs_type[1] += "WithoutSpectral"
            aug2_no_spec = True
    elif augs_type[1] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
        if args.use_spectral_topo==False:
            augs_type[1] += "WithoutSpectral"
            aug2_no_spec = True
    not_load_exist_model = (aug1_no_spec and aug2_no_spec)

    print(augs_type)

    checkpoints_path = "./checkpoints/{}/{}/{}_{}".format(dataset_name, args.mode, augs_type[0], augs_type[1])
    load_checkpoints_path = "./checkpoints/{}/{}/{}_{}".format(load_dataset_name, args.mode, "LearnableFeatureDroppingBySpectral", "LearnableEdgePerturbation")
    if not osp.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    
    # print_memory_usage()
    ################################# FineTune #################################
    best_acc = 0
    print("-"*40+f"Starting"+"-"*40)
    all_results = {'micro_f1': [], 
                   'macro_f1': []}
    all_mean_results = {'micro_f1': [],
                       'macro_f1': []}
    all_max_results = {'micro_f1': [],
                       'macro_f1': []}
    all_min_results = {'micro_f1': [],
                       'macro_f1': []}
    test_acc = "nan"
    if dataset_name == "PPI_sup":
        train_val_mask, test_mask = species_split(dataset)
        train_val_idx = torch.arange(len(dataset))[train_val_mask]
        test_idx = torch.arange(len(dataset))[test_mask]

        train_idx, val_idx = random_split(train_val_idx, frac_train=0.85, frac_test=0.15)
        print("species splitting")
    else:
        smiles_list = pd.read_csv(osp.join(path, 'chem_dataset', dataset_name, 'processed/smiles.csv'), header=None)[0].tolist()
        train_idx, val_idx, test_idx = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    split = {'train': torch.asarray(train_idx, device=device, dtype=torch.int64), 
             'valid': torch.asarray(val_idx, device=device, dtype=torch.int64),
             'test':  torch.asarray(test_idx, device=device, dtype=torch.int64)}
    
    ## semi supervised update the dataset_dict["train_mask"]
    new_train_mask = torch.zeros(len(dataset), dtype=bool)
    new_train_mask = new_train_mask.scatter_(0, torch.asarray(train_idx), 1)
    dataset_dict.update({"train_mask": new_train_mask})

    combined_dataset = CombinedDataset(**dataset_dict)
    dataloader = CustomDataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=True, collate_mode="trans_finetune", has_node_features=(num_node_features>=1))

    # each fold reload model
    encoder_model, evaluator, predictor = load_model_evaluator(load_checkpoints_path, post_fix=f"pretrain_10", par_id=args.par_id, device=device)
    gconv = GraphNodeEncoder(input_channels=input_fea_dim, hidden_channels=256, num_layers=2, edge_dim=num_edge_features).to(device)

    encoder_model = encoder_model.to(device)
    # each fold override predictor
    predictor = Predictor(encoder_model.graph_node_encoder.project_dim, num_classes, num_layers=num_pred_layer, loss_name="ce").to(device)
    aug1 = encoder_model.augmentor[0]
    aug2 = encoder_model.augmentor[1]
    encoder_model = GraphEncoder(graph_node_encoder=gconv, augmentor=(aug1, aug2)).to(device)

    # define optimizer
    params_for_optimizer = [{'params': encoder_model.parameters()}]
    if isinstance(aug1, LeA.Compose):
        for aug_elem in aug1.augmentors:
            aug_elem = aug_elem.to(device)
            if isinstance(aug_elem, LeA.LearnableAugmentor):
                params_for_optimizer.append({"params": aug_elem.parameters()})
    elif isinstance(aug1, LeA.LearnableAugmentor):
        print(augs_type[0], "learnable")
        aug1 = aug1.to(device)
        params_for_optimizer.append({"params": aug1.parameters()})
    if isinstance(aug2, LeA.Compose):
        for aug_elem in aug2.augmentors:
            aug_elem = aug_elem.to(device)
            if isinstance(aug_elem, LeA.LearnableAugmentor):
                params_for_optimizer.append({"params": aug_elem.parameters()})
    elif isinstance(aug2, LeA.LearnableAugmentor):
        print(augs_type[1], "learnable")
        aug2 = aug2.to(device)
        params_for_optimizer.append({"params": aug2.parameters()})
    params_for_optimizer.append({'params': predictor.parameters()})
    optimizer = Adam(params_for_optimizer, lr=0.01)

    all_results = []
    with tqdm(total=finetune_epochs, desc=f'(FT)') as pbar:
        for epoch in range(1, finetune_epochs+1):
            # print_memory_usage()
            loss, predict_loss, spectral_topo_loss, spectral_feature_loss = train(encoder_model, predictor, dataloader, optimizer, args, 
                                                                                  num_classes=num_classes, pretrain=False, 
                                                                                  augs_type=augs_type, device=device)
            
            if epoch % val_gap == 0:
                
                val_scores, test_scores = test(encoder_model, dataloader, args, split, predictor=predictor, num_classes=num_classes, epoch_select=args.epoch_select, device=device)
                
                best_idx = np.argmax(val_scores)
                if len(test_scores)>0:
                    test_acc = test_scores[best_idx]
                    if np.max(test_scores) > best_acc:
                        best_acc = np.max(test_scores)

                all_results.append(test_acc)
            pbar.set_postfix({'loss': loss, 
                            'spec_topo': spectral_topo_loss, 
                            'spec_feas': spectral_feature_loss,
                            'pred_loss': predict_loss,
                            'val_auc': val_scores[best_idx],
                            'test_auc': test_acc,
                            'best_auc': best_acc})
            pbar.update()

    
    logging.info("Aug_1: {}, Aug_2: {}, single_fold_results: {} {}".format(augs_type[0], augs_type[1], 
                                                                         np.max(all_results), best_acc))    
    print(f"Result: MAX AUC {np.max(all_results)}, all AUC {all_results}")

    print("-"*90)

if __name__ == '__main__':
    main()