import socket
import pandas as pd
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
from torch.utils.data import TensorDataset
from GCL.models.mymodels import GNN, GNN_graphpred, GraphNodeEncoder, GraphEncoder, Predictor
from GCL.utils import compute_infonce, cluster_get, CustomDataLoader, compute_cluster_constrain_loss, process_degree, save_model_evaluator, \
    load_model_evaluator, split_xy, str_to_bool, add_extra_pos_mask, degree, topo_cluster_labels_get, compute_spectral_topo_loss, compute_spectral_feature_loss, \
    process_i_degree_edge_weight, process_topo_eigens, process_xtopo_eigens
from sklearn.metrics import f1_score, roc_auc_score
from datetime import datetime
from general_data_loader import load_dataset_graphcls, CombinedDataset, get_subgraph_edge_attr_with_postfix
from torch_geometric.utils import scatter, to_networkx, to_dense_adj, get_laplacian
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from splitters import random_split, scaffold_split, species_split
from subgraph_structure import subgraph_structure_features
from joblib import Parallel, delayed, parallel_backend


def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--load_subgraph", type=str_to_bool, default=True)
    parser.add_argument("--use_degree", type=str_to_bool, default=True)
    parser.add_argument("--use_spectral_fea", type=str_to_bool, default=True)
    parser.add_argument("--use_spectral_topo", type=str_to_bool, default=True)

    parser.add_argument("--mode", type=str, default="transfer", choices=["unsup", "semisup", "transfer"])
    parser.add_argument("--dataset_name", type=str, default="ZINC") # ZINC or PPI_unsup
    parser.add_argument('--ft_dataset', type=str, default = 'bbbp')
    parser.add_argument("--linear", type=str_to_bool, default=True)
    
    # for subgraph structure
    parser.add_argument("--hll_p", type=int, default=8)
    parser.add_argument("--minhash_num_perm", type=int, default=128)
    parser.add_argument("--max_hash_hops", type=int, default=3)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024) # 512
    parser.add_argument("--ft_batch_size", type=int, default=512) # 512
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--par_id", type=int, default=0)

    args = parser.parse_args()
    return args


def train(encoder_model, predictor, dataloader, optimizer, args, augs_type=None, device='cpu'):
    encoder_model.train()
    epoch_loss = 0
    epoch_predict_loss = 0
    epoch_spectral_topo_loss = 0
    epoch_spectral_feature_loss = 0
    encoder_exec_time = 0
    spectral_exec_time = 0
    for data_idx, (data, extra_info_dict) in enumerate(dataloader):
        """
        data.cluster_labels: list of tensor: [n_nodes, 1]
        data.cluster_centrics: list of tensor: [n_cluster, fea_dim]
        data.cluster_num: list of int
        """
        data = data.to(device)
        if data_idx%100 == 0:
            current_time = datetime.now()
            print(f"inner epoch data index: {data_idx}. Current time: ", current_time.strftime("%Y-%m-%d %H:%M:%S"))

        # cluster_centrics = [c.to(device) for c in cluster_centrics]
        # cluster_labels = [cl.to(device) for cl in cluster_labels]
        # topo_cluster_labels = [cl.to(device) for cl in topo_cluster_labels]
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
        
        assert len(eigen_values)==(data.batch.max()+1)
        assert data.num_nodes==data.eigen_vectors.shape[0]
        # please make sure edge_weights [num_edges, 1]
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
        else:
            if data.x is None:
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        

        # x = encoder_model.graph_node_encoder.x_embedding1(data.x[:, 0]) + encoder_model.graph_node_encoder.x_embedding1(data.x[:, 1])
        x = data.x.float()
        # edge attr construct
        edge_attr = torch.pow(data.eigen_vectors[data.edge_index[0]] - data.eigen_vectors[data.edge_index[1]], 2)
        edge_attr = torch.cat([edge_attr, torch.concat([x[data.edge_index[0]], x[data.edge_index[1]]], dim=1)], dim=1)
        
        if x_eigen_values!=None:
            all_x_eigen_distance = []
            for i in range(len(x_eigen_vectors_U)): # for each single graph
                U_expanded = x_eigen_vectors_U[i].unsqueeze(1) # [n_nodes, 1, n_eigen_vecs]
                V_expanded = x_eigen_vectors_V[i].unsqueeze(0) # [1, n_feas, n_eigen_vecs]

                all_x_eigen_distance.append(torch.pow(U_expanded - V_expanded, 2)) # [n_nodes, n_feas, n_eigen_vecs]
            x_eigen_distance = torch.concat(all_x_eigen_distance)
        else:
            x_eigen_distance = None
        
        start_time = time.time()
        
        _, g, _, g1, _, g2, augs_g = encoder_model(x, data.edge_index, data.batch, edge_weights=edge_weights, 
                                                   edge_attr=edge_attr, gnn_edge_attr=gnn_edge_attr, 
                                                   eigen_values=eigen_values, eigen_vectors=data.eigen_vectors, x_eigen_distance=x_eigen_distance)
        g1, g2 = [encoder_model.graph_node_encoder.project(g) for g in [g1, g2]]
        encoder_exec_time += time.time() - start_time # encoder前向过程占据大量时间

        # denote positive samples, only for semi-supervise
        num_samples = len(data)
        pos_mask = torch.eye(num_samples, dtype=bool, device=device)
        if args.mode == "semisup":
            pos_mask = add_extra_pos_mask(pos_mask, data, data.train_mask)
        
        ssl_loss = compute_infonce(g1, g2, pos_mask)
        epoch_loss += ssl_loss.item()

        predict_loss = 0

        start_time = time.time()
        # spectral topological regularization
        aug_g1, aug_g2 = augs_g
        spectral_topo_loss = 0
        # if args.use_spectral_topo: 
        #     if augs_type[0] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
        #         spectral_topo_loss += compute_spectral_topo_loss(data.edge_index, edge_weights, aug_g1, eigen_values, 1, data.batch, augs_type[0])
        #     if augs_type[1] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
        #         spectral_topo_loss += compute_spectral_topo_loss(data.edge_index, edge_weights, aug_g2, eigen_values, 1, data.batch, augs_type[1])
        #     spectral_topo_loss = spectral_topo_loss/(data.batch.max()+1)
        #     epoch_spectral_topo_loss += spectral_topo_loss.item()

        # spectral feature regularization
        spectral_feature_loss = 0
        # if args.use_spectral_fea:
        #     if x_eigen_values!=None:
        #         if augs_type[0] in ["LearnableFeatureDroppingBySpectral"]:
        #             spectral_feature_loss += compute_spectral_feature_loss(aug_g1, x_eigen_values, data.batch)/(data.batch.max()+1)
        #             epoch_spectral_feature_loss += spectral_feature_loss.item()
        #         if augs_type[1] in ["LearnableFeatureDroppingBySpectral"]:
        #             spectral_feature_loss += compute_spectral_feature_loss(aug_g2, x_eigen_values, data.batch)/(data.batch.max()+1)
        #             epoch_spectral_feature_loss += spectral_feature_loss.item()
        # spectral_exec_time += time.time() - start_time

        loss = ssl_loss + predict_loss + 0.8*spectral_topo_loss + 0.3*spectral_feature_loss

        loss.backward() # on

        optimizer.step()

    return epoch_loss, epoch_predict_loss, epoch_spectral_topo_loss, epoch_spectral_feature_loss

def ft_train_and_eval(args, ft_model, device, ft_train_loader=None, ft_val_loader=None, ft_test_loader=None, ft_optimizer=None):
    best_test_acc = 0
    for ft_epoch in range(200):
        ft_train(args, ft_model, device, ft_train_loader, ft_optimizer)
        test_acc = ft_eval(args, ft_model, device, ft_test_loader)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    return best_test_acc

def ft_train(args, model, device, loader, optimizer):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction = "none")

    for step, data in enumerate(loader):
        data = data.to(device)
        # x = model.gnn.x_embedding1(data.x[:, 0]) + model.gnn.x_embedding1(data.x[:, 1])
        x = data.x.float()
        if 'edge_weights' in data.keys():
            edge_weights = data.edge_weights.reshape(-1,1)
        else:
            edge_weights = torch.ones((data.edge_index.shape[1], 1), device=device)
        pred = model(x, data.edge_index, data.batch, edge_weight=edge_weights, edge_attr=data.edge_attr.float())
        y = data.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def ft_eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, data in enumerate(loader):
        data = data.to(device)
        # x = model.gnn.x_embedding1(data.x[:, 0]) + model.gnn.x_embedding1(data.x[:, 1])
        x = data.x.float()
        if 'edge_weights' in data.keys():
            edge_weights = data.edge_weights.reshape(-1,1)
        else:
            edge_weights = torch.ones((data.edge_index.shape[1], 1), device=device)
        with torch.no_grad():
            pred = model(x, data.edge_index, data.batch, edge_weight=edge_weights, edge_attr=data.edge_attr.float())

        y_true.append(data.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    # if len(roc_list) < y_true.shape[1]:
    #     print("Some target is missing!")
    #     print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]






def main():
    hostname = socket.gethostname()
    print("Current hostname:", hostname)

    EIGEN_VEC_NUM = 50
    load_dataset = True
    x_eigen_load_dataset = True
    save_checkpoint_gap=5
    num_jobs = 4
    cpu_cores_num = os.cpu_count()
    args = get_config()

    batch_size = args.batch_size
    epochs = args.epochs
    if args.dataset_name in ["ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox"]:
        dataset_name = args.dataset_name.lower()
    else:
        dataset_name = args.dataset_name


    if args.ft_dataset == "tox21":
        num_tasks = 12
    elif args.ft_dataset == "hiv":
        num_tasks = 1
    elif args.ft_dataset == "pcba":
        num_tasks = 128
    elif args.ft_dataset == "muv":
        num_tasks = 17
    elif args.ft_dataset == "bace":
        num_tasks = 1
    elif args.ft_dataset == "bbbp":
        num_tasks = 1
    elif args.ft_dataset == "toxcast":
        num_tasks = 617
    elif args.ft_dataset == "sider":
        num_tasks = 27
    elif args.ft_dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")


    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    # 打印当前时间
    current_time = datetime.now()
    print("Current time:", current_time)
    # print settings
    print("dataset: ", args.dataset_name, "device: ", device, ", mode: ", args.mode)
    
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
    
    ################################# Data load and description #################################
    # data load
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset, split = load_dataset_graphcls(path, dataset_name, args)
    print("load data sucesses")

    num_node_features = dataset.num_node_features
    if "edge_attr" in dataset[0].keys() and dataset_name!="PPI_sup":
        num_edge_features = dataset[0].edge_attr.shape[1]
    else:
        num_edge_features = 0
    # print(data_summary)
    print(dataset_name, "num_node_features: ", num_node_features, "num_edge_features: ", num_edge_features)
    # description
    # data_summary = dataset.get_summary()
    
    
    ################################# Data PreProcess and Custom Data loader #################################
    dataset_dict = {"dataset": dataset}
    # degree calculating
    # whole_processed_dataset_save_path = osp.join(path, dataset_name, "whole_processed_dataset.pt")
    # if osp.exists(whole_processed_dataset_save_path):
    #     id_degree_dict = torch.load(whole_processed_dataset_save_path, map_location="cpu")
    # else:
    #     raise FileNotFoundError(f'file {whole_processed_dataset_save_path} not exist, please preprocess first! ')
    # dataset_dict.update({'graph_id': id_degree_dict["graph_id"]})
    # dataset_dict.update({'degree': id_degree_dict["degree"]})
    # print("load degree sucesses")
    

    # eigen info calculate
    extra_eig_info_dataset_save_path = osp.join(path, dataset_name, "eig_info.pt")
    if osp.exists(extra_eig_info_dataset_save_path):
        eig_info_dict = torch.load(extra_eig_info_dataset_save_path, map_location="cpu")
    else:
        raise FileNotFoundError(f'file {extra_eig_info_dataset_save_path} not exist, please preprocess first! ')
    dataset_dict.update({'eigen_values': eig_info_dict["eigen_values"]})
    dataset_dict.update({'eigen_vectors': eig_info_dict["eigen_vectors"]})
    print("load topo eigen sucesses")

    if num_node_features >= 1:
        extra_x_eig_info_dataset_save_path = osp.join(path, dataset_name, "x_eig_info.pt")
        X_EIGEN_VEC_NUM = min(num_node_features, EIGEN_VEC_NUM)
        if osp.exists(extra_x_eig_info_dataset_save_path) and x_eigen_load_dataset:
            x_eig_info_dict = torch.load(extra_x_eig_info_dataset_save_path, map_location="cpu")
        else:
            raise FileNotFoundError(f'file {extra_x_eig_info_dataset_save_path} not exist, please preprocess first! ')
        dataset_dict.update({'x_eigen_values': x_eig_info_dict["x_eigen_values"]})
        dataset_dict.update({'x_eigen_vectors_U': x_eig_info_dict["x_eigen_vectors_U"]})
        dataset_dict.update({'x_eigen_vectors_V': x_eig_info_dict["x_eigen_vectors_V"]})
        print("load xtopo eigen sucesses")

    if (num_node_features>=1):
        dataset_dict_keys = ["dataset", "eigen_values", "eigen_vectors", "x_eigen_values", "x_eigen_vectors_U", "x_eigen_vectors_V"]
    else:
        dataset_dict_keys = ["dataset", "eigen_values", "eigen_vectors"]
    dataset_dict = {key: dataset_dict[key] for key in dataset_dict_keys}
    print("key_num of dataset_dict: ", len(dataset_dict))

    combined_dataset = CombinedDataset(**dataset_dict)
    dataloader = CustomDataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=True, collate_mode="trans_pretrain", has_node_features=(num_node_features>=1))

    
    ################################# Finetune Data PreProcess and Custom Data loader #################################

    ft_dataset, split = load_dataset_graphcls(path, args.ft_dataset, args)

    if args.use_degree:
        input_fea_dim = max(num_node_features, 1)
    else:
        input_fea_dim = max(num_node_features, 1)

    if args.ft_dataset == "PPI_sup":
        train_val_mask, test_mask = species_split(ft_dataset)
        train_val_idx = torch.arange(len(ft_dataset))[train_val_mask]
        test_idx = torch.arange(len(ft_dataset))[test_mask]

        train_idx, val_idx = random_split(train_val_idx, frac_train=0.85, frac_test=0.15)
        print("species splitting")
    else:
        smiles_list = pd.read_csv(osp.join(path, 'chem_dataset', args.ft_dataset, 'processed/smiles.csv'), header=None)[0].tolist()
        ft_train_dataset, ft_valid_dataset, ft_test_dataset = scaffold_split(ft_dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, return_dataset=True)

    print(ft_train_dataset)
    ft_train_loader = DataLoader(ft_train_dataset, batch_size=args.ft_batch_size, shuffle=True)
    ft_val_loader = DataLoader(ft_valid_dataset, batch_size=args.ft_batch_size, shuffle=False)
    ft_test_loader = DataLoader(ft_test_dataset, batch_size=args.ft_batch_size, shuffle=False)

    ################################# Augmentation and Model define #################################
    if args.use_degree:
        input_fea_dim = max(num_node_features, 1)
    else:
        input_fea_dim = max(num_node_features, 1)

    # input_fea_dim = 128 # emb_dim in transfer setting

    # define augmentations
    aug1 = A.Identity()
    if num_node_features >= 1:
        x_eigen_distance_input_dim = X_EIGEN_VEC_NUM
        leA_FD = LeA.LearnableFeatureDroppingBySpectral(input_dim=x_eigen_distance_input_dim, hidden_dim=128).to(device)
        aug1 = leA_FD
    # aug1 = MaA.CrossClusterEdgeDropping(ratio=0.1)
    # aug2 = A.Identity()
    rand_FM = A.FeatureMasking(pf=0.1)
    rand_EA = A.EdgeAdding(pe=0.3)
    rand_ED = A.EdgeRemoving(pe=0.3)

    # edge_attr_input_dim = 10
    edge_attr_input_dim = input_fea_dim*2 + dataset_dict['eigen_vectors'][0].shape[1]
    leA_ED = LeA.LearnableEdgeDropping(input_dim=edge_attr_input_dim, hidden_dim=128, temp=1.0).to(device) # edge_attr: subg feas, eigen_vecs
    leA_EA = LeA.LearnableEdgeAdding(input_dim=edge_attr_input_dim, hidden_dim=128, sample_edges_ratio=0.2).to(device)
    leA_EP = LeA.LearnableEdgePerturbation(input_dim_drop=edge_attr_input_dim, input_dim_add=edge_attr_input_dim, hidden_dim=128, sample_edges_ratio=0.2).to(device)
    aug2 = leA_EP
    augs_type = [type(aug1).__name__, type(aug2).__name__]
    if augs_type[0] == "LearnableFeatureDroppingBySpectral":
        if args.use_spectral_fea==False:
            augs_type[0] += "WithoutSpectral"
    elif augs_type[0] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
        if args.use_spectral_topo==False:
            augs_type[0] += "WithoutSpectral"
    if augs_type[1] == "LearnableFeatureDroppingBySpectral":
        if args.use_spectral_fea==False:
            augs_type[1] += "WithoutSpectral"
    elif augs_type[1] in ["LearnableEdgeDropping", "LearnableEdgeAdding", "LearnableEdgePerturbation"]:
        if args.use_spectral_topo==False:
            augs_type[1] += "WithoutSpectral"
    print(augs_type)

    checkpoints_path = "./checkpoints/{}/{}/{}_{}".format(dataset_name, args.mode, augs_type[0], augs_type[1])
    if not osp.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    # define encoder
    # if dataset_name=="ZINC":
        # gconv = GraphNodeEncoder(input_channels=input_fea_dim, hidden_channels=256, num_layers=2, edge_dim=num_edge_features).to(device)
    # else:
    #     gconv = GraphNodeEncoder(input_channels=input_fea_dim, hidden_channels=256, num_layers=2).to(device)
    gconv = GNN(emb_dim=input_fea_dim, hidden_channels=256, edge_dim=num_edge_features).to(device) # transfer setting, x should be embedding[0]+embedding[1]
    encoder_model = GraphEncoder(graph_node_encoder=gconv, augmentor=(aug1, aug2)).to(device)
    # predictor = Predictor(gconv.project_dim, len(dataset.y.unique())).to(device) # num_classes = len(dataset.y.unique())
    predictor = None

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
    


    ################################# Training and tesing #################################
    # training
    print("-"*40+f"Starting"+"-"*40)
    # print_memory_usage()
    best_auc = 0
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1, epochs+1):
            # pre train
            loss, _, spectral_topo_loss, spectral_feature_loss = train(encoder_model, predictor, dataloader, optimizer, args, augs_type=augs_type, device=device)
            # if epoch % save_checkpoint_gap == 0:
            #     save_model_evaluator(checkpoints_path, encoder_model, best_evaluator=None, 
            #                          post_fix=f"pretrain_with_edgeattr_{epoch}" if num_edge_features!=0 else f"pretrain_{epoch}", 
            #                          par_id=args.par_id)

            # fine tune
            ft_model = GNN_graphpred(emb_dim=input_fea_dim, hidden_channels=256, num_tasks=num_tasks, edge_dim=num_edge_features)
            ft_model.gnn.load_state_dict(encoder_model.graph_node_encoder.state_dict())
            ft_model.gnn.to(device)
            ft_model.to(device)
            model_param_group = []
            model_param_group.append({"params": ft_model.gnn.parameters()})
            model_param_group.append({"params": ft_model.graph_pred_linear.parameters()})
            ft_optimizer = Adam(model_param_group, lr=0.01)
            test_auc = ft_train_and_eval(args, ft_model, device, ft_train_loader=ft_train_loader, 
                                         ft_test_loader=ft_test_loader, ft_optimizer=ft_optimizer)

            pbar.set_postfix({
                            'loss': loss, 
                            'spec_topo': spectral_topo_loss, 
                            'spec_feas': spectral_feature_loss,
                            'auc': test_auc})
            if best_auc<test_auc:
                best_auc = test_auc
                save_model_evaluator(checkpoints_path, encoder_model, best_evaluator=None, 
                                     post_fix=f"pretrain_with_edgeattr_{epoch}" if num_edge_features!=0 else f"pretrain_{epoch}", 
                                     par_id=args.par_id)
                
            pbar.update()

    print("AUC: ", best_auc)
    print("-"*90)

if __name__ == '__main__':
    main()