# CI-GCL

This is the code for "Community-Invariant Graph Contrastive Learning" (CI-GCL).
CI-GCL adopt learnable data augmentation with Community-Invariant constraint on both topology and features. 
And all these parts are jointly optimized to make sure the augmentation schemes can benefit from contrastive loss, CI constraints and downstream classifiers (semi-supervised).

## Requirement

Code is tested in **Python 3.10.13**. Some major requirements are listed below:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install dgl
pip install networkx
pip install tqdm tabulate
pip install scipy
pip install scikit-learn
```

Since we modify the framework of PyGCL, please do not include the PyGCL package in your enviroment.


## Run the code

For different graph-level tasks, we provides:

### to launch graph classification in unsupervised setting
```
python graph_classification.py --dataset_name ${dataset_name} --gpu ${gpu}
```
\${dataset_name} is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), \${gpu} is the lanched GPU ID


### to launch graph classification in semi-supervised setting
```
python graph_classification_semi_pretrain.py --dataset_name ${dataset_name}
```
```
python graph_classification_semi_finetune.py --dataset_name ${dataset_name}
```


### to launch graph classification in transfer learning setting
For transfer learning, please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.

When you can't manage to install the old torch-geometric versions, or you can't load the old datasets from "pretrain-gnns", you could install a new torch-geometric version and make modification based on https://github.com/pyg-team/pytorch_geometric/discussions/4502 (https://github.com/snap-stanford/pretrain-gnns/issues/14 this issue may also help).



```setup
python graph_transfer_pretrain_and_finetune.py --dataset_name ${dataset_name} --ft_dataset ${ft_dataset}
```
where \${dataset_name} is the pretrain dataset, and \${ft_dataset} is the finetune dataset


### further to check the robustness of the pretrained encoder
We randomly drop and add [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] edges in topology robustness test.
And randomly add gaussian noise to [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] positions in feature robustness test.
```
python robustness_check.py --dataset_name ${dataset_name}
python robustness_check_feature.py --dataset_name ${dataset_name}
```
