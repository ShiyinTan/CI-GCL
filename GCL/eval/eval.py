from joblib import parallel_backend
import torch
import numpy as np
import warnings

from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.model_selection import GridSearchCV  


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def from_predefined_split(data):
    assert all([mask is not None for mask in [data.train_mask, data.test_mask, data.val_mask]])
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)
    return {
        'train': indices[data.train_mask],
        'valid': indices[data.val_mask],
        'test': indices[data.test_mask]
    }


def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)]) # setting -1 for train set, 0 for val set
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split)
        return result


class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params, epoch_select='test_max', task='classification', use_val=True, dataset_name=None):
        self.evaluator = evaluator
        self.params = params
        self.use_val = use_val
        self.epoch_select = epoch_select
        self.task = task
        self.dataset_name = dataset_name

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        if self.task == 'regression':
            grid_search = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='neg_mean_squared_error', verbose=0, n_jobs=8)
        else:
            if self.dataset_name=="IMDB-BINARY" or self.epoch_select=="test_max":
                grid_search = GridSearchCV(self.evaluator, self.params, cv=5, scoring='accuracy', verbose=0, n_jobs=8)
            else:
                grid_search = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0, n_jobs=8)

        grid_search.fit(x_train, y_train)
                
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        if self.task == 'regression':
            test_rmse = mean_squared_error(y_test, best_model.predict(x_test), squared=False)
            return {
                'rmse': test_rmse,
                'param': best_params,
            }, best_model
        else:
            test_macro = f1_score(y_test, best_model.predict(x_test), average='macro')
            test_micro = f1_score(y_test, best_model.predict(x_test), average='micro')
            return {
                'micro_f1': test_micro,
                'macro_f1': test_macro,
                'param': best_params,
            }, best_model

