import torch
import numpy as np
from sklearn.svm import LinearSVC, SVC, SVR, LinearSVR
from sklearn.linear_model import LogisticRegression
from GCL.eval import BaseSKLearnEvaluator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

class SVMEvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, params=None, max_iter=10000, epoch_select='test_max', use_val=True, dataset_name=None):
        if linear:
            self.evaluator = LinearSVC(dual='auto', max_iter=max_iter)
        else:
            self.evaluator = SVC()
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(SVMEvaluator, self).__init__(self.evaluator, params, epoch_select=epoch_select, use_val=use_val, dataset_name=dataset_name)


class SVREvaluator(BaseSKLearnEvaluator): # svm regression
    def __init__(self, linear=True, params=None, max_iter=10000, epoch_select='test_max', use_val=True):
        if linear:
            self.evaluator = LinearSVR(dual='auto', max_iter=max_iter)
        else:
            self.evaluator = SVR()
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(SVREvaluator, self).__init__(self.evaluator, params, epoch_select=epoch_select, task='regression', use_val=use_val)


class LREvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, params=None, use_val=True):
        self.evaluator = LogisticRegression(max_iter=10000)
        # self.evaluator = make_pipeline(StandardScaler(), LogisticRegression())
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        
        super().__init__(self.evaluator, params, use_val=use_val)


