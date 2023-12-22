#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 22:25:18 2023

@author: yumsun
"""

import numpy as np
from sksurv.metrics import integrated_brier_score
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd
import os
import sys
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.svm import FastKernelSurvivalSVM
from simulate_stratified_cv import stratified_cv,one_fold_cv
from impute_missing import impute_mean

def ssvm_linear_sigmoid_cosine_tuner(data,outcome,kernel, nFold = 5):
    '''

    Parameters
    ----------
    data : dataframe
        Features.
    outcome : dataframe
        first column is event indicator
        second clolumn is survival time
    nFold : int
        The default is 5.

    Returns
    -------
    dataframe, cross validation results

    '''
    outcomeCensor = np.core.records.fromarrays(outcome.transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8')
    alpha = 2.** np.linspace(-10,-1,5)
    param_grid = {'alpha' : alpha}
    
    kssvm = FastKernelSurvivalSVM(kernel = kernel ,max_iter = 100, random_state = 0)
    kssvm_grid = GridSearchCV(estimator = kssvm,
                            param_grid = param_grid,
                            cv = stratified_cv(outcome,nFold = nFold),
                            n_jobs = -1)
    kssvm_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(kssvm_grid.cv_results_)
    return cvRes

def ssvm_poly_tuner(data,outcome, nFold = 5):
    outcomeCensor = np.core.records.fromarrays(outcome.transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8')
    kernel = 'poly'
    nFea = data.shape[1]
    alpha = 2.** np.linspace(-10,-1,5)
    degree  = np.arange(start = 3,stop = 5)
    gamma  = (10.0 ** np.arange(-2,2))/nFea
    param_grid = {'alpha' : alpha, 
                  'degree': degree, 'gamma':gamma}
    kssvm = FastKernelSurvivalSVM(kernel = kernel ,max_iter = 100, random_state = 0)
    kssvm_grid = GridSearchCV(estimator = kssvm,
                            param_grid = param_grid,
                            cv = stratified_cv(outcome, nFold = nFold),
                            n_jobs=-1)
    kssvm_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(kssvm_grid.cv_results_)
    return cvRes

def ssvm_rbf_tuner(data,outcome, nFold = 5):
    outcomeCensor = np.core.records.fromarrays(outcome.transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8')
    kernel = 'rbf'
    nFea = data.shape[1]
    alpha = 2.** np.linspace(-10,-1,5)
    gamma = (10.0 ** np.arange(-2,2))/nFea
    param_grid = {'alpha' : alpha, 
                  'gamma':gamma}
    kssvm = FastKernelSurvivalSVM(kernel = kernel ,max_iter = 100, random_state=0)
    kssvm_grid = GridSearchCV(estimator = kssvm,
                            param_grid = param_grid,
                            cv = stratified_cv(outcome, nFold = nFold),
                            n_jobs=-1)
    kssvm_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(kssvm_grid.cv_results_)
    return cvRes

def ssvm_tuner_all(data,outcome,kernel, nFold = 5):
    kernels = ["linear", "poly", "rbf", "sigmoid", "cosine"]

    if kernel not in kernels:
        raise ValueError('No such kernel')
    elif kernel in ["linear", "sigmoid", "cosine"]:
        cvRes = ssvm_linear_sigmoid_cosine_tuner(data,outcome,kernel,nFold = 5)
    elif kernel == "poly":
        cvRes = ssvm_poly_tuner(data,outcome,nFold)
    else:
        cvRes = ssvm_rbf_tuner(data,outcome,nFold)
    return cvRes

if __name__ == '__main__':
    numOfExp = int(sys.argv[1])
    kernel = sys.argv[2]
    missing = bool(int(sys.argv[3]))
    
    dataPath = ''
    result_path = ''
    missing_id_path = ''
    

    X = np.genfromtxt(os.path.join(dataPath,
                                   'X_{:03d}.csv'.format(numOfExp)),
                        delimiter = ',')
    y = np.genfromtxt(os.path.join(dataPath,
                                   'y_{:03d}.csv'.format(numOfExp)),
                        delimiter = ',')
    
    if missing:
        missing_id = np.genfromtxt(os.path.join(missing_id_path,
                                       'missing_id_{:03d}.csv'.format(numOfExp)),
                            delimiter = ',')
        
        X = impute_mean(X,missing_id)
    
    x_train = X[y[:,2] == 1,:]
    y_train = y[y[:,2] == 1,:2]
    
    results = ssvm_tuner_all(x_train,y_train, kernel)
    

    pickle.dump(results,open(os.path.join(result_path,kernel,
                                          'cv_Exp{:03d}.pkl'.format(numOfExp)),'wb'))

