#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:20:22 2023

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
from sksurv.ensemble import RandomSurvivalForest,ExtraSurvivalTrees 
from simulate_stratified_cv import stratified_cv,one_fold_cv
from impute_missing import impute_mean


def rf_tuner(data,outcome, nFold = 5):
    '''
    Parameters
    ----------
    data : DataFrame
        Features.
    outcome : DataFrame
        first column is event indicator
        second clolumn is survival time
    trainSize : Float, optional
        DESCRIPTION. The default is 0.8.
    nFold : Int, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    DataFrame, cross validation results

    '''

    outcomeCensor = np.core.records.fromarrays(outcome.transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8')
    param_grid = {
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': [100,300,500]
    }
    rsf = ExtraSurvivalTrees(n_jobs=-1,random_state=0)
    
    rsf_grid = GridSearchCV(estimator = rsf,
                            param_grid = param_grid,
                            cv = stratified_cv(outcome, nFold = nFold),
                            n_jobs = -1)
    rsf_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(rsf_grid.cv_results_)
    return cvRes

def rf_tuner_one_fold(data,outcome,trainTestId):
    outcomeCensor = np.core.records.fromarrays(outcome.transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8')
    param_grid = {
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': [100,300,500]
    }

    rsf = ExtraSurvivalTrees(n_jobs=-1,random_state=0)
    rsf_grid = GridSearchCV(estimator = rsf,
                            param_grid = param_grid,
                            cv = one_fold_cv(data,trainTestId),
                            n_jobs = -1)
    rsf_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(rsf_grid.cv_results_)
    return cvRes

if __name__ == '__main__':
    numOfExp = int(sys.argv[1])
    missing = bool(int(sys.argv[2]))
    
    dataPath = ''
    result_path = ''
    missing_id_path = ''
        
    X = np.genfromtxt(os.path.join(dataPath,
                                   'X_{:03d}.csv'.format(numOfExp)),
                        delimiter = ',')
    if missing:
        missing_id = np.genfromtxt(os.path.join(missing_id_path,
                                       'missing_id_{:03d}.csv'.format(numOfExp)),
                            delimiter = ',')
        
        X = impute_mean(X,missing_id)
        
    
    y = np.genfromtxt(os.path.join(dataPath,
                                   'y_{:03d}.csv'.format(numOfExp)),
                        delimiter = ',')
    
    
    
    x_train = X[y[:,2] == 1,:]
    y_train = y[y[:,2] == 1,:2]
    
    results = rf_tuner(x_train,y_train)

    pickle.dump(results,open(os.path.join(result_path,
                                           'cv_Exp{:03d}.pkl'.format(numOfExp)),'wb'))




    
    
    
    
    
    
