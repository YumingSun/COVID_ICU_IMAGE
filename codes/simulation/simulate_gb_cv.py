#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:44:41 2023

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
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from preprocess import clinic_preprocess,image_clinic_preprocess_model_fitting
from simulate_stratified_cv import stratified_cv,one_fold_cv
from impute_missing import impute_mean

def gb_tuner(data,outcome,nFold = 5):
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
        # 'max_features': ['sqrt','auto','log2'],
        'max_features': ['sqrt','log2','auto'],
        #'n_estimators': [150,300,600],
        'n_estimators': [100,300,500],
        # 'learning_rate': [0.01,0.05,0.1]
        'learning_rate': [0.05,0.1,0.3]
    }

    gb = GradientBoostingSurvivalAnalysis(random_state = 0)
    gb_grid = GridSearchCV(estimator = gb,
                            param_grid = param_grid,
                            cv = stratified_cv(outcome, nFold = nFold),
                            n_jobs = -1)
    gb_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(gb_grid.cv_results_)
    return cvRes

def gb_tuner_one_fold(data,outcome,trainTestId):
    outcomeCensor = np.core.records.fromarrays(outcome.to_numpy().transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8')
    param_grid = {
        'max_features': ['sqrt','log2'],
        'n_estimators': [100,200,400],
        'learning_rate': [0.05,0.1,0.3]
    }

    gb = GradientBoostingSurvivalAnalysis(random_state = 0)
    gb_grid = GridSearchCV(estimator = gb,
                            param_grid = param_grid,
                            cv = one_fold_cv(data,trainTestId),
                            n_jobs = -1)
    gb_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(gb_grid.cv_results_)
    return cvRes

if __name__ == '__main__':
    censor_rate = int(sys.argv[1])
    numOfExp = int(sys.argv[2])
    linear  = bool(int(sys.argv[3]))
    missing = bool(int(sys.argv[4]))
    
    if not linear:
        dataPath = '/home/yumsun/COVID_ICU/data/simulate/C{}'.format(censor_rate)
        if missing:
            result_path = '/home/yumsun/COVID_ICU/results/simulate_C20/C{}/gb_cv_mean_impute'.format(censor_rate)
            missing_id_path = '/home/yumsun/COVID_ICU/data/simulate/C{}/missing_id_C20'.format(censor_rate)
        else:    
            result_path = '/home/yumsun/COVID_ICU/results/simulate/C{}/gb_cv'.format(censor_rate)
    else:
        dataPath = '/home/yumsun/COVID_ICU/data/simulate_linear/C{}'.format(censor_rate)
        if missing:
            result_path = '/home/yumsun/COVID_ICU/results/simulate_linear_C20/C{}/gb_cv_mean_impute'.format(censor_rate)
            missing_id_path = '/home/yumsun/COVID_ICU/data/simulate_linear/C{}/missing_id_C20'.format(censor_rate)
        else:    
            result_path = '/home/yumsun/COVID_ICU/results/simulate_linear/C{}/gb_cv'.format(censor_rate)
        
   
    X = np.genfromtxt(os.path.join(dataPath,'x',
                                   'X_{:03d}.csv'.format(numOfExp)),
                        delimiter = ',')
    y = np.genfromtxt(os.path.join(dataPath,'y',
                                   'y_{:03d}.csv'.format(numOfExp)),
                        delimiter = ',')
    
    if missing:
        missing_id = np.genfromtxt(os.path.join(missing_id_path,
                                       'missing_id_{:03d}.csv'.format(numOfExp)),
                            delimiter = ',')
        
        X = impute_mean(X,missing_id)
            
    x_train = X[y[:,2] == 1,:]
    y_train = y[y[:,2] == 1,:2]
    
    results = gb_tuner(x_train,y_train)

    pickle.dump(results,open(os.path.join(result_path,
                                           'cv_Exp{:03d}.pkl'.format(numOfExp)),'wb'))


