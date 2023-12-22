#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:27:33 2023

@author: yumsun
"""

from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
from sksurv.metrics import integrated_brier_score
import pickle
import os
import sys
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from impute_missing import impute_mean



def evaluate_performance(X, y):
    x_train = X[y[:,2] == 1,:]
    y_train = y[y[:,2] == 1,:2]
    
    x_test = X[y[:,2] == 0,:]
    y_test = y[y[:,2] == 0,:2]
    
    y_train = np.core.records.fromarrays(y_train.transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    y_test = np.core.records.fromarrays(y_test.transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    
    mod = CoxPHSurvivalAnalysis(n_iter = 200).fit(x_train,
                                               y_train)
    
    c_stat_train = mod.score(x_train,y_train)
    c_stat_test = mod.score(x_test,y_test)


    return {'C Stat Train': c_stat_train, 
            'C Stat Test': c_stat_test
            }

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
    
    results = evaluate_performance(X,y)

    pickle.dump(results,open(os.path.join(result_path,
                                           'performance_{:03d}.pkl'.format(numOfExp)),'wb'))
