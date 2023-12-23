#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:44:50 2023

@author: yumsun
"""

from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
from sksurv.metrics import integrated_brier_score
import pickle
import sys
import os
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from scipy.stats import norm
from impute_missing import impute_mean

def rank_pct_probit(scores):
    n = scores.shape[0]
    rk = np.argsort(np.argsort(scores))+1
    pct = (rk - 0.5)/n
    probit = norm.ppf(pct)
    return probit

class Ensemble:
    def __init__(self,
                 svmParam={},
                 rsfParam={},gbParam={}):
        self.svmParam = svmParam
        self.rsfParam = rsfParam
        self.gbParam = gbParam


    def get_params(self,deep=True):
        return {"svmParam" : self.svmParam,
                "rsfParam": self.rsfParam,
                "gbParam" : self.gbParam}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self,data_x,data_y):
       self.cox_est = CoxPHSurvivalAnalysis(n_iter = 200 )
       self.svm_est = FastKernelSurvivalSVM(**self.svmParam)
       self.rsf_est = ExtraSurvivalTrees(**self.rsfParam)
       self.gb_est = GradientBoostingSurvivalAnalysis(**self.gbParam)

       self.cox_est.fit(data_x,data_y)
       self.svm_est.fit(data_x,data_y)
       self.rsf_est.fit(data_x,data_y)
       self.gb_est.fit(data_x,data_y)
       return self
   
    def predict(self, data_x):
        coxScore = self.cox_est.predict(data_x)
        if not getattr(self.cox_est, "_predict_risk_score", True):
            print('cox')
            coxScore *= -1
        coxScoreScale = rank_pct_probit(coxScore)
        coxScoreScale = np.expand_dims(coxScoreScale,axis = 1)

        svmScore = self.svm_est.predict(data_x)
        if not getattr(self.svm_est, "_predict_risk_score", True):
            print('svm')
            svmScore *= -1
        svmScoreScale = rank_pct_probit(svmScore)
        svmScoreScale = np.expand_dims(svmScoreScale,axis = 1)
        
        rsfScore = self.rsf_est.predict(data_x)
        if not getattr(self.rsf_est, "_predict_risk_score", True):
            print('rsf')
            rsfScore *= -1
        rsfScoreScale = rank_pct_probit(rsfScore)
        rsfScoreScale = np.expand_dims(rsfScoreScale,axis = 1)

        gbScore = self.gb_est.predict(data_x)
        if not getattr(self.gb_est, "_predict_risk_score", True):
            print('gb')
            gbScore *= -1
        gbScoreScale = rank_pct_probit(gbScore)
        gbScoreScale = np.expand_dims(gbScoreScale,axis = 1)
        
        scoreAll = np.concatenate([coxScoreScale,
                                   svmScoreScale,
                                   rsfScoreScale,
                                   gbScoreScale],axis = 1)
        scoreEnsemble = np.mean(scoreAll,axis = 1)

        return scoreEnsemble
    
    def score(self,data_x,y):
        y_pred =  self.predict(data_x)
        y_event = y['Status']
        y_time = y['Survival_in_days']
        c = concordance_index_censored(y_event,y_time,y_pred)[0]
        return c

def find_best_linear_rbf_params(resLoc,numOfExp):
    cvResLinear = pickle.load(open(
        os.path.join(resLoc,'linear',
                     'cv_Exp{:03d}.pkl'.format(numOfExp)),
        'rb'))
    best_c_linear = cvResLinear.loc[cvResLinear['rank_test_score'] == 1, 'mean_test_score'].values[0]
    cvResRbf = pickle.load(open(
        os.path.join(resLoc,'rbf',
                     'cv_Exp{:03d}.pkl'.format(numOfExp)),
        'rb'))
    best_c_rbf = cvResRbf.loc[cvResRbf['rank_test_score'] == 1, 'mean_test_score'].values[0]

    if best_c_linear >= best_c_rbf:
        bestParam = cvResLinear.loc[cvResLinear.rank_test_score == 1,'params'].values[0]
        bestParam['random_state'] = 0
        bestParam['max_iter'] = 100
        bestParam['kernel'] = 'linear'
    else:
        bestParam = cvResRbf.loc[cvResRbf.rank_test_score == 1,'params'].values[0]
        bestParam['random_state'] = 0
        bestParam['max_iter'] = 100
        bestParam['kernel'] = 'rbf'
    return bestParam

def find_best_params(resLoc,numOfExp):
    cvRes = pickle.load(open(
        os.path.join(resLoc,
                     'cv_Exp{:03d}.pkl'.format(numOfExp)),
        'rb'))
    bestParam = cvRes.loc[cvRes.rank_test_score == 1,'params'].values[0]
    bestParam['random_state'] = 0
    return bestParam

def evaluate_performance(X, y, svm_param, rsf_param, gb_param):
    
    x_train = X[y[:,2] == 1,:]
    y_train = y[y[:,2] == 1,:2]
    
    x_test = X[y[:,2] == 0,:]
    y_test = y[y[:,2] == 0,:2]
    
    y_train = np.core.records.fromarrays(y_train.transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    y_test = np.core.records.fromarrays(y_test.transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    
    mod = Ensemble(svmParam=svm_param, 
                   rsfParam=rsf_param,gbParam=gb_param).fit(x_train,y_train)
    
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
    svm_param_path = ''
    gb_param_path = ''
    rsf_param_path = ''
    
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
        mean_ob = np.mean(missing_id,axis = 0)
    
    svm_param = find_best_linear_rbf_params(svm_param_path,numOfExp)
    gb_param = find_best_params(gb_param_path,numOfExp)
    rsf_param = find_best_params(rsf_param_path,numOfExp)
    
    
    results = evaluate_performance(X,y,svm_param = svm_param, rsf_param = rsf_param,
                                   gb_param = gb_param)

    pickle.dump(results,open(os.path.join(result_path,
                                           'performance_{:03d}.pkl'.format(numOfExp)),'wb'))



   
