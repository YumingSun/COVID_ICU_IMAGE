#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:48:58 2023

@author: yumsun
"""

import pandas as pd
import numpy as np
import os
import pickle
# from preprocess import clinic_preprocess,image_clinic_preprocess_model_fitting
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
import sys
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
    
    def fit(self,data_x,data_y, nfold = 3 ):
        data_y_dead_time = data_y.loc[data_y.iloc[:,0],:]
        data_y_dead_time = data_y_dead_time.iloc[:,1]
        data_y_dead_risk = pd.DataFrame(
            rank_pct_probit(-1 * data_y_dead_time.to_numpy()),
            index = data_y_dead_time.index,columns=['risk_scores'])
        data_y_risk = data_y.merge(data_y_dead_risk,how = 'left',
                                          left_index=True, right_index=True)

        if nfold == 1:
            data_y_train = np.core.records.fromarrays(data_y.iloc[:,[0,1]].to_numpy().transpose(),
                                                      names='Status, Survival_in_days',
                                                      formats = 'bool, f8')
            data_y_censor_train = data_y.iloc[:,[0,1]]
            data_y_censor_train.iloc[:,0] = ~data_y_censor_train.iloc[:,0]
            data_y_censor_train = np.core.records.fromarrays(data_y_censor_train.to_numpy().transpose(),
                                                           names='Status, Survival_in_days',
                                                           formats = 'bool, f8')
            cox_est_censor_fold = CoxPHSurvivalAnalysis(n_iter = 200 )

            self.cox_est = CoxPHSurvivalAnalysis(n_iter = 200 )
            self.svm_est = FastKernelSurvivalSVM(**self.svmParam)
            self.rsf_est = ExtraSurvivalTrees(**self.rsfParam)
            self.gb_est = GradientBoostingSurvivalAnalysis(**self.gbParam)
            
            self.cox_est.fit(data_x,data_y_train)
            self.svm_est.fit(data_x,data_y_train)
            self.rsf_est.fit(data_x,data_y_train)
            self.gb_est.fit(data_x,data_y_train)
            cox_est_censor_fold.fit(data_x,data_y_censor_train)
            
            censor_func = cox_est_censor_fold.predict_survival_function(data_x,
                                                                        return_array=False)
            censor_prob = []
            
            for j in range(data_x.shape[0]):
                censor_prob.append(censor_func[j](np.clip(data_y.iloc[j,1],
                                                          a_min = censor_func[j].x.min(),
                                                          a_max = censor_func[j].x.max())
                                                  ))

            censor_prob = pd.DataFrame(censor_prob,index = data_x.index,
                                       columns = ['censoring_prob'])
            
            cox_score = self.cox_est.predict(data_x)
            if not getattr(self.cox_est, "_predict_risk_score", True):
                print('cox')
                cox_score *= -1
            cox_score = np.expand_dims(cox_score,axis = 1)

            svm_score = self.svm_est.predict(data_x)
            if not getattr(self.svm_est, "_predict_risk_score", True):
                print('svm')
                svm_score *= -1
            svm_score = np.expand_dims(svm_score,axis = 1)

            rsf_score = self.rsf_est.predict(data_x)
            if not getattr(self.rsf_est, "_predict_risk_score", True):
                print('rsf')
                rsf_score *= -1
            rsf_score = np.expand_dims(rsf_score,axis = 1)
            
            gb_score = self.gb_est.predict(data_x)
            if not getattr(self.gb_est, "_predict_risk_score", True):
                print('gb')
                gb_score *= -1
            gb_score = np.expand_dims(gb_score,axis = 1)

            score_all = np.concatenate((cox_score, svm_score,
                                        rsf_score, gb_score),
                                       axis = 1)
            score_all = pd.DataFrame(score_all, index = data_x.index,
                                     columns = ['cox', 'svm', 'rsf', 'gb'])

            score_dead = score_all.loc[data_y.iloc[:,0],:]
            score_dead = score_dead.apply(rank_pct_probit,axis = 0)
            score_dead = score_dead.rename(columns = {'cox':'cox_scale',
                                                                'svm': 'svm_scale',
                                                                'rsf': 'rsf_scale',
                                                                'gb': 'gb_scale'})
            true_score = data_y_risk.loc[:,['risk_scores']].dropna()

            score_for_error = score_dead.merge(true_score, how = 'inner',
                                               left_index = True, right_index = True)
            score_for_error_censor_prob = score_for_error.merge(censor_prob,
                                                                how = 'inner',
                                                                left_index = True,
                                                                right_index = True)
            score_for_error_censor_prob_np = score_for_error_censor_prob.to_numpy()

            error = score_for_error_censor_prob_np[:,:4] - score_for_error_censor_prob_np[:,[4]]

            weight_sqrt = np.sqrt(1/np.clip(score_for_error_censor_prob.iloc[:,[5]].to_numpy(),
                                            a_min = 1e-3,a_max = None))

            weighted_error_all = error * weight_sqrt
        else:
            data_y_np = np.core.records.fromarrays(data_y.to_numpy().transpose(),
                                                           names='Status, Survival_in_days',
                                                           formats = 'bool, f8')
            self.cox_est = CoxPHSurvivalAnalysis(n_iter = 200 )
            self.svm_est = FastKernelSurvivalSVM(**self.svmParam)
            self.rsf_est = ExtraSurvivalTrees(**self.rsfParam)
            self.gb_est = GradientBoostingSurvivalAnalysis(**self.gbParam)

            self.cox_est.fit(data_x,data_y_np)
            self.svm_est.fit(data_x,data_y_np)
            self.rsf_est.fit(data_x,data_y_np)
            self.gb_est.fit(data_x,data_y_np)
            
            skf = StratifiedKFold(n_splits=nfold)
            fold_ind = {}
            for i, (train_index, test_index) in enumerate(skf.split(np.zeros(data_y.shape[0]),
                                                                    data_y.iloc[:,0])):

                fold_ind[i] = (train_index,test_index)



            weighted_error_list = []
            for i in range(nfold):
                data_x_train_fold = data_x.iloc[fold_ind[i][0],:]

                data_y_train_fold = np.core.records.fromarrays(data_y.iloc[fold_ind[i][0],[0,1]].to_numpy().transpose(),
                                                               names='Status, Survival_in_days',
                                                               formats = 'bool, f8')

                data_y_censor_train_fold = data_y.iloc[fold_ind[i][0],[0,1]]
                data_y_censor_train_fold.iloc[:,0] = ~data_y_censor_train_fold.iloc[:,0]
                data_y_censor_train_fold = np.core.records.fromarrays(data_y_censor_train_fold.to_numpy().transpose(),
                                                               names='Status, Survival_in_days',
                                                               formats = 'bool, f8')


                cox_est_fold = CoxPHSurvivalAnalysis(n_iter = 200 )
                svm_est_fold = FastKernelSurvivalSVM(**self.svmParam)
                rsf_est_fold = ExtraSurvivalTrees(**self.rsfParam)
                gb_est_fold = GradientBoostingSurvivalAnalysis(**self.gbParam)
                cox_est_censor_fold = CoxPHSurvivalAnalysis(n_iter = 200 )

                cox_est_fold.fit(data_x_train_fold,data_y_train_fold)
                svm_est_fold.fit(data_x_train_fold,data_y_train_fold)
                rsf_est_fold.fit(data_x_train_fold,data_y_train_fold)
                gb_est_fold.fit(data_x_train_fold,data_y_train_fold)
                cox_est_censor_fold.fit(data_x_train_fold,data_y_censor_train_fold)
                
                censor_func = cox_est_censor_fold.predict_survival_function(data_x,
                                                                            return_array=False)
                censor_prob = []
                for j in range(data_x.shape[0]):
                    censor_prob.append(censor_func[j](np.clip(data_y.iloc[j,1],
                                                              a_min = censor_func[j].x.min(),
                                                              a_max = censor_func[j].x.max())
                                                      ))

                censor_prob = pd.DataFrame(censor_prob,index = data_x.index,
                                           columns = ['censoring_prob'])

                cox_fold_score = cox_est_fold.predict(data_x)
                if not getattr(cox_est_fold, "_predict_risk_score", True):
                    print('cox')
                    cox_fold_score *= -1
                cox_fold_score = np.expand_dims(cox_fold_score,axis = 1)

                svm_fold_score = svm_est_fold.predict(data_x)
                if not getattr(svm_est_fold, "_predict_risk_score", True):
                    print('svm')
                    svm_fold_score *= -1
                svm_fold_score = np.expand_dims(svm_fold_score,axis = 1)

                rsf_fold_score = rsf_est_fold.predict(data_x)
                if not getattr(rsf_est_fold, "_predict_risk_score", True):
                    print('rsf')
                    rsf_fold_score *= -1
                rsf_fold_score = np.expand_dims(rsf_fold_score,axis = 1)

                gb_fold_score = gb_est_fold.predict(data_x)
                if not getattr(gb_est_fold, "_predict_risk_score", True):
                    print('gb')
                    gb_fold_score *= -1
                gb_fold_score = np.expand_dims(gb_fold_score,axis = 1)
                
                fold_score_all = np.concatenate((cox_fold_score, svm_fold_score,
                                                 rsf_fold_score, gb_fold_score),
                                                axis = 1)
                fold_score_all = pd.DataFrame(fold_score_all, index = data_x.index,
                                              columns = ['cox', 'svm', 'rsf', 'gb'])

                fold_score_dead = fold_score_all.loc[data_y.iloc[:,0],:]
                fold_score_dead = fold_score_dead.apply(rank_pct_probit,axis = 0)
                fold_score_dead = fold_score_dead.rename(columns = {'cox':'cox_scale',
                                                                    'svm': 'svm_scale',
                                                                    'rsf': 'rsf_scale',
                                                                    'gb': 'gb_scale'})

                fold_score_all =  fold_score_all.merge(fold_score_dead,how = 'left',
                                                  left_index=True, right_index=True)
                
                test_fold_score = fold_score_all.iloc[fold_ind[i][1],:].dropna()
                test_fold_true_score = data_y_risk.iloc[fold_ind[i][1],:].dropna().loc[:,['risk_scores']]

                test_fold_score_all = test_fold_score.merge(test_fold_true_score,how = 'inner',
                                                            left_index=True, right_index=True)
                test_fold_score_all_censor_prob = test_fold_score_all.merge(censor_prob,
                                                                            how = 'left',
                                                                            left_index=True,
                                                                        right_index = True)
                test_fold_score_all_np = test_fold_score_all_censor_prob.iloc[:,4:9].to_numpy()

                error = test_fold_score_all_np[:,:4] - test_fold_score_all_np[:,[4]]

                weight_sqrt = np.sqrt(1/np.clip(test_fold_score_all_censor_prob.iloc[:,[9]].to_numpy(),
                                                a_min = 1e-3,a_max = None))

                weighted_error = error * weight_sqrt
                weighted_error_list.append(weighted_error)
            
            weighted_error_all = np.concatenate(weighted_error_list,axis = 0)
        
        C = np.matmul(weighted_error_all.T,weighted_error_all)/data_y.shape[0]
        # C_inv = np.linalg.inv(C)
        C_inv = np.diag(1/np.diag(C))
        self.w_opt = np.matmul(C_inv,np.ones((C.shape[0],1)))/np.matmul(np.matmul(
            np.ones((1,C.shape[0])),C_inv),np.ones((C.shape[0],1)))

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
        scoreEnsemble = np.matmul(scoreAll,self.w_opt)
        scoreEnsemble = np.squeeze(scoreEnsemble)

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
    
    
    x_train_df = pd.DataFrame(x_train)
    y_train_df = pd.DataFrame(y_train)
    y_train_df = y_train_df.astype({0: bool})
    
    mod = Ensemble(svmParam=svm_param, 
                   rsfParam=rsf_param,gbParam=gb_param).fit(x_train_df,
                                                            y_train_df, nfold = 5)
    
    
    y_train = np.core.records.fromarrays(y_train.transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    y_test = np.core.records.fromarrays(y_test.transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    
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
    
    svm_param = find_best_linear_rbf_params(svm_param_path,numOfExp)
    gb_param = find_best_params(gb_param_path,numOfExp)
    rsf_param = find_best_params(rsf_param_path,numOfExp)
    
    
    results = evaluate_performance(X,y,svm_param = svm_param, rsf_param = rsf_param,
                                   gb_param = gb_param)

    pickle.dump(results,open(os.path.join(result_path,
                                           'performance_{:03d}.pkl'.format(numOfExp)),'wb'))



            
