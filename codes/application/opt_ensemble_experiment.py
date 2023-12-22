#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 16:16:25 2023

@author: yumsun
"""
import pandas as pd
import numpy as np
import os
import pickle
from preprocess import clinic_preprocess,image_clinic_preprocess_model_fitting
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
            
            self.cox_est = CoxPHSurvivalAnalysis(n_iter = 200 )
            self.svm_est = FastKernelSurvivalSVM(**self.svmParam)
            self.rsf_est = ExtraSurvivalTrees(**self.rsfParam)
            self.gb_est = GradientBoostingSurvivalAnalysis(**self.gbParam)
            cox_est_censor_fold = CoxPHSurvivalAnalysis(n_iter = 200 )
            
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
    
def find_best_rbf_poly_params(resLoc,var,numOfExp):
    cvResPoly = pickle.load(open(
        os.path.join(resLoc,var,'poly',
                     'cv_{}_Exp{:03d}.pkl'.format(var,numOfExp)),
        'rb'))
    best_c_poly = cvResPoly.loc[cvResPoly['rank_test_score'] == 1, 'mean_test_score'].values[0]
    cvResRbf = pickle.load(open(
        os.path.join(resLoc,var,'rbf',
                     'cv_{}_Exp{:03d}.pkl'.format(var,numOfExp)),
        'rb'))
    best_c_rbf = cvResRbf.loc[cvResRbf['rank_test_score'] == 1, 'mean_test_score'].values[0]

    if best_c_poly >= best_c_rbf:
        bestParam = cvResPoly.loc[cvResPoly.rank_test_score == 1,'params'].values[0]
        bestParam['random_state'] = 0
        bestParam['max_iter'] = 100
    else:
        bestParam = cvResRbf.loc[cvResRbf.rank_test_score == 1,'params'].values[0]
        bestParam['random_state'] = 0
        bestParam['max_iter'] = 100
    return bestParam

def find_best_poly_params(resLoc,var,numOfExp):
    cvResPoly = pickle.load(open(
        os.path.join(resLoc,var,'poly',
                     'cv_{}_Exp{:03d}.pkl'.format(var,numOfExp)),
        'rb'))
    bestParam = cvResPoly.loc[cvResPoly.rank_test_score == 1,'params'].values[0]
    bestParam['random_state'] = 0
    bestParam['max_iter'] = 100
    bestParam['kernel'] = 'poly'
    return bestParam

def find_best_rbf_params(resLoc,var,numOfExp):
    cvResRbf = pickle.load(open(
        os.path.join(resLoc,var,'rbf',
                     'cv_{}_Exp{:03d}.pkl'.format(var,numOfExp)),
        'rb'))
    bestParam = cvResRbf.loc[cvResRbf.rank_test_score == 1,'params'].values[0]
    bestParam['random_state'] = 0
    bestParam['max_iter'] = 100
    bestParam['kernel'] = 'rbf'
    return bestParam

def find_best_linear_params(resLoc,var,numOfExp):
    cvResRbf = pickle.load(open(
        os.path.join(resLoc,var,'linear',
                     'cv_{}_Exp{:03d}.pkl'.format(var,numOfExp)),
        'rb'))
    bestParam = cvResRbf.loc[cvResRbf.rank_test_score == 1,'params'].values[0]
    bestParam['random_state'] = 0
    bestParam['max_iter'] = 100
    return bestParam

def find_svm_best_params(resLoc,var,numOfExp):
    poly_len = len(os.listdir(os.path.join(resLoc,var,'poly')))
    rbf_len = len(os.listdir(os.path.join(resLoc,var,'rbf')))

    if poly_len > 0  and rbf_len > 0:
        bestParam = find_best_rbf_poly_params(resLoc,var,numOfExp)
    elif poly_len > 0 and rbf_len == 0:
        bestParam = find_best_poly_params(resLoc,var,numOfExp)
    elif poly_len == 0 and rbf_len > 0:
        bestParam = find_best_rbf_params(resLoc,var,numOfExp)

    return bestParam

def find_rsf_best_params(resLoc,var,numOfExp):
    cvRes = pickle.load(open(
        os.path.join(resLoc,var,
                     'cv_{}_Exp{:03d}.pkl'.format(var,numOfExp)),
        'rb'))
    bestParam = cvRes.loc[cvRes.rank_test_score == 1,'params'].values[0]
    bestParam['random_state'] = 0
    return bestParam

def find_gb_best_params(resLoc,var,numOfExp):
    cvRes = pickle.load(open(
        os.path.join(resLoc,var,
                     'cv_{}_Exp{:03d}.pkl'.format(var,numOfExp)),
        'rb'))
    bestParam = cvRes.loc[cvRes.rank_test_score == 1,'params'].values[0]
    bestParam['random_state'] = 0
    return bestParam

def evaluate_performance(allData, trainTestId, selectedClinic,
                         selectedImageClinic,
                         imageNameAll,clinicNameAll,
                         transform,
                         clinicParamSvm, clinicImageParamSvm,
                         clinicParamRsf, clinicImageParamRsf,
                         clinicParamGb, clinicImageParamGb):

    trainId = trainTestId['TrainId']
    testId = trainTestId['TestId']


    trainClinicImage = allData.loc[trainId,selectedImageClinic]
    testClinicImage = allData.loc[testId,selectedImageClinic]

    trainClinic = allData.loc[trainId,selectedClinic]
    testClinic = allData.loc[testId,selectedClinic]


    trainClinic  = clinic_preprocess(trainClinic)
    testClinic  = clinic_preprocess(testClinic)
    
    trainClinicImage,trainClinicImageNames = image_clinic_preprocess_model_fitting(
        trainClinicImage,selectedImageClinic,imageNameAll, clinicNameAll,
        transform,deleteSkew=False)
    testClinicImage,_ = image_clinic_preprocess_model_fitting(testClinicImage,
                                          trainClinicImageNames,
                                          imageNameAll, clinicNameAll,
                                          transform,deleteSkew=False)

    outcomeTrain = allData.loc[trainId,['Event_First_ICU_to_Any_Death',
                                        'Time_First_ICU_to_Any_Death']]
    outcomeTest = allData.loc[testId,['Event_First_ICU_to_Any_Death',
                                      'Time_First_ICU_to_Any_Death']]

    clinicImageEnsemble = Ensemble(
        svmParam = clinicImageParamSvm,
        rsfParam=clinicImageParamRsf,
        gbParam=clinicImageParamGb).fit(trainClinicImage, outcomeTrain,nfold =3)
    clinicEnsemble = Ensemble(
        svmParam = clinicParamSvm,
        rsfParam=clinicParamRsf,
        gbParam=clinicParamGb).fit(trainClinic,outcomeTrain,nfold = 3)
    
    outcomeTrain = np.core.records.fromarrays(outcomeTrain.to_numpy().transpose(),
                                              names='Status, Survival_in_days', 
                                              formats = 'bool, f8')
    outcomeTest = np.core.records.fromarrays(outcomeTest.to_numpy().transpose(),
                                             names='Status, Survival_in_days', 
                                             formats = 'bool, f8')
    
    clinicResTrain = clinicEnsemble.score(trainClinic,outcomeTrain)
    clinicResTest = clinicEnsemble.score(testClinic,outcomeTest)

    clinicImageResTrain = clinicImageEnsemble.score(trainClinicImage,
                                               outcomeTrain)
    clinicImageResTest = clinicImageEnsemble.score(testClinicImage,
                                              outcomeTest)

    return {'Clinic Train': clinicResTrain, 'Clinic Test': clinicResTest,
            'Clinic Image Train': clinicImageResTrain,
            'Clinic Image Test': clinicImageResTest
            }
    
if __name__ == '__main__':
    numOfExp = int(sys.argv[1])
    dataTransform = sys.argv[2]

    dataSplitPath = ''
    dataPath = ''
    
    selectedClinicImagePath = ''
    selectedClinicPath = ''

    resultPath = ''
    paramPath = ''
    
    allDataAll =  pd.read_csv(os.path.join(dataPath,'data.csv'),
                           index_col = ['PatientID'])
    
    ids = pickle.load(open(os.path.join(dataSplitPath,'train_test_{:03d}.pkl'.format(numOfExp)),'rb'))
    
    selectedImageClinic = pickle.load(open(os.path.join(selectedClinicImagePath,
                                                  'selectImageClinic.pkl'),
                                           'rb'))
    
    imageNamesAll = pickle.load(open(os.path.join(dataPath,
                                               'imageFeatureNames.pkl'),'rb'))
    clinicNamesAll = pickle.load(open(os.path.join(dataPath,
                                                   'clinicFeatureNames.pkl'),'rb'))

    selectedClinic = [f for f in selectedImageClinic if f in clinicNamesAll]
    
    bestClinicParamSvm = pickle.load(open(
        os.path.join(paramPath,
                     'clinicParamSvm_{:03d}.pkl'.format(numOfExp)),
        'rb'))
    bestClinicImageParamSvm = pickle.load(open(
        os.path.join(paramPath,
                     'clinicImageParamSvm_{:03d}.pkl').format(numOfExp),
        'rb'))
    
    bestClinicParamRsf = pickle.load(open(
        os.path.join(paramPath,
                     'clinicParamRsf_{:03d}.pkl'.format(numOfExp)),
        'rb'))
    bestClinicImageParamRsf = pickle.load(open(
        os.path.join(paramPath,
                     'clinicImageParamRsf_{:03d}.pkl').format(numOfExp),
        'rb'))
    
    bestClinicParamGb = pickle.load(open(
        os.path.join(paramPath,
                     'clinicParamGb_{:03d}.pkl'.format(numOfExp)),
        'rb'))
    bestClinicImageParamGb = pickle.load(open(
        os.path.join(paramPath,
                     'clinicImageParamGb_{:03d}.pkl').format(numOfExp),
        'rb'))
    
    results = evaluate_performance(allDataAll,ids,selectedClinic,
                                   selectedImageClinic,
                                   imageNamesAll,
                                   clinicNamesAll,
                                   dataTransform,
                                   bestClinicParamSvm,
                                   bestClinicImageParamSvm,
                                   bestClinicParamRsf,
                                   bestClinicImageParamRsf,
                                   bestClinicParamGb,
                                   bestClinicImageParamGb)

    pickle.dump(results,open(os.path.join(resultPath,
                                           'performance_{:03d}.pkl'.format(numOfExp)),'wb'))


