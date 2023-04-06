#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 23:00:47 2023

@author: yumsun
"""

from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
from sksurv.metrics import integrated_brier_score
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
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from eli5.sklearn import PermutationImportance
from preprocess import clinic_preprocess,image_clinic_preprocess_model_fitting
from scipy.stats import norm

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
    
def get_feature_importance(model,x_train,x_test,y_train,y_test,niter,bestParam):
    if model == 'cox':
        est = CoxPHSurvivalAnalysis(n_iter = 200 )
    elif model == 'randomForest':
        est = ExtraSurvivalTrees(**bestParam)
    elif model == 'gradientBoosting':
        est = GradientBoostingSurvivalAnalysis(**bestParam)
    elif model == 'svm':
        est = FastKernelSurvivalSVM(**bestParam)

    est.fit(x_train, y_train)
    perm = PermutationImportance(est, n_iter=niter)
    perm.fit(x_test, y_test)
    featureImportance = pd.DataFrame({'score' : perm.feature_importances_},
                 index = x_train.columns.tolist())

    return featureImportance

def get_ensemble_feature_importance(model,x_train,x_test,y_train,y_test,niter,
                                    bestRsfParam,bestGbParam,bestSvmParam):

    est = Ensemble(rsfParam=bestRsfParam,gbParam=bestGbParam,svmParam = bestSvmParam)
    est.fit(x_train, y_train)
    perm = PermutationImportance(est, n_iter=niter)
    perm.fit(x_test, y_test)
    featureImportance = pd.DataFrame({'score' : perm.feature_importances_},
                 index = x_train.columns.tolist())

    return featureImportance

if __name__ == '__main__':
    numOfExp = int(sys.argv[1])
    dataTransform = sys.argv[2]
    model = sys.argv[3]

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
                                                  'selectedImageClinic_{:03d}.pkl'.format(numOfExp)),
                                           'rb'))
    
    imageNamesAll = pickle.load(open(os.path.join(dataPath,
                                               'imageFeatureNames.pkl'),'rb'))
    clinicNamesAll = pickle.load(open(os.path.join(dataPath,
                                                   'clinicFeatureNames.pkl'),'rb'))

    selectedClinic = [f for f in selectedImageClinic if f in clinicNamesAll]
    
    trainId = ids['TrainId']
    testId = ids['TestId']


    trainClinicImage = allDataAll.loc[trainId,selectedImageClinic]
    testClinicImage = allDataAll.loc[testId,selectedImageClinic]

    trainClinic = allDataAll.loc[trainId,selectedClinic]
    testClinic = allDataAll.loc[testId,selectedClinic]


    trainClinic  = clinic_preprocess(trainClinic)
    testClinic  = clinic_preprocess(testClinic)

    trainClinicImage,trainClinicImageNames = image_clinic_preprocess_model_fitting(
        trainClinicImage,selectedImageClinic,imageNamesAll, clinicNamesAll,
        dataTransform,deleteSkew=False)
    testClinicImage,_ = image_clinic_preprocess_model_fitting(testClinicImage,
                                          trainClinicImageNames,
                                          imageNamesAll, clinicNamesAll,
                                          dataTransform,deleteSkew=False)

    outcomeTrain = allDataAll.loc[trainId,['Event_First_ICU_to_Any_Death',
                                        'Time_First_ICU_to_Any_Death']]
    outcomeTest = allDataAll.loc[testId,['Event_First_ICU_to_Any_Death',
                                      'Time_First_ICU_to_Any_Death']]
    
    if model == 'svm':
        bestClinicImageParam = pickle.load(open(
            os.path.join(paramPath,
                         'clinicImageParamSvm_{:03d}.pkl').format(numOfExp),
            'rb'))
    elif model == 'rsf':
        bestClinicImageParam = pickle.load(open(
            os.path.join(paramPath,
                         'clinicImageParamRsf_{:03d}.pkl').format(numOfExp),
            'rb'))
    elif model == 'gb':
        bestClinicImageParam = pickle.load(open(
            os.path.join(paramPath,
                         'clinicImageParamGb_{:03d}.pkl').format(numOfExp),
            'rb'))
    elif model == 'ensemble':
        bestClinicImageSvmParam = pickle.load(open(
            os.path.join(paramPath,
                         'clinicImageParamSvm_{:03d}.pkl').format(numOfExp),
            'rb'))
        
        bestClinicImageRsfParam = pickle.load(open(
            os.path.join(paramPath,
                         'clinicImageParamRsf_{:03d}.pkl').format(numOfExp),
            'rb'))
        
        bestClinicImageGbParam = pickle.load(open(
            os.path.join(paramPath,
                         'clinicImageParamGb_{:03d}.pkl').format(numOfExp),
            'rb'))
    
    if model == 'Ensemble':
        fi = get_ensemble_feature_importance(model,trainClinicImage,testClinicImage,
                               outcomeTrain,outcomeTest,20,
                               bestClinicImageRsfParam,
                               bestClinicImageGbParam,
                               bestClinicImageSvmParam)
    else:
        fi = get_feature_importance(model,trainClinicImage,testClinicImage,
                               outcomeTrain,outcomeTest,20,bestClinicImageParam)

    pickle.dump(fi, open(os.path.join(resultPath,model,
                             'feature_importance_Exp{:03d}.pkl'.format(
                                 numOfExp)),'wb'))