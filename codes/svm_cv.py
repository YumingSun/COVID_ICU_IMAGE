#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:21:53 2023

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
from preprocess import clinic_preprocess,image_clinic_preprocess_model_fitting
from stratified_cv import stratified_cv


def ssvm_linear_sigmoid_cosine_tuner(data,outcome,kernel,trainSize = 0.8, nFold = 5):
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
    outcomeCensor = np.core.records.fromarrays(outcome.to_numpy().transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8')
    alpha = 2.** np.linspace(-10,-1,5)
    param_grid = {'alpha' : alpha}

    kssvm = FastKernelSurvivalSVM(kernel = kernel ,max_iter = 100, random_state = 0)
    kssvm_grid = GridSearchCV(estimator = kssvm,
                            param_grid = param_grid,
                            cv = stratified_cv(outcome,trainSize=trainSize, nFold = nFold),
                            n_jobs = -1)
    kssvm_grid.fit(data, outcomeCensor)
    
def ssvm_rbf_tuner(data,outcome, trainSize = 0.8, nFold = 5):
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
    dataframe, cross validation results
    '''
    outcomeCensor = np.core.records.fromarrays(outcome.to_numpy().transpose(),names='Status, Survival_in_days',
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
                            cv = stratified_cv(outcome, trainSize=trainSize, nFold = nFold),
                            n_jobs=-1)
    kssvm_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(kssvm_grid.cv_results_)
    return cvRes

def ssvm_poly_tuner(data,outcome, trainSize = 0.8, nFold = 5):
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
    dataframe, cross validation results
    '''
    outcomeCensor = np.core.records.fromarrays(outcome.to_numpy().transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8') 
    kernel = 'poly'
    nFea = data.shape[1]
    alpha = 2.** np.linspace(-10,-5,5)
    optimizer = ['avltree', 'rbtree']
    degree  = np.arange(start = 3,stop = 5)
    gamma  = (10.0 ** np.arange(-2,2))/nFea
    param_grid = {'alpha' : alpha, 'optimizer': optimizer,
                  'degree': degree, 'gamma':gamma}
    kssvm = FastKernelSurvivalSVM(kernel = kernel ,max_iter = 200)
    kssvm_grid = GridSearchCV(estimator = kssvm,
                            param_grid = param_grid,
                            cv = stratified_cv(outcome,trainSize=trainSize, nFold = nFold),
                            n_jobs=-1)
    kssvm_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(kssvm_grid.cv_results_)
    return cvRes

def ssvm_tuner_all(data,outcome,kernel, trainSize = 0.8, nFold = 5):
    '''
    Parameters
    ----------
    data : DataFrame
        Features.
    outcome : DataFrame
        first column is event indicator
        second clolumn is survival time
    kernel : String
        SVM kernel.
    trainSize : Float, optional
        DESCRIPTION. The default is 0.8.
    nFold : Int, optional
        DESCRIPTION. The default is 5.
    Returns
    -------
    dataframe, cross validation results
    '''
    kernels = ["linear", "rbf", "sigmoid", "cosine"]

    if kernel not in kernels:
        raise ValueError('Wrong Kernel')
    elif kernel in ["linear", "sigmoid", "cosine"]:
        cvRes = ssvm_linear_sigmoid_cosine_tuner(data,outcome,kernel,trainSize,nFold = 5)
    elif kernel == "rbf":
        cvRes = ssvm_rbf_tuner(data,outcome,trainSize,nFold)
    else:
        cvRes = ssvm_poly_tuner(data,outcome,trainSize,nFold)
    return cvRes

if __name__ == '__main__':
    dataTransform = sys.argv[1]
    kernel = sys.argv[2]
    numOfExp = int(sys.argv[3])
    var = sys.argv[4]
    
    
    trainSize = 0.8
    dataSplitPath = ''
    dataPath = ''
    
    selectedClinicImagePath = ''
    selectedClinicPath = ''

    paramPath = ''
    resultPath = ''
    
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
    allDataAll  = allDataAll.loc[trainId,:]

    clinic = allDataAll.loc[:,selectedClinic]
    clinic = clinic_preprocess(clinic)
    clinicImage,_ = image_clinic_preprocess_model_fitting(
        allDataAll,selectedImageClinic,imageNamesAll, clinicNamesAll,
        dataTransform,deleteSkew=False)
    
    outcome = allDataAll[['Event_First_ICU_to_Any_Death', 
                          'Time_First_ICU_to_Any_Death']]
    
    if var == 'clinic':
        cvRes = ssvm_tuner_all(data = clinic, outcome = outcome, kernel = kernel,
                               trainSize = trainSize, nFold = 5)
    elif var == 'clinicImage':
        cvRes = ssvm_tuner_all(data = clinicImage, outcome = outcome, kernel = kernel,
                               trainSize = trainSize, nFold = 5)
    
    pickle.dump(cvRes,open(
        os.path.join(resultPath,var,kernel,
                     'cv_{}_{}_Exp{:03d}.pkl'.format(var,kernel,numOfExp)),
        'wb'))


