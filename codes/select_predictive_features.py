#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:44:44 2023

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
from sklearn.feature_selection import SequentialFeatureSelector
from preprocess import clinic_preprocess,image_clinic_preprocess_model_fitting

def get_predictive_features(features,outcome):
    '''
    Parameters
    ----------
    features : DataFrame
    outcome : DataFrame
    Returns
    -------
    selectedFeatures : List
        Selected Features by forward selection based on C-index
    '''
    outcomeCensor = np.core.records.fromarrays(outcome.to_numpy().transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8')
    sfs = SequentialFeatureSelector(CoxPHSurvivalAnalysis(n_iter = 200 ), 
                                    n_features_to_select="auto",
                                    tol = 1e-4, cv = 5,
                                    n_jobs = -1)
    sfs.fit(features, outcomeCensor)
    selectedFeatures = features.columns[sfs.get_support()].tolist()
    return selectedFeatures



if __name__ == '__main__':
    dataTransform = sys.argv[1]
    numOfExp = int(sys.argv[2])
    
    dataPath = ''
    dataSplitPath = ''
    selectedClinicImagePath = ''
    selectedClinicPath = ''
    
    allDataAll =  pd.read_csv(os.path.join(dataPath,'data.csv'))
    
    ids = pickle.load(open(os.path.join(dataSplitPath,'train_test_{:03d}.pkl'.format(numOfExp)),'rb'))
    trainId = ids['TrainId']
    allDataAll = allDataAll.loc[trainId,:]
    
    selectedImageClinic = pickle.load(open(os.path.join(
        selectedClinicImagePath,'lowCorImageClinic_{:03d}.pkl'.format(numOfExp)),'rb'))
    selectedClinic = pickle.load(open(os.path.join(
        selectedClinicPath,'lowCorClinic_{:03d}.pkl'.format(numOfExp)),'rb'))

    imageNamesAll = pickle.load(open(os.path.join(dataPath,'imageFeatureNames.pkl'),'rb'))
    clinicNamesAll = pickle.load(open(os.path.join(dataPath,'clinicFeatureNames.pkl'),'rb'))
    
    
    outcome = allDataAll.loc[:,['Event_First_ICU_to_Any_Death', 
                                'Time_First_ICU_to_Any_Death']]
    clinic = allDataAll.loc[:,selectedClinic]
    clinic  = clinic_preprocess(clinic)
    
    clinicImage = allDataAll.loc[:,selectedImageClinic]
    clinicImage,_ = image_clinic_preprocess_model_fitting(clinicImage,selectedImageClinic,
                                          imageNamesAll,clinicNamesAll,
                                          dataTransform)

    predictiveClinicFeatures = get_predictive_features(clinic,outcome)
    print(predictiveClinicFeatures)
    predictiveImageClinicFeatures = get_predictive_features(clinicImage,
                                                            outcome)
    print(predictiveImageClinicFeatures)
    
    pickle.dump(predictiveClinicFeatures,
                open(os.path.join(selectedClinicPath,
                                  'selectedImageClinic_{:03d}.pkl'.format(numOfExp)),
                     'wb'))
    pickle.dump(predictiveImageClinicFeatures,
                open(os.path.join(selectedClinicImagePath,
                                  'selectedImageClinic_{:03d}.pkl'.format(numOfExp)),
                     'wb'))