# Prognostic Utility of Radiomic Features for COVID-19 patients admitted to ICU

Table of contents
=================

<!--tc-->
   * [Table of contents](#table-of-contents)
   * [Overview](#overview)
   * [Requirements](#requirements)
<!--tc-->

Overview
========

Severe cases of COVID-19 often require Intensive Care Unit (ICU) escalation, where patients may experience serious disease courses and outcomes, including mortality. Chest X-rays are an essential part of diagnostic practice in evaluating patients with COVID-19. Our team has partnered with Michigan Medicine to monitor the outcomes of COVID-19 patients in the ICU. Our experience has enabled us to explore the potential benefits of using clinical information and chest X-ray images to predict patient outcomes. We propose an analytic workflow to overcome challenges, such as the lack of standardized approaches for image pre-processing and utilization, working with these data. In our study, we observed a mortality rate of 21\%, and 
important risk factors included age, vaccination status, fluid and electrolyte disorders, metastatic cancers, and neurological disorders, oxygen saturation, race, and four radiomic texture features. Across four individual prediction models and an ensemble predictor, we observed significant improvement in performance with the inclusion of imaging data over models trained on only clinical risk factors. Further, the enhancement with the addition of radiomic features was significantly higher among older and more severely ill patients. We believe this work presents both clinical and statistical novelty. The clinical innovation stems from utilizing extensive COVID-19 data resources and identifying important radiomic features for COVID-19 survival prediction in a highly vulnerable subset of patients with the greatest disease severity. By integrating the COVID-19 patient electronic health record and X-ray databases, we provide a convenient framework for connecting imaging studies to valuable clinical information and a reproducible, standardized workflow for image pre-processing, feature selection, and predictive modeling.

Requirements
============

The project has been tested on Python 3.9.7 with `scikit-survival == 0.19.0` , `Pandas == 1.5.3` and `Numpy == 1.24.2`.
