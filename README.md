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
Severe cases of COVID-19 often necessitate escalation to the Intensive Care Unit (ICU), where patients may face grave outcomes, including mortality. Chest X-rays play a crucial role in the diagnostic process for evaluating COVID-19 patients. Our collaborative efforts with Michigan Medicine in monitoring patient outcomes within the ICU have motivated us to investigate the potential advantages of incorporating clinical information and chest X-ray images for predicting patient outcomes. We propose an analytical workflow to address challenges such as the absence of standardized approaches for image pre-processing and data utilization. We then propose an ensemble learning approach designed to maximize the information derived from multiple prediction algorithms. This entails optimizing the weights within the ensemble and considering the common variability present in individual risk scores. Our simulations demonstrate the superior performance of this weighted ensemble averaging approach across various scenarios. We apply this refined ensemble methodology to analyze post-ICU COVID-19 mortality, an occurrence observed in 21\% of COVID-19 patients admitted to the ICU at Michigan Medicine. Our findings reveal substantial performance improvement when incorporating imaging data compared to models trained solely on clinical risk factors. Furthermore, the addition of radiomic features yields even larger enhancements, particularly among older and more medically compromised patients. These results may carry implications for enhancing patient outcomes in similar clinical contexts.

Requirements
============

The project has been tested on Python 3.9.7 with `scikit-survival == 0.19.0` , `Pandas == 1.5.3` and `Numpy == 1.24.2`.
