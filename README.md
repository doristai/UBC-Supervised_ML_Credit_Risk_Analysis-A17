# UBC-Supervised_ML_Credit_Risk_Analysis-A17

Using scikit-learn and imbalanced-learn on sample credit data for credit risk analysis 

## Overview 

- **Aim** : To build and apply different machine learning models for analysing individual customer credit risk. 

- **Dataset** : Lending Club
    - 68,817 entries 
    - Unbalanced 
        - Only 0.5% (347) of entries considered as "high-risk"

- **ML Algorithms Used** : 
    - RandomOverSampler 
    - SMOTE
    - ClusterCentroids
    - SMOTEENN
    - BalancedRandomForestClassifier
    - EasyEnsembleClassifier

## Results 

Balance Accuracy Score and the Imbalamced Classification Report (ICR) are the two things we are looking at for each machine learning model. Particularly the F-Score ("f1" column) - the number from the bottom row ("avg/total") and "high risk" row from ICR. 

### Cluster Centroids Undersampling 

- Worst results
- Accuracy Score : 0.5295 -- the worst
    - Did a slightly better than 50% for credit risk prediction 
- F-score 
    - Avg : 0.56
    - high-risk prediction : 0.01 -- very poor

### Combination Sampling

- Second worst
- Accurancy score : 0.6529 (slight better than 65% for credit risk prediction)
- F-score
    - Avg : 0.72
    - High-risk prediction : 0.02 

### SMOTE Oversampling

- 3rd worst 
- Accurancy : 0.662 (66%)
- F-score
    - Avg : 0.80
    - High-risk prediction : 0.02

### Naive Random Oversampling 

- 3rd best 
- Accuracy : 0.6732 (67%)
- F-score
    - Avg : 0.76
    - High-risk prediction : 0.02

### Balanced Random Forest Classifier

- 2nd best 
- Accuracy : 0.7615 
- F-score
    - Avg : 0.92
    - High-risk prediction : 0.06

### Easy Ensemble AdaBoost Classifier 

- Best performance model 
- Accurancy : 0.9319
- F-score
    - Avg : 0.97
    - High-risk prediction : 0.16

## Summary 
Overall, it is hard to predict credit risk accurately, even for advanced ML model. From our results, Easy Ensemble AdaBoost Classifier had the best performance with an accurancy % of 93.19% and an average F-score of 97%. However, non of the model has a great F-score for high-risk prediction. A better ML algorithms should be use to obtain a higher F-score for predicting high-risk for credit risk analysis. 