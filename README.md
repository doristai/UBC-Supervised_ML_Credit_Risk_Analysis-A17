# UBC-Supervised_ML_Credit_Risk_Analysis-A17

Using scikit-learn and imbalanced-learn on sample credit data for credit risk analysis 

## Overview 

- **Aim** : To build and apply different machine learning models for analysing individual customer credit risk. 

- **Dataset** : Lending Club
    - 68,817 entries 
    - Unbalanced 
        - Only 0.5% (347) of entries considered as "high-risk"
    <img width="268" alt="low_high_val_count" src="https://user-images.githubusercontent.com/70616488/127429435-6ce3dcfe-60ba-43ef-a212-8d68cf840289.png">


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

<img width="348" alt="ccu_bal_acc" src="https://user-images.githubusercontent.com/70616488/127429164-2f0d8036-9ec8-413b-a295-d94aa4643b82.png">
<img width="699" alt="ccu_imbal_class" src="https://user-images.githubusercontent.com/70616488/127429180-b4fbbca0-fe81-4df1-abdd-ae738be8eff1.png">

### Combination Sampling

- Second worst
- Accurancy score : 0.6529 (slight better than 65% for credit risk prediction)
- F-score
    - Avg : 0.72
    - High-risk prediction : 0.02 
<img width="350" alt="combsamp_bal_acc" src="https://user-images.githubusercontent.com/70616488/127429244-70e80717-c329-4b1d-98e4-442aa56bc795.png">
<img width="697" alt="combsamp_imbal_class" src="https://user-images.githubusercontent.com/70616488/127429261-50f5f49e-e8d4-4b2c-9c29-7d3c3119e39b.png">


### SMOTE Oversampling

- 3rd worst 
- Accurancy : 0.662 (66%)
- F-score
    - Avg : 0.80
    - High-risk prediction : 0.02

<img width="346" alt="smote_bal_acc" src="https://user-images.githubusercontent.com/70616488/127429082-f4e3396a-3b7f-4625-9d43-2bdb406f9ada.png">
<img width="701" alt="smote_imbal_class" src="https://user-images.githubusercontent.com/70616488/127429101-dada4d91-f6cd-4f59-95b8-bc86877984fb.png">

### Naive Random Oversampling 

- 3rd best 
- Accuracy : 0.6732 (67%)
- F-score
    - Avg : 0.76
    - High-risk prediction : 0.02

<img width="444" alt="ros_bal_acc" src="https://user-images.githubusercontent.com/70616488/127429286-392d06cc-6fc5-4454-8217-de6a3ba6b4db.png">
<img width="704" alt="ros_imbal_class" src="https://user-images.githubusercontent.com/70616488/127429294-df3d63c0-a6ba-41ca-bbd8-c0b7020e3a91.png">

### Balanced Random Forest Classifier

- 2nd best 
- Accuracy : 0.7615 
- F-score
    - Avg : 0.92
    - High-risk prediction : 0.06

<img width="710" alt="brfc_bal_acc" src="https://user-images.githubusercontent.com/70616488/127429383-d8b25881-012f-4b74-8efc-94c097c00519.png">
<img width="702" alt="brfc_imbal_class" src="https://user-images.githubusercontent.com/70616488/127429402-f9a0cbc7-6174-48c1-b32c-77bfc7452d13.png">

### Easy Ensemble AdaBoost Classifier 

- Best performance model 
- Accurancy : 0.9319
- F-score
    - Avg : 0.97
    - High-risk prediction : 0.16

<img width="347" alt="eec_bal_acc" src="https://user-images.githubusercontent.com/70616488/127429352-45b5895b-0f79-4eb0-b3a2-c6b4c4986ac8.png">
<img width="697" alt="eec_imbal_class" src="https://user-images.githubusercontent.com/70616488/127429359-634e769e-c917-4f66-962b-4848f0d7d763.png">

## Summary 
Overall, it is hard to predict credit risk accurately, even for advanced ML model. From our results, Easy Ensemble AdaBoost Classifier had the best performance with an accurancy % of 93.19% and an average F-score of 97%. However, non of the model has a great F-score for high-risk prediction. A better ML algorithms should be use to obtain a higher F-score for predicting high-risk for credit risk analysis. 
