# Bank_Customer_Churn
# Abstract:
Since keeping current clients is frequently more cost-effective than finding new ones, predicting customer churn is a crucial responsibility in the banking industry. Using a real-world bank dataset, we created a machine learning-based system in this project to forecast client attrition. Using oversampling techniques, the data was preprocessed to accommodate missing values, encode categorical features, and balance the distribution of classes. Three models—XGBoost Classifier, Random Forest Classifier, and Logistic Regression—were trained and assessed.
Accuracy, precision, recall, F1-score, and confusion matrices were used to evaluate performance. Although XGBoost and Logistic Regression produced results that were competitive, the Random Forest Classifier outperformed the others overall, tolerating data imbalance and capturing intricate feature relationships with ease. Random Forest was chosen as the last model to be deployed as a result.
The findings of this investigation provide valuable business insights for customer retention strategies, enabling banks to proactively identify at-risk customers and implement targeted retention policies.
# Data Source:
Data Source:
The dataset of bank churning is imported from Kaggle in CSV file.
# Import Libraries
import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# Import Dataset:
<img width="974" height="331" alt="image" src="https://github.com/user-attachments/assets/92a9ba07-45de-483a-aef9-d166f1579187" />
<img width="974" height="319" alt="image" src="https://github.com/user-attachments/assets/3f43d853-9925-4aa0-bd0e-06bc1ea52ea7" />
<img width="974" height="309" alt="image" src="https://github.com/user-attachments/assets/33f2978e-1728-450a-aa2c-ec03effd335c" />
# Cleaning and Preprocessing on Wrangler:
# Data Information
<img width="845" height="695" alt="image" src="https://github.com/user-attachments/assets/0842779d-dbea-473d-95ba-8a98c927c325" />
#  Exploratory Data Analysis(EDA): 
<img width="975" height="668" alt="image" src="https://github.com/user-attachments/assets/745cfb2d-6d93-4b36-a9ce-6e954254ffbb" />
<img width="906" height="711" alt="image" src="https://github.com/user-attachments/assets/eb73acb9-ebd1-42d7-9573-dc4300cf4e2d" />

<img width="975" height="636" alt="image" src="https://github.com/user-attachments/assets/92c1ff79-b8f5-495f-9d41-2428a34c0776" />
# Separate Target Feature:
separate the target feature for y variable.
# Split The Dataset:
split data into X_train and y_train,X_test and y_test.
# Test models
Apply 3 models which model accuracy is good pik that one model.
# Train Model:
train RandomForestClassifier Model.
# Evaluation:
<img width="956" height="584" alt="image" src="https://github.com/user-attachments/assets/7da1f758-c746-4002-95b4-ce49a3866b59" />

# Prediction System
Apply prediction system
# Save File
save file with joylib library.




