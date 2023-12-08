import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, average_precision_score



import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from model import Word2Vec_neg_sampling
from utils_modified import count_parameters
from datasets import word2vec_dataset
from helper import evaluate,data_loader
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)


def preprocess_split():
    file_path = 'admit_modified.csv'
    data = pd.read_csv(file_path)
    data.dropna(inplace=True, subset=['PROCEDURE_AND_DIAGNOSIS_ICD', 'LOS', 'AGE', 'GENDER_M', 'ETHNICITY_Asian', 'ETHNICITY_Black', 
                'ETHNICITY_Hispanic', 'ETHNICITY_Native_Hawaiian', 'ETHNICITY_Other', 
                'ETHNICITY_White', 'PROCEDURE_AND_DIAGNOSIS_ICD'])

    # Selecting specified features and target variable
    features = ['LOS', 'AGE', 'GENDER_M', 'ETHNICITY_Asian', 'ETHNICITY_Black', 
                'ETHNICITY_Hispanic', 'ETHNICITY_Native_Hawaiian', 'ETHNICITY_Other', 
                'ETHNICITY_White', 'PROCEDURE_AND_DIAGNOSIS_ICD']
    target = 'MORTALITY_30_DAY'
    #target = "MORTALITY_1_YEAR"

    # Splitting the dataset into features (X) and target (y)
    X = data[features]
    y = data[target]

    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)

    # Separating the text column for vectorization
    X_train_text = X_train['PROCEDURE_AND_DIAGNOSIS_ICD']
    X_test_text = X_test['PROCEDURE_AND_DIAGNOSIS_ICD']
    X_train = X_train.drop(columns='PROCEDURE_AND_DIAGNOSIS_ICD')
    X_test = X_test.drop(columns='PROCEDURE_AND_DIAGNOSIS_ICD')

    # Vectorizing and reducing the text column
    vectorizer = CountVectorizer(max_features=100)
    svd = TruncatedSVD(n_components=10, random_state=42)
    X_train_vectorized = vectorizer.fit_transform(X_train_text)
    X_test_vectorized = vectorizer.transform(X_test_text)
    X_train_reduced = svd.fit_transform(X_train_vectorized)
    X_test_reduced = svd.transform(X_test_vectorized)

    # Converting reduced data back to DataFrame and renaming columns
    X_train_reduced_df = pd.DataFrame(X_train_reduced, columns=[f'svd_{i}' for i in range(X_train_reduced.shape[1])])
    X_test_reduced_df = pd.DataFrame(X_test_reduced, columns=[f'svd_{i}' for i in range(X_test_reduced.shape[1])])

    # Concatenating the reduced text data with the other features
    X_train_final = pd.concat([X_train.reset_index(drop=True), X_train_reduced_df], axis=1)
    X_test_final = pd.concat([X_test.reset_index(drop=True), X_test_reduced_df], axis=1)
    return X_train_final, X_test_final, y_train, y_test

X_train_final, X_test_final, y_train, y_test = preprocess_split()



def random_forest_baseline(param_grid, X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    print("Best Parameters:", best_params)
    print("Accuracy:", accuracy)
    print("AUROC:", auroc)
    print("AUPRC:", auprc) 
    print("Classification Report:\n", report)
    return {'accuracy': accuracy, 'AUROC': auroc, 'AUPRC': auprc, 'report': report}





import xgboost as xgb




def xgb_baseline(X_train, y_train, X_test, y_test, param_grid):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)

    print("Best Parameters:", best_params)
    print("Accuracy:", accuracy)
    print("AUROC", auroc)
    print("AUPRC:", auprc) 
    print("Classification Report:\n", report)
    return {'accuracy': accuracy, 'AUROC': auroc, 'AUPRC': auprc, 'report': report}


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def logistic_baseline(X_train, y_train, X_test, y_test):

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('AUROC', auroc)
    print("AUPRC:", auprc) 
    print('Classification Report:')
    print(report)
    return {'accuracy': accuracy, 'AUROC': auroc, 'AUPRC': auprc, 'report': report}





param_grid_rf = {
    'n_estimators': [100, 150],
    'max_depth': [10, 15, None],
}

param_grid_xgb = {
    'n_estimators': [ 150, 200],
    'max_depth': [20, None],
    'learning_rate': [0.01, 0.1, 0.05]
}


rf_result = random_forest_baseline(param_grid_rf, X_train_final, y_train, X_test_final, y_test)
xgb_result = xgb_baseline(X_train_final, y_train, X_test_final, y_test, param_grid_xgb)
lr_result = logistic_baseline(X_train_final, y_train, X_test_final, y_test)


