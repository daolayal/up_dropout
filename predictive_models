import os, pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, missingno as msno, statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy import stats
from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from itertools import chain
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns; sns.set()
from sklearn.externals import joblib
import xgboost as xgb

def predictive_model(df_train,predictors,target,model):
    
    if model == "log_reg":
        logreg = LogisticRegression(random_state = 0, solver = "saga", max_iter = 100, penalty = "l2")
        fitted_model = logreg.fit(df_train[predictors],df_train[target])
    elif model == "xgboost":
        xg_model = xgb.XGBClassifier(objective = "binary:logistic",colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 30)
        fitted_model = xg_model.fit(df_train[predictors],df_train[target])
    elif model == "rf":
        rf_model = RandomForestClassifier(n_estimators = 100, criterion = "entropy",random_state=1)
        fitted_model = rf_model.fit(df_train[predictors],df_train[target])
    return fitted_model;

def predicted_prob(fitted_model,df_test,predictors,target):
    pred_prob = fitted_model.predict_proba(df_test[predictors])[::,1]
    fpr, tpr, _ = metrics.roc_curve(df_test[target],  pred_prob)
    auc = metrics.roc_auc_score(df_test[target], pred_prob)
    summary = [fpr,tpr,auc]
    return summary;
