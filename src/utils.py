import sys
import os
import pandas as pd 
import dill
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_obj(filepath,obj):
    try:
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        with open(filepath, "wb") as file:
            dill.dump(obj, file)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(filepath):
    try:
        with open(filepath,"rb") as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)
    
def categorize_features(df):
        return {
            'categorical':[f for f in df.columns if df[f].dtype=='object'],
            'numerical':[f for f in df.columns if df[f].dtype!='object']
        }
        
def evaluate_model(models,params,x_train,x_test,y_train,y_test):
    score_dict={}
    fitted_models={}
    for name,model in models.items():
        gs=GridSearchCV(estimator=model,param_grid=params.get(name,{}),n_jobs=-1,cv=3)
        gs.fit(x_train,y_train)
        model=gs.best_estimator_
        xtestp=model.predict(x_test)
        score=r2_score(y_test,xtestp)
        score_dict[name]=score
        fitted_models[name]=model
    return score_dict,fitted_models