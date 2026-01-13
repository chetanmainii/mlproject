import pandas as pd 
import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj,evaluate_model

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import *

@dataclass
class ModelTrainerConfig:
    model_file_path=os.path.join('artifacts','model.pkl')
        
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self,x_train,x_test,y_train,y_test):
        try:
            logging.info("Initiated Model Training")
            models={
                'Linear Regression':LinearRegression(),
                'Ridge':Ridge(),
                'Lasso':Lasso(),
                'SVR':SVR(),
                'KNN':KNeighborsRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Random Forest':RandomForestRegressor(),
                'Adaboost':AdaBoostRegressor(),
                'Gradient Boost':GradientBoostingRegressor(),
                'XGBoost':XGBRegressor()
            }
            tuning={
            'Ridge':{'alpha':[0.01,0.1,1,10,100]},

            'Lasso':{'alpha':[0.001,0.01,0.1,1,10],
                            'max_iter':[5000,10000]},

            'SVR':{'C':[0.1,1,10,100],
                        'kernel':['linear','rbf'],
                        'gamma':['scale','auto'],
                        'epsilon':[0.1,0.2,0.5]},

            'KNN':{'n_neighbors':[3,5,7,9,11],
                                        'weights':['uniform','distance'],
                                        'metric':['euclidean','manhattan','minkowski'],
                                        'p':[1,2]},

            'Decision Tree':{'criterion':['squared_error','absolute_error'],
                                                    'max_depth':[None,5,10,20,30],
                                                    'min_samples_leaf':[1,2,4],
                                                    'min_samples_split':[2,5,10]},

            'Random Forest':{'n_estimators':[100,200,500],
                                                    'max_depth':[None,5,10,20],
                                                    'min_samples_leaf':[1,2,4],
                                                    'min_samples_split':[2,5,10]},

            'Adaboost':{'n_estimators':[50,100,200],
                                            'learning_rate':[0.01,0.1,1.0]},

            'Gradient Boost':{'n_estimators':[50,100,200],
                                                        'learning_rate':[0.01,0.1,0.2],
                                                        'max_depth':[3,5],
                                                        'subsample':[0.8,1.0]},

            'XGBoost':{'n_estimators':[100,300,500],
                                        'learning_rate':[0.01,0.05,0.1],
                                        'max_depth':[3,5,7],
                                        'subsample':[0.8,1.0],
                                        'colsample_bytree':[0.8,1.0],
                                        'gamma':[0,0.1,0.3],
                                        'reg_alpha':[0,0.1,1],
                                        'reg_lambda':[1,1.5,2]}
            }
            
            logging.info(f"Evaluating models - {list(models.keys())} on basis of R2 score")
            result:dict=evaluate_model(models=models,params=tuning,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test)
            
            logging.info("Choosing the model with best r2 score")
            best_r2_score=max(sorted(list(result.values())))
            model_name=list(result.keys())[list(result.values()).index(best_r2_score)]
            model=models[model_name]
            
            logging.info("Saving the model.pkl")
            save_obj(
                filepath=self.model_trainer_config.model_file_path,
                obj=model
            )
            
            logging.info("Model Training Completed")
            return model_name,best_r2_score
        
        except Exception as e:
            raise CustomException(e,sys)