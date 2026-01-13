import pandas as pd 
import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj,categorize_features

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join('artifacts','preprocessor.pkl')
        
class DataTransformation:
    def __init__(self):
        self.data_transf_config=DataTransformationConfig()
    
    def get_preprocessor(self,df,targetcol):
        try:
            
            logging.info("Categorizing features into categorical and numerical")
            d=categorize_features(df)
            cat=d['categorical']
            num=d['numerical']
            
            if targetcol in cat:
                cat.remove(targetcol)
            elif targetcol in num:
                num.remove(targetcol)
            
            pipeline1=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder()),
                ]
            )
            
            pipeline2=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            logging.info("Defining a ColumnTransformer")
            preprocessor=ColumnTransformer(
                [
                    ('catpipeline',pipeline1,cat),
                    ('numpipeline',pipeline2,num)
                ]
            )    
            
            logging.info("Preprocessor Returned")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,trainpath,testpath,targetcol):
        try:
            logging.info("Iniatiated Data Transformation")
            logging.info("Reading the Training and Testing set")
            trainset=pd.read_csv(trainpath)
            testset=pd.read_csv(testpath)
            
            logging.info("Separating Dependent and independent features")
            X_train = trainset.drop(columns=[targetcol])
            y_train = trainset[targetcol]

            X_test = testset.drop(columns=[targetcol])
            y_test = testset[targetcol]
            
            preprocessor=self.get_preprocessor(trainset,targetcol)
            
            logging.info("Fit and Transforming train set")
            tr_xtrain=preprocessor.fit_transform(X_train)
            logging.info("Fitting and transforming test set")
            tr_xtest=preprocessor.transform(X_test)
            
            logging.info("Saving the preprocessor .pkl file")
            save_obj(
                filepath=self.data_transf_config.preprocessor_file_path,
                obj=preprocessor
            )
            
            logging.info("x_train,x_test,y_train,y_test returned")
            logging.info("Data Transformation Completed")
            return tr_xtrain,tr_xtest,y_train,y_test
        
        except Exception as e:
            raise CustomException(e,sys)