import os
import sys
import pandas as pd 

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import *
from src.components.model_trainer import *

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_dataingestion(self):
        try:
            logging.info("Initiated Data Ingestion process")

            df = pd.read_csv(r'F:\code\Python\udemy_project\notebook\data\StudentsPerformance.csv')
            logging.info("Raw data read successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info(f"Directory '{os.path.dirname(self.ingestion_config.train_data_path)}' created/already exists")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            train, test = train_test_split(df, test_size=0.2, train_size=0.8, random_state=42)
            logging.info("Data split into train and test sets")

            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f"Training data saved at {self.ingestion_config.train_data_path}")

            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Testing data saved at {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion process Completed")
            
            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
            
        except Exception as e:
            logging.error("Error occurred in data ingestion")
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    trainpath,testpath=obj.initiate_dataingestion()
    
    obj2=DataTransformation()
    x_train,x_test,y_train,y_test=obj2.initiate_data_transformation(trainpath,testpath,'math score')
    
    obj3=ModelTrainer()
    modelname,r2score=obj3.initiate_model_training(x_train,x_test,y_train,y_test)
    print(modelname)
    print(r2score)
    