import sys
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path=r'artifacts\model.pkl'
            preprocessor_path=r'artifacts\preprocessor.pkl'
            model=load_obj(model_path)
            preprocessor=load_obj(preprocessor_path)
            scaled_data=preprocessor.transform(features)
            pr=model.predict(scaled_data)
            return pr
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
        
    def data_to_dataframe(self):
        try:
            data = {
                'gender': [self.gender],
                'race/ethnicity': [self.race_ethnicity],
                'parental level of education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test preparation course': [self.test_preparation_course],
                'reading score': [self.reading_score],
                'writing score': [self.writing_score]
            }
            return pd.DataFrame(data)
        
        except Exception as e:
            raise CustomException(e,sys)
    