from flask import request,render_template,Flask
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictionPipeline

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def prediction():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race'),
            parental_level_of_education=request.form.get('parent_edu'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('prep_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )
        
        df=data.data_to_dataframe()
        obj=PredictionPipeline()
        results=obj.predict(df)
        return render_template('form.html',result=float(results[0]))
    
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
