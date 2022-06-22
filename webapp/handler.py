import pandas as pd
import pickle
import os

from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann


# loading model
model = pickle.load(open('model/xgb_tuned_model.pkl', 'rb'))

app = Flask( __name__ )

@app.route('/rossmann/predict',methods=['POST'])

def rossmann_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict): # Unique example
            test_raw = pd.DataFrame(test_json, index[0])
        else: # multiple example
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
        
        # Instantiate Rossmann class
        pipeline = Rossmann()
        
        # data cleaning
        data_cleaning_df = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        feature_engineering_df = pipeline.feature_engineering(data_cleaning_df)
        
        # data preparation
        data_preparation_df = pipeline.data_preparation(feature_engineering_df)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, data_preparation_df)
        
        return df_response
    else:
        return Response('{}', status=200, mimetype='application/json')
    
if __name__ == '__main__':
    port = os.environ.get('PORT',5000)
    app.run(host='0.0.0.0',port=port)
