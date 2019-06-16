from joblib import load
from data_transform import DataTransform
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import numpy as np
import pandas as pd

MODEL_PATHS={'RF':['models/rf.pkl','models/rf_scaler.pkl']}

class Model:

    def __init__(self,backsteps=10,model_type='RF'):
        self.backsteps=backsteps
        
        self.model_type = model_type
        self.clf =  load(MODEL_PATHS[self.model_type][0])
        self.scaler = load(MODEL_PATHS[self.model_type][1])
        self.data_transformer = DataTransform(self.model_type,self.backsteps,1)
    
    def predict(self,df):
        df = self.data_transformer.get_eval_data(df)

        predictions = pd.DataFrame(columns=['group','demand'])
        
        for name,group in df.groupby('group'):
            x = group.values
            x = self.scaler.transform(x)
            y_hat = self.clf.predict(x)

            predictions.append({'group':[name for i in range(y_hat.shape[0])],'prediction':y_hat})


        print(predictions)

    def get_baseline_score(self,df):
        scores_df = pd.DataFrame(columns=['group','RMSLE','RMSE'])

        for name,group in df.groupby('group'):
            y = group.y
            y_hat = y.shift(1)
            y_hat[0] = y[0]


            RMSE = np.sqrt(mean_squared_error(y,y_hat))
            scores_df.append({'group':name,'RMSE':RMSE})
        
        return scores_df