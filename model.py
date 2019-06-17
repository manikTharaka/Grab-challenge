from joblib import load
from data_transform import DataTransform
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K


MODEL_PATHS={'LSTM':['models/LSTM-1.hdf5','models/lstm_scaler.pkl']}

class Model:

    def __init__(self,backsteps=24,model_type='LSTM'):
        self.backsteps=backsteps
        
        self.model_type = model_type
        if model_type == 'LSTM':
            self.clf = load_model(MODEL_PATHS[self.model_type][0])
            print(self.clf.summary())
        else:
            self.clf =  load(MODEL_PATHS[self.model_type][0])
        
        self.scaler = load(MODEL_PATHS[self.model_type][1])
        self.val_fsteps = 5
        self.data_transformer = DataTransform(self.model_type,self.backsteps,self.val_fsteps)
    
    def interpolate(self,ar):
        t1 = ar[:,0]
        t3 = ar[:,1]
        t5 = ar[:,2]

        t2 = (t3 + t1) /2
        t4 = (t3+t5) / 2

        res = np.hstack([t1,t2,t3,t4,t5]).reshape(-1,5)

        return res

    def predict(self,df):
        df = self.data_transformer.get_eval_data(df)

        predictions = pd.DataFrame(columns=['group','y1','y2','y3','y4','y5'])
        

        rmse_list = []
        for name,group in df.groupby('group'):
            print(f'group {name}')
            x = group[group.columns[:-6]]
            x = self.scaler.transform(x)
            x = x.reshape(x.shape[0],24,-1)
            y = group[group.columns[-5:]]
            y_hat = self.clf.predict(x)
            y_hat = self.interpolate(y_hat)
            
            
            rmse = self.RMSE(y_hat,y)
            
            group_predictions = pd.DataFrame({'group':[name for i in range(y_hat.shape[0])],'y1':y_hat[:,0],'y1':y_hat[:,0],'y2':y_hat[:,1],'y3':y_hat[:,2],'y4':y_hat[:,3],'y5':y_hat[:,4]})
            predictions = pd.concat([predictions,group_predictions])
            rmse_list.append([name ,rmse])

        predictions.to_csv('results/predictions.csv')

        rmse_df = pd.DataFrame(data=rmse_list,columns=['group','rmse'])
        rmse_df.to_csv('results/rmse.csv')
        print(rmse_df)

    def get_baseline_score(self,df):
        scores_df = pd.DataFrame(columns=['group','RMSLE','RMSE'])

        for name,group in df.groupby('group'):
            y = group.y
            y_hat = y.shift(1)
            y_hat[0] = y[0]


            RMSE = np.sqrt(mean_squared_error(y,y_hat))
            scores_df.append({'group':name,'RMSE':RMSE})
        
        return scores_df
    
    def RMSE(self,y_true,y_pred):
        return np.sqrt(mean_squared_error(y_true,y_pred))