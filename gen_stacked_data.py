import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import  mean_squared_error, r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout     
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard 
from keras.models import load_model
# from geohash import decode
import argparse
from data_util import pad_zeros, create_index, convert_data_uni, prep_df, stack_by_groups


def main():
    
    parser = argparse.ArgumentParser(description='Generate stacked data file for the type of model you want to train')
    parser.add_argument('--t', metavar='N', type=str, nargs='?',
                        help='Model type',default='lstm',choices=['XGB','LSTM'])

    args = parser.parse_args()
    model_type = args.t

    backsteps = 24
    forwardsteps = 5
    fsteps_eval = 3
    frequency = '30T'


    if args.t == 'LSTM':
        train_fname = 'data/train_lstm.npy'
        ev_fname = 'data/eval_lstm.csv'
        norepeat_geo = False
    else:
        train_fname = 'data/train_ml.npy'
        ev_fname = 'data/eval_ml.csv'
        norepeat_geo = True

    df = pd.read_csv('data/training.csv')

    #TODO
    df= df.iloc[:10000]

    part = int(df.shape[0]*0.8)
    df = create_index(df)

    df_tr = df.iloc[:part]
    df_ev = df.iloc[part:]

    #restrict the train set to a few days
    df_tr = df_tr['1900-02-01':'1900-02-07']

    groups = df_tr.groupby('geohash6')

    X,Y = stack_by_groups(groups,backsteps,forwardsteps,norepeat_geo=norepeat_geo,prev_y=False)

    X_out = X.reshape(X.shape[0],-1)
    outar = np.hstack([X_out,Y])

    np.save(train_fname,outar)

    out_df = None
    ev_groups = list()
    
    for name,group in df_ev.groupby('geohash6'):
        preped = prep_df(group)
        preped = preped.resample(frequency).mean()
        preped = preped.fillna(method='ffill')
    
        if len(preped) < (backsteps+1) and len(preped) > 4:
            npad = backsteps - len(preped) + fsteps_eval +1
            preped = pad_zeros(preped,npad)
        
        x_,y_ = convert_data_uni(preped.values,backsteps=backsteps,forwardsteps=fsteps_eval,prev_y=True,norepeat_geo=False)
        
        
        if x_.shape[0] > 0: 

            group_df = pd.DataFrame(data=x_,columns=[f'x_{i}' for i in range(x_.shape[1])])
            group_df['group'] = [name for i in range(x_.shape[0])]
            
            group_y = pd.DataFrame(data=y_,columns=[f'y_{i}' for i in range(y_.shape[1])])
        
            group_df = pd.concat([group_df,group_y],axis=1)
        
        
            if out_df is None:
                out_df = group_df.copy()
            else:
                out_df = pd.concat([out_df,group_df])


    
    out_df.to_csv(ev_fname)


if __name__ == '__main__':
    main()