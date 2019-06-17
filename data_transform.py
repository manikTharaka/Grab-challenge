import numpy as np
from data_util import convert_data_uni, prep_df, pad_zeros
import pandas as pd


class DataTransform:

    def __init__(self,model_type,backsteps,forwardsteps):
        self.backsteps = backsteps
        self.forwardsteps = forwardsteps
        self.model_type = model_type

    

    def get_eval_data(self,df):
        
        out_df= None
        for name,group in df.groupby('geohash6'):
            
            preped = prep_df(group)

        
            if len(preped) < (self.backsteps+1) and len(preped) > 4:
                npad = self.backsteps - len(preped) + self.forwardsteps +1
                preped = pad_zeros(preped,npad)
        
        
            x_,y_ = convert_data_uni(preped.values,backsteps=self.backsteps,forwardsteps=self.forwardsteps,prev_y=False,norepeat_geo=True)
            
            if x_.shape[0] > 0: 
                group_df = pd.DataFrame(data=x_,columns=[f'x_{i}' for i in range(x_.shape[1])])
                group_df['group'] = [name for i in range(x_.shape[0])]
                group_y = pd.DataFrame(data=y_,columns=[f'y_{i}' for i in range(y_.shape[1])])
                group_df = pd.concat([group_df,group_y],axis=1)
                
                if out_df is None:
                    out_df = group_df.copy()
                else:
                    out_df = pd.concat([out_df,group_df])
        

        return out_df