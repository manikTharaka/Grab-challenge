import numpy as np
import sys
import pandas as pd

from geohash import decode, decode_exactly


def convert(timestamp):
    t = str(timestamp)
    h,m = list(map(int,t.split(':')))
    
    return (h*60) + m


def create_index(df):
    datetime = df['day'].map(str)+'-'+df['timestamp'].map(str)
    datetime = pd.to_datetime(datetime,format='%j-%H:%M')
    df.index = datetime
    df.sort_index(inplace=True)
    
    return df



def prep_df(df,keep_meta=False):
    
    cols = list(df.columns)
    cols[0] = 'geo'
    df.columns = cols
    
    time_delta = df['timestamp'].apply(convert)
    latlong = df['geo'].apply(lambda x: list(decode(x)))
    
    lat = latlong.apply(lambda x: x[0])
    long = latlong.apply(lambda x: x[1])
    
    df.insert(0,'lat',lat)
    df.insert(1,'long',long)
    df.insert(len(df.columns)-1,'delta',time_delta)
    
    if not keep_meta:
        df = df.drop(['geo','timestamp'],axis=1)
    
    return df    



def pad_zeros(df,npad):
    pad = pd.DataFrame(data=np.zeros((npad,len(df.columns))),columns=df.columns)
    
    df = pd.concat([pad,df])
    
    return df


def convert_data_uni(input_data,backsteps=10,forwardsteps=1,prev_y=False,norepeat_geo=False):

    X = list()
    Y = list()
    

    for i in range(input_data.shape[0]- backsteps - forwardsteps):
        row = list()
        for j in range(0,backsteps):
            if prev_y:
                row.append(input_data[i+j,2:])
            else:
                row.append(input_data[i+j,2:-1])
            
            if norepeat_geo:
                if j ==0:
                    row.append(input_data[-1,:2])
            else: 
                row.append(input_data[-1,:2])

        
        row = np.hstack(row)

        X.append(row)
        
      
        yrow = list()
        for k in range(0,forwardsteps):
            yrow.append(input_data[i+backsteps+k,-1])
        
        
        Y.append(yrow)
        
       
    return np.array(X),np.array(Y)    


def stack_by_groups(groups,backsteps,forwardsteps,norepeat_geo,prev_y=False):
    X=list()
    Y=list()

    i=0
    preped = None
    for name,group in groups:

        preped = prep_df(group)
        

        if len(preped) < (backsteps+1) and len(preped) > 4:
            npad = backsteps - len(preped) + forwardsteps +1
            preped = pad_zeros(preped,npad)

        x_,y_ = convert_data_uni(preped.values,backsteps=backsteps,forwardsteps=forwardsteps,prev_y=prev_y,norepeat_geo=norepeat_geo)


        if x_.shape[0] >0:
            X.append(x_)
            Y.append(y_)


        print(f'group {name} : {i}')
        i +=1
    
    X = np.vstack(X)
    Y = np.vstack(Y)
    
    return X,Y
