
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def clean_atipicis(df, col_x, col_y):
    col = col_x
    df_clean_atipics = df
    desv_col = df_clean_atipics[col].std()
    mean_col = df_clean_atipics[col].mean()
    
    df_clean_atipics[col+'_with_atipic'] = df_clean_atipics[col]
    df_clean_atipics['atipic_'+col] = np.where((df_clean_atipics[col] < (mean_col - 2 * desv_col)) | 
                                                (df_clean_atipics[col] > (mean_col + 2 * desv_col)), 1, 0)
    
    to_interpol = df_clean_atipics[df_clean_atipics['atipic_'+col] == 1].index

    for index in to_interpol:
        df_clean_atipics.loc[index, col] = np.nan
    
    df_clean_atipics[col] = df_clean_atipics[col].interpolate(method='linear').ffill().bfill()

    data_cleaned = []
    d_cleaned = 0

    for iter in range(len(df_clean_atipics)):
        if iter > 0 and int(df_clean_atipics.iloc[iter][col_y].split("-")[1]) <= int(df_clean_atipics.iloc[iter - 1][col_y].split("-")[1]):
            d_cleaned = df_clean_atipics.iloc[iter][col+'_with_atipic']
        else:
            d_cleaned = df_clean_atipics.iloc[iter][col]
        data_cleaned.append(d_cleaned)
    
    df_clean_atipics[col] = data_cleaned

    return df_clean_atipics

def scalated_dataframe(df, col):

    df_to_std = df
    df_to_std[col+'_trans'] = df_to_std[col] + (abs(df_to_std[col].min()) + 1)

    values_m = df_to_std[col].values.astype(float).reshape(-1, 1)
    values_trans_m = df_to_std[col+'_trans'].values.astype(float).reshape(-1, 1)

    # if values_m.shape[0] != 0:
    minmaxsc_m = MinMaxScaler(feature_range=(0, 1)).fit_transform(values_m)
    stdscaled_m = StandardScaler().fit_transform(values_m)
    log_m = np.log(values_trans_m)

    df_to_std[col+'_scal01'] = minmaxsc_m
    df_to_std[col+'_scal11'] = stdscaled_m
    df_to_std[col+'_scallg'] = log_m
    df_to_std = df_to_std.drop(columns = [col+'_trans'])
    df_to_std[col+'_og'] = df_to_std[col]
    return df_to_std

def de_escalate(pred, col_pronos, scale, df_acumulate, bm=''):
    if 'neuronalnetwork' in bm:
        pred = np.array(pred)
    if scale == col_pronos+'_scal01' or scale == col_pronos+'_scal11':
        if scale == col_pronos+'_scal01':
            scaler = MinMaxScaler()
        elif scale == col_pronos+'_scal11':
            scaler = StandardScaler()
        scaler.fit(df_acumulate)
        try:
            pred = scaler.inverse_transform(pred.astype(float).reshape(-1, 1)).reshape(-1)
        except:
            try:
                pred = scaler.inverse_transform(pred.values.reshape(-1, 1)).reshape(-1)
            except:
                pred = scaler.inverse_transform(np.array(pred).reshape(-1, 1)).reshape(-1)
                
    elif scale == col_pronos:
        pred = np.array(list(pred))

    elif scale == col_pronos+'_scallg':
        k = abs(df_acumulate.min()) + 1
        pred = np.array(list(pred))
        pred = np.exp(pred)
        pred = pred - k
         
    return pred