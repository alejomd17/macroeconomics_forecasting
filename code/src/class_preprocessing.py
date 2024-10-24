import pandas as pd
import numpy as np
import os
from src.parameters import Parameters

def data_to_class(df_variables, train = False):
    df_variables = df_variables.drop(columns = ['autoID'])
    df_variables = df_variables[(df_variables.Demand != ' ')]
    df_variables.SeniorCity = df_variables.SeniorCity.apply(str)
    df_variables.Charges = df_variables.Charges.apply(float)
    df_variables.Demand = df_variables.Demand.apply(float)
    df_variables.Service2 = np.where((df_variables.Service2 == 'No phone service') | (df_variables.Service2 == 'No'), 'No','Yes')
    df_variables.Security = np.where((df_variables.Security == 'No internet service') | (df_variables.Security == 'No'), 'No','Yes')
    df_variables.OnlineBackup = np.where((df_variables.OnlineBackup == 'No internet service') | (df_variables.OnlineBackup == 'No'), 'No','Yes')
    df_variables.DeviceProtection = np.where((df_variables.DeviceProtection == 'No internet service') | (df_variables.DeviceProtection == 'No'), 'No','Yes')
    df_variables.TechSupport = np.where((df_variables.TechSupport == 'No internet service') | (df_variables.TechSupport == 'No'), 'No','Yes')
    for col in df_variables.select_dtypes(exclude=['number']).columns:
        df_variables[col] = df_variables[col].astype('category')
    class_raw = df_variables['Class']
    features_raw = df_variables.drop(['Class'], axis = 1)
    features_final = pd.get_dummies(features_raw)
    clase = class_raw.map({'Alpha':0,'Betha':1})
    if train == True:
        features_final[:0].to_excel(os.path.join(Parameters.results_path, 'df_col_dummies.xlsx'), index=False)
                        
    return features_final, clase