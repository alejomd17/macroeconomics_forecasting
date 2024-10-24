import pandas as pd
import uvicorn
from fastapi import FastAPI
from src.class_preprocessing import data_to_class
from src.class_classification import fn_classification
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from src.parameters import Parameters

app = FastAPI()
@app.post('/predict')
def class_compras(macro_vble:str
                ):
    
    if macro_vble == 'trm':
        df_trm          = pd.read_csv(os.path.join(Parameters.raw_path, f'df_{macro_vble}.csv'), sep='|')
        initial_date    =  (pd.to_datetime(df_trm.año_mes.max()) + relativedelta(months=1)).strftime("%Y-%m-01")
        final_date      = (datetime.now().replace(day=1) - timedelta(days=1)).strftime("%Y-%m-01")
        date_run        = datetime.now().strftime("%Y-%m")
        URLConsumer     = URLConsumer(initial_date, final_date)
        df_trm_prom_m   = URLConsumer.trm_prom_month()
        df_trm = pd.concat([df_trm, df_trm_prom_m.reset_index()], axis = 0 )
        df_trm.to_csv(os.path.join(Parameters.raw_path, f'df_trm.csv'), sep='|', index=False)
        
    
    df_trm
    X, y                    = data_to_class(df_new_class)
    df_new_classicated, df_new_classicated_consolidated      = fn_classification(df_new_class, X, stage="API")

    return {
        'historical':df_trm.to_dict(orient='records'),
        'predict': df_new_classicated[-1:].to_dict(orient='records')
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)