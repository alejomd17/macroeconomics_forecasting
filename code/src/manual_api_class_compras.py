    import pandas as pd
import uvicorn
from fastapi import FastAPI
from src.class_preprocessing import data_to_class
from src.class_classification import fn_classification
app = FastAPI()

@app.post('/classification')
def class_compras(autoID:str          
                , SeniorCity:str      
                , Partner:str         
                , Dependents:str      
                , Service1:str        
                , Service2:str        
                , Security:str        
                , OnlineBackup:str    
                , DeviceProtection:str
                , TechSupport:str     
                , Contract:str        
                , PaperlessBilling:str
                , PaymentMethod:str   
                , Charges:float       
                , Demand:float        
                ):
    
    df_new_class = pd.DataFrame([{
                        'autoID':autoID
                        , 'SeniorCity':SeniorCity
                        , 'Partner':Partner
                        , 'Dependents':Dependents
                        , 'Service1':Service1
                        , 'Service2':Service2
                        , 'Security':Security
                        , 'OnlineBackup':OnlineBackup
                        , 'DeviceProtection':DeviceProtection
                        , 'TechSupport':TechSupport
                        , 'Contract':Contract
                        , 'PaperlessBilling':PaperlessBilling
                        , 'PaymentMethod':PaymentMethod
                        , 'Charges':Charges
                        , 'Demand':Demand
                        ,'Class':''
                            }])
    
    X, y                    = data_to_class(df_new_class)
    df_new_classicated, df_new_classicated_consolidated      = fn_classification(df_new_class, X)

    return {
        'classification':df_new_classicated['Class'].tolist(),
        'df_full': df_new_classicated[-1:].to_dict(orient='records')
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)