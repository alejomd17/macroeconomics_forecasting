# Librerías principales
# # ====================================================================================================
import pandas as pd
from src.pred_models import Models
from src.parameters import Parameters
import numpy as np
from src.pred_preprocessing import de_escalate
from src.save_results import save_model, save_winner_metrics
# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')


def evaluate_models(steps, df_train, df_test, df_filtered_de_escalate, col_pronos, scale):
    
    model_lasso, pred_lasso = Models.lasso_model(df_train,df_test,steps)
    model_sarima, pred_sarima = Models.sarima_model(df_train,df_test,steps)
    model_rf, pred_rf = Models.randomforest_model(df_train,df_test,steps)
    model_xgb, pred_xgb = Models.xgboost_model(df_train,df_test,steps)
    model_nn, pred_nn = Models.neuronal_network_tf(df_train,df_test,steps)
    
    df_test = de_escalate(df_test, col_pronos, scale, df_filtered_de_escalate)
    pred_lasso = de_escalate(pred_lasso, col_pronos, scale, df_filtered_de_escalate)
    pred_sarima = de_escalate(pred_sarima, col_pronos, scale, df_filtered_de_escalate)
    pred_rf = de_escalate(pred_rf, col_pronos, scale, df_filtered_de_escalate)
    pred_xgb = de_escalate(pred_xgb, col_pronos, scale, df_filtered_de_escalate)
    pred_nn = de_escalate(pred_nn, col_pronos, scale, df_filtered_de_escalate,'neuronalnetwork')
    
    RMSE_sarima = np.sqrt(mean_squared_error(df_test, pred_sarima))
    MAPE_sarima= mean_absolute_percentage_error(df_test, pred_sarima)
    R2_sarima = r2_score(df_test,pred_sarima)
        
    RMSE_lasso = np.sqrt(mean_squared_error(df_test, pred_lasso))
    MAPE_lasso = mean_absolute_percentage_error(df_test, pred_lasso)
    R2_lasso = r2_score(df_test, pred_lasso)
    
    RMSE_rf = np.sqrt(mean_squared_error(df_test, pred_rf))
    MAPE_rf = mean_absolute_percentage_error(df_test, pred_rf)
    R2_rf = r2_score(df_test, pred_rf)
    
    RMSE_xgb = np.sqrt(mean_squared_error(df_test, pred_xgb))
    MAPE_xgb = mean_absolute_percentage_error(df_test, pred_xgb)
    R2_xgb = r2_score(df_test, pred_xgb)
    
    RMSE_nn = np.sqrt(mean_squared_error(df_test, pred_nn))
    MAPE_nn = mean_absolute_percentage_error(df_test, pred_nn)
    R2_nn = r2_score(df_test, pred_nn)
    
    model_list = [col_pronos+'_sarima_'+scale,
                  col_pronos+'_lasso_'+scale, 
                  col_pronos+'_randomforest_'+scale, 
                  col_pronos+'_xgboost_'+scale,
                  col_pronos+'_neuronalnetwork_'+scale
                  ]
    
    rmse_list = [RMSE_sarima, RMSE_lasso, RMSE_rf, RMSE_xgb,RMSE_nn ]
    mape_list = [MAPE_sarima, MAPE_lasso, MAPE_rf, MAPE_xgb,MAPE_nn ]
    r2_list = [R2_sarima, R2_lasso, R2_rf, R2_xgb,R2_nn ]
    
    dict_metrics = pd.DataFrame({'model' : model_list,
                    'rmse' : rmse_list,
                    'mape' : mape_list,
                    'r2' : r2_list})

    dict_models = {col_pronos+'_sarima_'+scale: model_sarima,
                col_pronos+'_lasso_'+scale : model_lasso,
                col_pronos+'_randomforest_'+scale : model_rf,
                col_pronos+'_xgboost_'+scale : model_xgb,
                col_pronos+'_neuronalnetwork_'+scale : model_nn,
                }
    
    dict_pred = {col_pronos+'_sarima_'+scale: [pred_sarima.tolist()],
                col_pronos+'_lasso_'+scale : [pred_lasso.tolist()],
                col_pronos+'_randomforest_'+scale : [pred_rf.tolist()],
                col_pronos+'_xgboost_'+scale : [pred_xgb.tolist()],
                col_pronos+'_neuronalnetwork_'+scale : [pred_nn.tolist()],
                col_pronos+'_test_'+scale: [df_test.tolist()]}
    
    return dict_metrics, dict_models, dict_pred

def model_competition(dict_metrics, dict_models, dict_pred, test_index): # Competicion de Modelos
    for index, row in dict_metrics.iterrows():
        contador=0
        if index == 0:
            best_rmse = dict_metrics['rmse'][0]
            best_mape = dict_metrics['mape'][0]
            best_r2 = dict_metrics['r2'][0]
            best_model = dict_metrics['model'][0]
        else:
            rmse = dict_metrics['rmse'][index]
            mape = dict_metrics['mape'][index]
            r2 = dict_metrics['r2'][index]
            model = dict_metrics['model'][index]

            if rmse < best_rmse:
                contador +=1
            if mape < best_mape:
                contador +=1
            if r2 > best_r2:
                contador +=1

            if contador >= 2:
                best_rmse = rmse
                best_mape = mape
                best_r2 = r2
                best_model = model
                
    winner_metrics = {'model_name':best_model,
                    'rmse': [best_rmse], 
                    'mape' : [best_mape], 
                    'r2':  [best_r2], 
                    'pred': dict_pred[best_model], 
                    'test': dict_pred[best_model.split("_")[0]+'_test_'+best_model.split("_")[-2]+"_"+best_model.split("_")[-1]],
                    'col_pronos': [best_model.split("_")[0]],
                    'model': [best_model.split("_")[1]],
                    'scale': [best_model.split("_")[-1]],
                    'pred_index': [test_index.tolist()]
                    }
     
    save_winner_metrics(winner_metrics, Parameters.results_path, 'regression')

    model = dict_models[best_model]

    save_model(model, Parameters.models_path, best_model)

    return winner_metrics, model
