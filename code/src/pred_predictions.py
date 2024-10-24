import os
import numpy as np
import pandas as pd
from datetime import datetime

from src.parameters import Parameters
from src.pred_competition import evaluate_models, model_competition
from src.pred_preprocessing import de_escalate
from src.pred_models import Models

import matplotlib.pyplot as plt

import warnings;
warnings.simplefilter('ignore')

def fn_prediction(df_gold, steps, scales, col_pronos, col_y, steps_pred):
    df_pred = pd.DataFrame()
    df_filtered = df_gold
    test_index = df_filtered[-steps:][col_y]
    pred_index = (pd.to_datetime(df_filtered[-steps_pred:][col_y]) + pd.DateOffset(months=steps_pred)).dt.strftime('%Y-%m-01').reset_index(drop=True)
    
    data_train = df_filtered[:-steps]
    data_test = df_filtered[-steps:]
    
    dict_metrics = pd.DataFrame()
    dict_models = {}
    dict_pred = {}
    
    for scale in scales:
        df_train = data_train[scale]
        df_test = data_test[scale]

        df_filtered_de_escalate = df_filtered[col_pronos].values.astype(float).reshape(-1, 1)

        print("="*7,scale,"="*7)

        dict_metrics_by_scale, \
        dict_models_by_scale, \
        dict_pred_by_scale = evaluate_models(
                                steps, 
                                df_train, 
                                df_test,
                                df_filtered_de_escalate, 
                                col_pronos, 
                                scale
                                )
        
        dict_metrics = pd.concat([dict_metrics,dict_metrics_by_scale], axis = 0)
        dict_models.update(dict_models_by_scale)
        dict_pred.update(dict_pred_by_scale)
        
    # Selección del modelo
    dict_metrics = dict_metrics.reset_index(drop=True)
    _winner_metrics, model = model_competition(
                                        dict_metrics, 
                                        dict_models,dict_pred, 
                                        test_index)
    
    bm = _winner_metrics.pop('model_name')
    if 'neuronalnetwork' in bm:
        pred = Models.pred_neuronalnetwork(df_train, steps_pred, model)
    else:
        pred = model.predict(steps_pred)
    winner_scale = bm.split("_")[-2]+"_"+ bm.split("_")[-1]

    pred = de_escalate(pred, col_pronos, winner_scale, df_filtered_de_escalate, bm)
    
    df_col_pronos = pd.DataFrame({col_pronos:col_pronos,col_y:pred_index.tolist(), 'pred':pred})
    # pred = pred.reset_index(drop=True)
        
    df_pred = pd.concat([df_pred, df_col_pronos], axis = 0)

    return df_pred

def plot_pred(df_macro_vble, df_pred, col_pronos, col_y):
    df_winner_metrics = pd.read_excel(os.path.join(Parameters.results_path,'winner_metrics_pred.xlsx'))[-1:]
    
    df_real = df_macro_vble[[col_y,col_pronos]].rename(columns={col_pronos:'real'})
    df_eval = pd.DataFrame({col_y:eval(df_winner_metrics['pred_index'].iloc[0]), 'eval':eval(df_winner_metrics['pred'].iloc[0])})
    df_forc = df_pred[[col_y,'pred']]
    df_plot = df_real.merge(df_eval, on = col_y , how = 'left' )
    df_plot = pd.concat([df_plot, df_forc]).drop(columns=['pred'])
    df_forc = pd.concat([df_macro_vble[[col_y,col_pronos]][-1:].rename(columns={col_pronos:'pred'}), df_pred[[col_y,'pred']]], axis = 0)
    df_plot = df_plot.merge(df_forc, on = col_y , how = 'left' )
    
    df_plot['date_run']  = pd.to_datetime(datetime.now()).strftime('%Y-%m')
    df_plot['col_pronos'] = col_pronos
    df_plot_0 = pd.read_excel(os.path.join(Parameters.results_path,'df_plot.xlsx'))
    df_plot = pd.concat([df_plot_0, df_plot], axis = 0)
    df_plot.to_excel(os.path.join(Parameters.results_path, 'df_plot.xlsx'), index=False)

    rmse = round(df_winner_metrics['rmse'].iloc[0])
    mape = round(df_winner_metrics['mape'].iloc[0],3) *100
    r2 = round(df_winner_metrics['r2'].iloc[0], 3) * 100

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    with plt.style.context('fivethirtyeight'):

        # Plot real sales data
        df_wm = df_winner_metrics[(df_winner_metrics['col_pronos'] == str(col_pronos))].iloc[-1]
        ax.plot(df_plot[col_y], df_plot['real'], color='blue', label='real')
        ax.plot(df_plot[col_y], df_plot['eval'], color='orange', label='eval')
        ax.plot(df_plot[col_y], df_plot['pred'], color='green', label='pred')
        ax.legend(loc='upper left')
        ax.set_title(f"pred_{df_wm['model_name']}")
        # Añadir texto en una posición específica
        plt.text(1, 1, f'RMSE: {rmse}\nMAPE: {mape}%\nR2: {r2}%', fontsize=12, color='red',bbox=dict(facecolor='lightgray', edgecolor='gray'))

        # Rotate x-axis labels
        ax.tick_params(axis='x', labelrotation=90)
        
        path_visual_pred = os.path.join(Parameters.plots_path,
                        f"pred_{col_pronos}.png")
        plt.savefig(path_visual_pred)
        plt.close()