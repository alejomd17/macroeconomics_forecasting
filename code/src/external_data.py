# !pip install pydataxm
# !pip install suds
# !pip install lxml

from pydataxm import *
import pandas as pd
import datetime as dt
from datetime import datetime
from bs4 import BeautifulSoup
from suds.client import Client
import yfinance as yf
from datetime import time, date, datetime
import requests
from io import BytesIO
import time

class URLConsumer:
    def __init__(self, fecha_inicial, fecha_final):
        self.fecha_inicial  = fecha_inicial
        self.fecha_final    = fecha_final

    def costo_marginal_panama(self):

        url_costo_marginal_panama = f'https://sitioprivado.cnd.com.pa/Estadistica/Download/18?key=VXd9e23Z9JRA5aIUR21R-P8gocoGOMqdvSo79FduN'
        response = requests.get(url_costo_marginal_panama)
 
        if response.status_code == 200:
            excel_data = BytesIO(response.content)
            df_costo_marginal_panama = pd.read_excel(excel_data)

        df_cmp = df_costo_marginal_panama[['MERCADO MAYORISTA DE ELECTRICIDAD DE PANAMA',	'Unnamed: 1']]
        df_cmp.columns = ['año_mes', 'CMP']
        df_cmp = df_cmp.dropna()
        df_cmp = df_cmp[~(df_cmp['año_mes'].str.contains('TOTAL|MESES', case=False).fillna(False))]
        df_cmp['año_mes'] = pd.to_datetime(df_cmp['año_mes']).dt.strftime('%Y-%m').astype(str).str[:7]
        df_cmp = df_cmp[(df_cmp.año_mes >= self.fecha_inicial.strftime('%Y-%m'))
                        & (df_cmp.año_mes <=  self.fecha_final.strftime('%Y-%m'))]
        return df_cmp
    
    def trm_prom_month(self):
        WSDL_URL = 'https://www.superfinanciera.gov.co/SuperfinancieraWebServiceTRM/TCRMServicesWebService/TCRMServicesWebService?WSDL'

        def trm(date): 
            try: 
                client = Client(WSDL_URL, location=WSDL_URL, faults=True) 
                trm = client.service.queryTCRM(date) 
            except Exception as e: 
                return str(e) 
            return trm

        df_TRM = pd.DataFrame()
        for date in pd.date_range(start = self.fecha_inicial, end = self.fecha_final):
            df_temp = pd.DataFrame()
            df_temp['año_mes'] = [date]
            df_temp['trm'] = [trm(date)['value']]
            df_temp.set_index('año_mes', inplace = True)
            df_TRM = pd.concat([df_TRM, df_temp], axis = 0)
        df_TRM.reset_index(inplace=True)
        alpha = []
        for num, i in enumerate(df_TRM.año_mes.dt.month):
            if len(str(i))== 1:
                alpha.append(str(df_TRM.año_mes.dt.year.iloc[num]) + "-0" + str(i) + "-01")
            else:
                alpha.append(str(df_TRM.año_mes.dt.year.iloc[num]) + "-" + str(i)  + "-01")
        df_TRM['año_mes'] = alpha
        df_TRM = df_TRM.groupby(['año_mes'])[['trm']].mean()
        return df_TRM
    
    def embi(self):
        url = "https://bcrdgdcprod.blob.core.windows.net/documents/entorno-internacional/documents/Serie_Historica_Spread_del_EMBI.xlsx"
        embi = pd.read_excel(url)
        embi.columns = embi.iloc[0].tolist()
        embi = embi[1:]
        embi['Fecha'] = pd.to_datetime(embi['Fecha']).dt.strftime('%Y-%m-%d').apply(str)
        embi.set_index('Fecha', inplace=True)
        try:
            embi = embi.drop(columns=['RD-LATINO'])
        except:
            embi = embi
        return embi

    def ipc(self):
        list_columns_month = ['Mes','Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
        list_columns_month_1 = list_columns_month[1:]
        # Obtener la fecha y hora actual
        fecha_actual = datetime.now()

        # Obtener el año actual
        año_actual = fecha_actual.year

        # Obtener el mes actual en letras
        mes_actual = fecha_actual.month-1
        mes_anterior = fecha_actual.month-2
        nombre_mes_anterior = list_columns_month_1[mes_anterior]
        nombre_mes_actual = list_columns_month_1[mes_actual]
        ruta_actual = nombre_mes_actual[:3].lower()+str(año_actual)
        ruta_anterior = nombre_mes_anterior[:3].lower()+str([año_actual-1 if mes_actual == 0 else año_actual][0])

        try:
            url_ipc = f'https://www.dane.gov.co/files/operaciones/IPC/anex-IPC-{ruta_actual}.xlsx'
            url_ipc_last = f'https://www.dane.gov.co/files/operaciones/IPC/anex-IPC-{ruta_anterior}.xlsx'
            if requests.get(url_ipc).status_code == 200:
                df_ipc_temp = pd.read_excel(url_ipc)
                print(url_ipc)
            else: 
                if requests.get(url_ipc_last).status_code == 200:
                    df_ipc_temp = pd.read_excel(url_ipc_last)
                    print(url_ipc_last)
                else: 
                    print(url_ipc_last)
                    raise RuntimeError(f'Failed to connect {url_ipc_last}')
        except Exception as inst:
            print(type(inst))    # the exception type

        df_ipc_temp = df_ipc_temp[df_ipc_temp['Unnamed: 0'].isin(list_columns_month)]
        df_ipc_temp.columns = [str(x).replace(".0","") for x in list(df_ipc_temp.iloc[0])]
        df_ipc_temp = df_ipc_temp[1:].set_index('Mes')

        ipc_temp = df_ipc_temp.copy()
        for year in list(ipc_temp.columns[1:]):
            ipc_temp[year] = (df_ipc_temp[year]/ df_ipc_temp[str(int(year)-1)]) -1
        ipc_temp = ipc_temp[list(ipc_temp.columns[1:])]

        ipc = pd.DataFrame()
        for year in list(ipc_temp.columns):
            for num, month in enumerate(list(ipc_temp.index)):
                ipc_itr = pd.DataFrame()
                ipc_itr['año_mes'] = [datetime(int(year),num+1,1).strftime('%Y-%m')]
                ipc_itr['ipc'] = [ipc_temp[year][ipc_temp[year].index == month].iloc[0]]
                ipc = pd.concat([ipc, ipc_itr], axis= 0)
        ipc = ipc[~(ipc['ipc'].isnull())]
        return ipc
    
    def ipc_minhacienda(self):
        url_ipc_minhacienda = 'https://exceloriginales.blob.core.windows.net/descargas/MACRO_Inflacion.xlsx'
        if requests.get(url_ipc_minhacienda).status_code == 200:
            df_ipc_temp_minhacienda = pd.read_excel(url_ipc_minhacienda)
            print(url_ipc_minhacienda)
        else: 
            print(url_ipc_minhacienda)
            raise RuntimeError(f'Failed to connect {url_ipc_minhacienda}')

        df_ipc_temp_minhacienda['año_mes'] = df_ipc_temp_minhacienda['Fecha'].str[:4] + "-" + df_ipc_temp_minhacienda['Fecha'].str[-2:]
        df_ipc_temp_minhacienda = df_ipc_temp_minhacienda.rename(columns = {'Inflacion':'ipc'})[['año_mes', 'ipc']]
        df_ipc_temp_minhacienda['ipc'] = df_ipc_temp_minhacienda['ipc'] / 100
        return df_ipc_temp_minhacienda
    
class APIConsumer:
    def __init__(self, fecha_inicial, fecha_final):
        self.API_XM         = pydataxm.ReadDB()
        self.fecha_inicial  = fecha_inicial
        self.fecha_final    = fecha_final

    def precio_contratos_energia(self):
        time.sleep(0.01)

        df_xm = self.API_XM.request_data("PrecPromCont"
                                         ,"Sistema"
                                         ,self.fecha_inicial.date()
                                         ,self.fecha_final.date()).drop(columns=['Id'])
        
        df_xm['año_mes'] =  df_xm['Date'].dt.strftime('%Y-%m')
        df_xm = df_xm.groupby(['año_mes'])[['Value']].mean()
        df_xm = df_xm.reset_index()
        df_xm = df_xm.rename(columns = {'Value':'PrecContEner'})
        return df_xm

    def precio_bolsa_energia(self):
        time.sleep(0.01)

        df_xm = self.API_XM.request_data("PPPrecBolsNaci"
                                         ,"Sistema"
                                         ,self.fecha_inicial.date()
                                         ,self.fecha_final.date()).drop(columns=['Id'])
        
        df_xm['año_mes'] =  df_xm['Date'].dt.strftime('%Y-%m')
        df_xm = df_xm.groupby(['año_mes'])[['Value']].mean()
        df_xm = df_xm.reset_index()
        df_xm = df_xm.rename(columns = {'Value':'PrecBolsEner'})
        return df_xm
    
    def px_acciones(self, stock):
        
        df_stocks = pd.DataFrame(pd.date_range(self.fecha_inicial, self.fecha_final, freq='D'), columns=["Fecha"])
        df_stocks['Fecha'] = df_stocks['Fecha'].dt.strftime('%Y-%m-%d').apply(str)
        df_stocks.set_index('Fecha', inplace=True)
        stock_temp = yf.Ticker(stock + '.cl').history(period="max")[['Close']].reset_index()
        stock_temp['Date'] = stock_temp['Date'].dt.strftime('%Y-%m-%d').apply(str)
        stock_temp.columns = ['Fecha', stock]
        stock_temp.set_index('Fecha', inplace=True)
        df_stocks = df_stocks.join(stock_temp)
        df_stocks  = df_stocks.reset_index()
        df_stocks['año_mes'] = pd.to_datetime(df_stocks['Fecha']).dt.strftime('%Y-%m')
        df_stocks = df_stocks.groupby(['año_mes'])[[stock]].mean().reset_index()
        return df_stocks

    def tasa_libre_riesgo(self):
        stock = '^TNX'
        df_stocks = pd.DataFrame(pd.date_range(self.fecha_inicial, self.fecha_final, freq='D'), columns=["Fecha"])
        df_stocks['Fecha'] = df_stocks['Fecha'].dt.strftime('%Y-%m-%d').apply(str)
        df_stocks.set_index('Fecha', inplace=True)
        stock_temp = stock_temp = yf.Ticker(stock).history(period="max")[['Close']].reset_index()
        stock_temp['Date'] = stock_temp['Date'].dt.strftime('%Y-%m-%d').apply(str)
        stock_temp.columns = ['Fecha', stock]
        stock_temp.set_index('Fecha', inplace=True)
        df_stocks = df_stocks.join(stock_temp)
        df_stocks  = df_stocks.reset_index()
        df_stocks['año_mes'] = pd.to_datetime(df_stocks['Fecha']).dt.strftime('%Y-%m')
        df_stocks = df_stocks.groupby(['año_mes'])[[stock]].mean().reset_index()
        df_stocks = df_stocks.rename(columns={stock:'rf'})
        return df_stocks

class ScrappingConsumer:
    def __init__(self, fecha_inicial, fecha_final):
        self.fecha_inicial  = fecha_inicial
        self.fecha_final    = fecha_final

    def enso(self):
        req_headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.8',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
        }

        urls = 'https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php'

        r1 = requests.get(urls, headers = req_headers, verify=False)
        r1 = BeautifulSoup(r1.content, 'lxml')
        r1 = r1.findAll('td',  attrs={'width':'7%'})

        alpha = []
        betha = []
        for i in range(len(r1)):
            if 'color' in str(r1[i]):
                col = str(r1[i]).split("color:")[1][0:3]
                if col == 'red':
                    alpha.append(1)
                elif col == 'blu':
                    alpha.append(2) 
                elif col == 'bla':
                    alpha.append(0)
            else:
                alpha.append(3)
            betha.append(r1[i].text.replace("\n",""))
        gamma = pd.DataFrame({'data':betha,'clase':alpha})

        mes_dic = {'DJF':'01', 'JFM':'02', 'FMA':'03', 'MAM':'04', 'AMJ':'05', 'MJJ':'06', 'JJA':'07', 'JAS':'08', 'ASO':'09', 'SON':'10', 'OND':'11', 'NDJ':'12'}
        miu = pd.DataFrame()

        for i in range(0, len(r1)+1,13):
            j = i+13
            miu = pd.concat([miu,gamma[i:j].reset_index()], axis=1)
            
        miu = miu[['data']].T.reset_index(drop=True)
        delete_index = list(miu[miu[0]== 'Year'].index)
        miu = miu.drop_duplicates()

        miu.columns  = list(miu.iloc[0])
        miu = miu[1:].set_index(['Year'])
        omega = pd.DataFrame()

        for year in list(miu.index):
            for month in list(miu.columns):
                try:
                    omega[year+"-"+mes_dic[month]] = [float(miu.loc[year][month])]
                except:
                    omega[str(int(list(miu.index)[-2])+1)+"-"+mes_dic[month]] = [float(miu.loc[year][month])]
        omega = omega.T.reset_index()
        omega.columns = ['año_mes', 'enso']

        delta = pd.DataFrame()
        for i in range(0, len(r1)+1,13):
            j = i+13
            delta = pd.concat([delta,gamma[i:j].reset_index()], axis=1)

        delta = delta[['clase']].T.reset_index(drop=True)
        delta = delta[~delta.index.isin(delete_index)]
        delta.columns  = list(miu.reset_index().columns)
        delta['Year'] = list(miu.reset_index()['Year'])
        delta = delta.set_index(['Year'])

        iota = pd.DataFrame()
        for year in list(delta.index):
            for month in list(delta.columns):
                try:
                    iota[year+"-"+mes_dic[month]] = [float(delta.loc[year][month])]
                except:
                    iota[str(int(list(miu.index)[-2])+1)+"-"+mes_dic[month]] = [float(delta.loc[year][month])]
        iota = iota.T.reset_index()
        iota.columns = ['año_mes', 'enso']
        sigma = omega.merge(iota, how = 'left', on = 'año_mes')
        sigma.columns = ['año_mes', 'enso', 'fenom_enso']
        df_enso = sigma

        df_enso_last = df_enso.dropna(subset=['enso']).tail(1)
        df_enso_last['año_mes'] = '2024-01'

        df_enso = df_enso[~(df_enso.año_mes == '2024-01')]
        df_enso = pd.concat([df_enso, df_enso_last], axis = 0)
        df_enso = df_enso.sort_values(by= 'año_mes')

        return df_enso