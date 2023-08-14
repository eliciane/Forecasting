
#importando bibliotecas

import pandas as pd
import seaborn
from dateutil.relativedelta import relativedelta

pd.plotting.register_matplotlib_converters()
from pmdarima import auto_arima

import pmdarima as pm

pd.plotting.register_matplotlib_converters()

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.tools import diff

pd.plotting.register_matplotlib_converters()
from datetime import datetime

import numpy as np
import statsmodels.api as sm

pd.plotting.register_matplotlib_converters()


from matplotlib import pyplot as plt

df_mes = pd.read_excel('C:/Users/eliciane/PycharmProjects/excel/demandaMasterLPmensal_TCC.xlsx')


#ilustrando informações do dataset
print(df_mes.info())
print(df_mes.head())


# os resultados mostram que não temos dados nulos
#verificar se tem dados nulos
print('Resultado Nulo:', df_mes.isnull().sum())

#transformar o dataframe em data time e interpolar o novo índice
df_mes['DATA_DESEJADA_REM']=pd.to_datetime(df_mes['DATA_DESEJADA_REM'])
pd.to_datetime(['01/01/2019', '02/01/2019', '03/01/2019'], format='%d/%m/%Y')
print(df_mes.info())


#transformando em indice:
df_mes = df_mes.set_index('DATA_DESEJADA_REM')

print(df_mes)

#converter a coluna em um ojeto Series para evitar referindo-se a nomes de colunas cada vez que usar os TS
ts = df_mes['QTDE_OV']

#fazer um plot para visualizar o padrão da demanda
plt.figure(figsize=(10, 4))
plt.plot(ts)
plt.axhline(0, linestyle='--', color='k')
plt.title('Produto A - DEMANDA NACIONAL', fontsize=20)
plt.ylabel('VOLUME', fontsize=16)
plt.show()

#fazer um box plot para visualizar a tendencia central e dispersão
plt.figure(figsize=(5.5, 5.5))
plt.plot(ts)
g = seaborn.boxplot(data=ts, showmeans=True)
g.set_title('PRODUTO A - DEMANDA NACIONAL - OUTLIERS')
plt.show()


'''
#graficos da decomposição
plt.rcParams['figure.figsize'] = [20/10, 20/10]
plt.title('MASTER LP - DECOMPOSIÇÃO DA DEMANDA', fontsize=20)
result = seasonal_decompose(MasterLP_mes['QTDE_OV'], model='additive')
result.plot()
plt.show()
'''


def test_stationarity(ts):
    # determinando as estatísticas para analisar se a série é estacionária
    rolmean = pd.Series(ts).rolling(window=12).mean()
    rolstd = pd.Series(ts).rolling(window=12).std()

    # plotar o gráfico das estatísticas
    orig = plt.plot(ts, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')

    # desempenhar o teste Dickey-Fuller
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    # plt.figure(figsize=(20, 10))
    plt.show()


test_stationarity(ts)


#fazendo a diferença entre os lags para período de 12 meses de sazonalizade
diffts = diff(ts, k_diff=1, k_seasonal_diff=True, seasonal_periods=12)
print(diffts)

#fazer o teste estatístico para verificar se ficou estacionário
# Augmented Dickey-Fuller test (ADF Test)
ad_fuller_result = adfuller(diffts)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')
#plt.plot(diffts)

# Obter os parâmetros Gráficos ACF e PACF - ATENÇÃO: o ideal é fazer o ACF E PACF depois da diferenciação
sm.graphics.tsa.plot_acf(diffts.values.squeeze(), lags=20) # You can change the lags value if you want to see more lags
sm.graphics.tsa.plot_pacf(diffts.values.squeeze(), lags=20)
plt.show()

# Separar o datase em treino e teste:
len(ts)
train_size = int(len(ts)* 2/3)
print('tamanho dos dados para treino:', train_size)
train_set = ts[:train_size]
print(train_set)
test_set = ts[train_size:]
print(test_set)
print(len(test_set))
print(len(train_set))

#tentar encontrar os melhores parâmetros com a variável exogena usando autorima
#auto_arima = auto_arima(MasterLP_mes.QTDE_OV, seasonal= True, m=12, trace=True)
#print(auto_arima.summary())


stepwise_model = pm.auto_arima(ts, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())

def future_preds_df(model,series,num_months):
    pred_first = series.index.max()+relativedelta(months=1)
    pred_last = series.index.max()+relativedelta(months=num_months)
    date_range_index = pd.date_range(pred_first,pred_last,freq = 'MS')
    vals = model.predict(n_periods = num_months)
    return pd.DataFrame(vals,index = date_range_index)

preds = future_preds_df(stepwise_model,ts,100)

plt.plot(ts)
plt.plot(preds)
plt.show()

print(stepwise_model.plot_diagnostics())
plt.show()

print('auto-fit order: :', stepwise_model.order)
print('auto-fit seasonal_order: :', stepwise_model.seasonal_order)


#MODELO SARIMA
mod = sm.tsa.statespace.SARIMAX(train_set, order=(0, 0, 0), seasonal_order=(0,1, 0, 12))
results = mod.fit(dis=-1)
print(results.summary())
results.plot_diagnostics(figsize=(15, 12))
plt.show()

#Validar a previsão nos dados de teste e obter a previsão

#obter resíduos e previsão
predictions = results.forecast(len(test_set))
predictions = pd.Series(predictions, index=test_set.index)
residuals = test_set - predictions

#grafico dos residuos
plt.figure(figsize=(10,4))
plt.plot(residuals)
plt.axhline(0, linestyle='--', color='k')
plt.title('Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=16)
#plt.show()

#gráfico da predição
plt.figure(figsize=(10,4))

plt.plot(ts)
plt.plot(predictions)

plt.legend(('Data', 'Predictions'), fontsize=16)

start_date = datetime(2015,1,1)
end_date = datetime(2022,1,1)

plt.title('Vendas Master LP', fontsize=20)
plt.ylabel('Volume', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
plt.show()

#medindo o MAPE
print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/test_set)),4))

#outra forma de encontrar o MAPE e o MAE
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
y_true = ts[58:]
y_pred = predictions
print('MAPE:', mean_absolute_percentage_error(y_true, y_pred))
print('MAE:', mean_absolute_error(y_true, y_pred))

print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))

# outra forma de encontrar o RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test_set, predictions))
print('RSME:', rmse)

#Fazer a previsão para os 36 meses a frente a partir dos dados de teste

#If forecast one day ahead
print(results.forecast(36))
plt.show()

forecast = results.forecast(36)



