import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from funcoes import *
from datetime import datetime
import numpy as np
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.statespace.tools import diff
pd.plotting.register_matplotlib_converters()
import warnings
import seaborn as sns
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from statsmodels.tools.eval_measures import mse,rmse
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

warnings.filterwarnings("ignore")

#USANDO A VARIÁVEL EXOGENA

df_Exog = pd.read_excel('C:/Users/eliciane.silva/PycharmProjects/Forecast/ProdutoA_Regressor.xlsx')

print(df_Exog)
print(df_Exog.info())

#transformando a data em indice:
df_Exog.set_index('DATA_DESEJADA_REM', inplace=True)
print(df_Exog.head(10))

df_Exog = df_Exog.loc['2017-01-01':'2022-03-01']
print(df_Exog.head())
print(df_Exog.info())

#fazer um box plot para visualizar a tendencia central e dispersão
plt.figure(figsize=(5.5, 5.5))
plt.plot(df_Exog.QTDE_OV)
g = sns.boxplot(data=df_Exog.QTDE_OV, showmeans=True)
g.set_title('Produto A - OUTLIERS')
plt.show()

#fazer um plot para visualizar o padrão da demanda
plt.figure(figsize=(10, 4))
plt.plot(df_Exog.QTDE_OV)
plt.axhline(0, linestyle='--', color='k')
plt.title('Produto A', fontsize=20)
plt.ylabel('VOLUME', fontsize=16)
plt.show()

#graficos da decomposição
test_stationarity(df_Exog['QTDE_OV'])

#fazendo a diferença entre os lags para período de 12 meses de sazonalizade
diffts = diff(df_Exog['QTDE_OV'], k_diff=0, k_seasonal_diff=True, seasonal_periods=12)
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

#tentar encontrar os melhores parâmetros com a variável exogena usando autorima
#auto_arima = auto_arima(BR801msExog['QTDE_OV'], exogenous=BR801msExog[['P_Metas']], seasonal= True, m=12, trace=True)
#print(auto_arima.summary())

# Separar o datase da coluna VOLUME em treino e teste:
len(df_Exog.QTDE_OV)
train_size = int(len(df_Exog['QTDE_OV'])* 2/3)
print('tamanho dos dados para treino:', train_size)
train_set = df_Exog['QTDE_OV'][:train_size]
print(train_set.head())
test_set = df_Exog['QTDE_OV'][train_size:]
print(test_set.head(20))

train_setnew = df_Exog.loc['2017-01-01':'2020-06-01', 'QTDE_OV']
print(train_setnew)
test_setnew = df_Exog.loc['2020-07-01':'2022-03-01', 'QTDE_OV']
print(test_setnew)

# Separar o dataset da coluna P_Metas em treino e teste:
len(df_Exog['P_Metas'])
train_size_metas = int(len(df_Exog['P_Metas'])* 2/3)
print('tamanho dos dados para treino:', train_size_metas)
train_set_metas = df_Exog['P_Metas'][:train_size_metas]
print(train_set_metas)
test_set_metas = df_Exog['P_Metas'][train_size_metas:]
print(test_set_metas)

train_set_metasnew = df_Exog.loc['2017-01-01':'2020-06-01', 'P_Metas']
print(train_set_metasnew)
test_set_metasnew = df_Exog.loc['2020-07-01':'2022-03-01', 'P_Metas']
print(test_set_metasnew)

train_set_umi = df_Exog.loc['2017-01-01':'2020-06-01', 'UmidSolo (% )']
print(train_set_umi)
test_set_umi = df_Exog.loc['2020-07-01':'2022-03-01', 'UmidSolo (% )']
print(test_set_umi)



#MODELO SARIMA COM VARIAVEL EXOGENA
endog = train_setnew
exog = sm.add_constant(train_set_metasnew)

mod = sm.tsa.statespace.SARIMAX(endog, exog=(exog) , order=(0, 1, 1), seasonal_order=(1, 0, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
results = mod.fit(disp=False)
print(results.summary())
results.plot_diagnostics(figsize=(15, 12))
plt.show()

res = results.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res, lags=12, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=12, ax=ax[1])
plt.show()

#obter os valores da predição
exog_forecast = len(test_set_metasnew)

#predictions = results.predict(len(test_set), exog=test_set_metas)
predictions = results.predict(test_setnew.index[0], test_setnew.index[20], exog=(sm.add_constant(test_set_metasnew)))

print('SARIMAX model MSE:{}'.format(mean_squared_error(test_setnew,predictions)))
print('SARIMAX, Model RMSE:{}'.format(rmse(test_setnew, predictions)))

print(predictions)

#FAZER O PLOT APENAS DA PREDIÇÃO DOS DADOS TEST
pd.DataFrame({'test':test_setnew,'pred':predictions}).plot()
plt.show()

forecastexo = df_Exog.loc['2020-07-01':'2022-12-01', 'P_Metas']
print(forecastexo.head(10))
exog_forecast2= sm.add_constant(forecastexo)
print(exog_forecast2)
print(results.forecast((30), exog=(sm.add_constant(forecastexo))))

#obter erros MAPE E MAE
y_true = test_setnew
y_pred = predictions
print('MAPE:', mean_absolute_percentage_error(y_true, y_pred))
print('MAE:', mean_absolute_error(y_true, y_pred))
