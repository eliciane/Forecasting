
import pyodbc
import pandas as pd
from funcoes import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import math
import matplotlib.pyplot as plt

# Entrar com a base de dados
MasterLP = pd.read_sql(sql,cnxn)

dataframe = pd.read_sql(sql,cnxn)
dataframe.info()




#AGRUPAR A QUANTIDADE POR DIA E SOMAR
dataframepos = (dataframe.groupby('DATA_DESEJADA_REM').QTDE_OV.sum().reset_index())

print(dataframepos[:30])
print(dataframepos[30:])
print(dataframepos.info())
#observação o dataframe inicia em 2016-01-12 e termina em 2022-01-05 e está com 1304 linhas.
#se contarmos os dias corridos neste período serão: 2.185 dias
#devemos criar um novo índice para interporlar o dias faltantes nos dias demanda
novo_indice = pd.date_range(start = '2016-01-12', end = '2022-01-05', freq='D')
print(novo_indice)

#transformar o dataframe em data time
dataframepos['DATA_DESEJADA_REM']=pd.to_datetime(dataframepos['DATA_DESEJADA_REM'])
pd.to_datetime(['12/01/2016', '13/01/2016', '14/01/2016'], format='%d/%m/%Y')
print(dataframepos.info())

#transformando em indice:
dataframepos.set_index('DATA_DESEJADA_REM', inplace=True)

#check datatype of index
dataframepos.index

#interpolar o novo índice
dataframepos = dataframepos.reindex(novo_indice, fill_value=0)
dataframepos.info()

#passando para frequencia mensal (MS) e com a data sempre no primeiro dia do mês
dataframeposmen = dataframepos.groupby(pd.Grouper(freq="MS")).sum()
dataframeposmen.info()

#Elimina o indice para poder particionar o dado.
dataframeposmen.reset_index(drop=True, inplace=True)
print(dataframeposmen)

#transformando o dataframe em array para normalizar os dados e depois particionar
dataset = dataframeposmen.values
dataset = dataset.astype('float32')
print(dataset)

# normalizar dados e criar uma coluna para os dados normalizados
scaler_dsmen, dataset = Normalizar_dsmen(dataset)
print (dataset)

#separação dos dados: treinamento e validação
train_size = int(len(dataset) * 0.67)
print('Tamanho train arrays:', train_size)
print(train_size)
test_size = len(dataset) - train_size
print('Tamanho train arrays:', test_size)
print(test_size)
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

#gerar os vetores numpy de treinamento e validação
lookback = 12
X_train, y_train = makeXy2(train, lookback)
print('Shape of train arrays:', X_train.shape, y_train.shape)
X_val, y_val = makeXy2(test, lookback)
print('Shape of validation arrays:', X_val.shape, y_val.shape)

print('X(t)', 'Y(t+1)')
for i in range(20):
  print(X_train[i], y_train[i])

# reshape entrada na forma 3D [samples, time steps, features]
# X-train original eh vetor 2D [[a,b,c,d,f],.......,[1,2,3,4,5]]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

#criar a rede LSTM com os parâmetros a seguir
model = Sequential()

print("dimensaao dados entrada", X_train.shape[-1])

model.add(LSTM(4, input_shape=(1, lookback)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

#treinamento
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)

# fazer as predicoes
trainPredict = model.predict(X_train)
testPredict = model.predict(X_val)

# inverter antes as predicoes
trainPredict = scaler_dsmen.inverse_transform(trainPredict)
y_train = scaler_dsmen.inverse_transform([y_train])
testPredict = scaler_dsmen.inverse_transform(testPredict)
y_val = scaler_dsmen.inverse_transform([y_val])

# calcular o RMSE error root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_val[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# calcular o MAPE
trainScore_MAPE = (mean_absolute_percentage_error(y_train[0], trainPredict[:,0]))
print('Train Score: %.2f MAPE' % (trainScore_MAPE))
testScore_MAPE = (mean_absolute_percentage_error(y_val[0], testPredict[:,0]))
print('Test Score: %.2f MAPE' % (testScore_MAPE))

# calcular o MAE
trainScore_MAE = (mean_absolute_error(y_train[0], trainPredict[:,0]))
print('Train Score: %.2f MAE' % (trainScore_MAE))
testScore_MAE = (mean_absolute_error(y_val[0], testPredict[:,0]))
print('Test Score: %.2f MAE' % (testScore_MAE))


# shift train predictions for plotting
#dataset = dataframeposmen.values
#print(dataset.shape)
#dataset = dataset.reshape(dataset.shape[0],1)
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
print(trainPredictPlot)
trainPredictPlot[lookback:len(trainPredict)+lookback, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(lookback*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler_dsmen.inverse_transform(dataset), color="blue")
plt.plot(trainPredictPlot, color="green")
plt.plot(testPredictPlot, color="red")
plt.show()

dataset_pred = scaler_dsmen.inverse_transform(dataset)
print(dataset_pred)

print(trainPredict)

print(testPredict)


