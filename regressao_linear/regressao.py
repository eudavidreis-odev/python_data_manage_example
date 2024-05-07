import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

############# Pré-processamento ###############
# Coleta e Integração
arquivo = pd.read_csv('./regressao_linear/dados_dengue.csv')

anos = arquivo[['ano']]
casos = arquivo[['casos']]

############## Mineração #################
regr = LinearRegression()
regr.fit(anos, casos)  # Ajustando o modelo sem fornecer nomes de características

ano_futuro = [[2018]]
casos_2018 = regr.predict(ano_futuro)[0]

print('Casos previstos para 2018 ->', int(casos_2018[0]))

############ Pós-processamento ################
plt.scatter(anos, casos, color='black')
plt.scatter(ano_futuro, casos_2018, color='red')
plt.plot(anos, regr.predict(anos), color='blue')

plt.xlabel('Anos')
plt.ylabel('Casos de dengue')
plt.xticks([2018])

# Correção do aviso de depreciação
plt.yticks([int(casos_2018[0])])  # Extraído o valor único do array

plt.show()
