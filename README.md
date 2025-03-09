# atividade-4-parte-2

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Coletando dados do CoinGecko
url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30'
response = requests.get(url)
data = response.json()

# Extraindo preços
prices = data['prices']
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df['target'] = np.where(df['price'].shift(-1) > df['price'], 1, 0)

# Removendo valores nulos
df.dropna(inplace=True)

# Separando variáveis independentes e dependentes
X = df[['price']]
y = df['target']

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Calculando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')
