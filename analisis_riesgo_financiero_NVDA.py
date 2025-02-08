#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Paquetes a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 
from scipy.stats import t
get_ipython().system('pip install yfinance')
import yfinance as yf

# Descarga de los datos de NVDA (NVIDIA)
data = yf.download("NVDA", start="2015-01-01", end="2023-01-01")
data.head()  # Nos muestra los primeros registros

#  Primeras filas 
print(data.head())

# Información general 
print(data.info())

# Resumen estadístico
print(data.describe())

# Verificamos valores nulos
print(data.isnull().sum())

#Columnas del DataFrame
print(data.columns)

# Acceso a la columna 'Close' de NVDA
close_nvda = data[('Close', 'NVDA')]

# Tipo de dato
print(close_nvda.dtype)

# Valores nulos
print(close_nvda.isnull().sum())

# Cálculo de los retornos logarítmicos
data[('Return', 'NVDA')] = np.log(data[('Close', 'NVDA')] / data[('Close', 'NVDA')].shift(1))

# Resultados
print(data[[('Close', 'NVDA'), ('Return', 'NVDA')]].head())

# Gráfico del precio de cierre a lo largo del tiempo
plt.figure(figsize=(10,6))
plt.plot(data[('Close', 'NVDA')], color='black', label='Precio de Cierre')

# Configuración del gráfico
plt.title('Precio de Cierre de NVIDIA (NVDA) (2015-2023)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.show()

# Gráfico del volumen de transacciones
plt.figure(figsize=(10,6))
plt.plot(data[('Volume', 'NVDA')], color='red', label='Volumen de transacción')

# Configuración del gráfico
plt.title('Volumen de Transacción de NVIDIA (NVDA) (2015-2023)')
plt.xlabel('Fecha')
plt.ylabel('Volumen de Transacción')
plt.legend()
plt.show()

# Calcular la matriz de correlación
corr = data.corr()

# Mostrar la matriz de correlación como un mapa de calor
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Configuración del título
plt.title('Matriz de Correlación - NVIDIA (NVDA)')
plt.show()

# Gráfico de los retornos logarítmicos a lo largo del tiempo
plt.figure(figsize=(10,6))
plt.plot(data.index, data[('Return', 'NVDA')], label='Retornos Diarios', color='purple')

# Configuración del gráfico
plt.title('Retornos Diarios de NVIDIA (NVDA) (2015-2023)')
plt.xlabel('Fecha')
plt.ylabel('Retorno Logarítmico')
plt.legend()
plt.show()

# Gráfico de la distribución de los retornos diarios
plt.figure(figsize=(10,6))
sns.histplot(data[('Return', 'NVDA')].dropna(), bins=100, kde=True, color='purple')

# Configuración del gráfico
plt.title('Distribución de los Retornos Diarios de NVIDIA (NVDA)')
plt.xlabel('Retorno Diario Logarítmico')
plt.ylabel('Frecuencia')
plt.show()

# Crear el Q-Q plot para los retornos de NVIDIA
stats.probplot(data[('Return', 'NVDA')].dropna(), dist="norm", plot=plt)

# Configuración del gráfico
plt.title('Q-Q Plot para los Retornos Diarios de NVIDIA (NVDA)')
plt.show()

# Cálculo de la volatilidad (desviación estándar móvil de 30 días)
data[('Volatility', 'NVDA')] = data[('Return', 'NVDA')].rolling(window=30).std()

# Gráfico de la volatilidad
plt.figure(figsize=(10,6))
plt.plot(data.index, data[('Volatility', 'NVDA')], label='Volatilidad (30 días)', color='red')

# Configuración del gráfico
plt.title('Volatilidad de NVIDIA (NVDA) (2015-2023)')
plt.xlabel('Fecha')
plt.ylabel('Volatilidad')
plt.legend()
plt.show()

# Cálculo de sesgo (asimetría) y curtosis para NVIDIA (NVDA)
skewness = data[('Return', 'NVDA')].skew()
kurtosis = data[('Return', 'NVDA')].kurtosis()

# Mostrar los resultados
print(f"Sesgo (Skewness): {skewness}")
print(f"Curtosis (Kurtosis): {kurtosis}")

# Método Histórico
var_95 = np.percentile(data[('Return', 'NVDA')].dropna(), 5)
print(f"VaR Histórico al 95%: {var_95:.2%}")

mean = data[('Return', 'NVDA')].mean()
std_dev = data[('Return', 'NVDA')].std()
var_parametrico_95 = mean - 1.65 * std_dev
print(f"VaR Paramétrico al 95%: {var_parametrico_95:.2%}")

df = 5  # Grados de libertad, ajustable según los datos
simulations = t.rvs(df, loc=mean, scale=std_dev, size=10000)
var_montecarlo_t_95 = np.percentile(simulations, 5)
print(f"VaR Monte Carlo con t al 95%: {var_montecarlo_t_95:.2%}")

# Cálculo del backtesting para método histórico
violations_historic = data[('Return', 'NVDA')] < -var_95
num_violations_historic = violations_historic.sum()
print(f"Número de violaciones del VaR: {num_violations_historic}")

# Gráfico del backtesting de VaR Histórico
plt.figure(figsize=(10,6))
plt.plot(data.index, data[('Return', 'NVDA')], label='Retornos Diarios', color='yellow')
plt.axhline(y=-var_95, color='r', linestyle='--', label='VaR Histórico (95%)')
plt.scatter(data.index[violations_historic], data[('Return', 'NVDA')][violations_historic], color='orange', label='Violaciones')

# Configuración del gráfico
plt.title('Backtesting del VaR Histórico (95%) - NVIDIA (NVDA)')
plt.xlabel('Fecha')
plt.ylabel('Retorno Diario')
plt.legend()
plt.show()

# Cálculo del backtesting para el método paramétrico
violations_parametrico = data[('Return', 'NVDA')] < -var_parametrico_95
num_violations_parametrico = violations_parametrico.sum()
print(f"Número de violaciones del VaR Paramétrico: {num_violations_parametrico}")

# Gráfico del backtesting de VaR Paramétrico
plt.figure(figsize=(10,6))
plt.plot(data.index, data[('Return', 'NVDA')], label='Retornos Diarios', color='blue')
plt.axhline(y=-var_parametrico_95, color='r', linestyle='--', label='VaR Paramétrico (95%)')
plt.scatter(data.index[violations_parametrico], data[('Return', 'NVDA')][violations_parametrico], color='black', label='Violaciones')

# Configuración del gráfico
plt.title('Backtesting del VaR Paramétrico (95%) - NVIDIA (NVDA)')
plt.xlabel('Fecha')
plt.ylabel('Retorno Diario')
plt.legend()
plt.show()

# Cálculo del backtesting para el método de simulación de Monte Carlo
violations_montecarlo = data[('Return', 'NVDA')] < -var_montecarlo_t_95
num_violations_montecarlo = violations_montecarlo.sum()
print(f"Número de violaciones del VaR Monte Carlo: {num_violations_montecarlo}")

# Gráfico del backtesting de VaR Monte Carlo
plt.figure(figsize=(10,6))
plt.plot(data.index, data[('Return', 'NVDA')], label='Retornos Diarios', color='brown')
plt.axhline(y=-var_montecarlo_t_95, color='r', linestyle='--', label='VaR Monte Carlo (95%)')
plt.scatter(data.index[violations_montecarlo], data[('Return', 'NVDA')][violations_montecarlo], color='green', label='Violaciones')

# Configuración del gráfico
plt.title('Backtesting del VaR Monte Carlo (95%) - NVIDIA (NVDA)')
plt.xlabel('Fecha')
plt.ylabel('Retorno Diario')
plt.legend()
plt.show()

# Cálculo del Expected Shortfall para VaR Histórico
ES_historic = data[('Return', 'NVDA')][data[('Return', 'NVDA')] < -var_95].mean()
print(f"Expected Shortfall (ES) Histórico al 95%: {ES_historic:.2%}")

# Cálculo del Expected Shortfall para VaR Paramétrico
ES_parametric = data[('Return', 'NVDA')][data[('Return', 'NVDA')] < -var_parametrico_95].mean()
print(f"Expected Shortfall (ES) Paramétrico al 95%: {ES_parametric:.2%}")

# Cálculo del Expected Shortfall para Monte Carlo
ES_montecarlo = np.mean(simulations[simulations < -var_montecarlo_t_95])
print(f"Expected Shortfall (ES) Monte Carlo al 95%: {ES_montecarlo:.2%}")

# Gráfica de VaR y Expected Shortfall para los tres métodos
plt.figure(figsize=(10,6))
plt.plot(data.index, data[('Return', 'NVDA')], label='Retornos Diarios', color='gray', alpha=0.6)
plt.axhline(y=var_95, color='r', linestyle='--', label='VaR Histórico (95%)')
plt.axhline(y=var_parametrico_95, color='g', linestyle='--', label='VaR Paramétrico (95%)')
plt.axhline(y=var_montecarlo_t_95, color='b', linestyle='--', label='VaR Monte Carlo (95%)')
plt.axhline(y=ES_historic, color='r', linestyle=':', label='ES Histórico (95%)')
plt.axhline(y=ES_parametric, color='g', linestyle=':', label='ES Paramétrico (95%)')
plt.axhline(y=ES_montecarlo, color='b', linestyle=':', label='ES Monte Carlo (95%)')

# Configuración del gráfico
plt.legend()
plt.title('Comparación del VaR y Expected Shortfall - NVIDIA (NVDA)')
plt.xlabel('Fecha')
plt.ylabel('Retorno Diario')
plt.show()

