import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

datos = pd.read_excel("./oxigeno.xlsx")
x = datos[["Reduccion de solidos"]]
y = datos[["Reduccion de la demanda de oxigeno"]]

print(datos)
plt.scatter(x,y)
plt.xlabel("Reduccion de solidos")
plt.ylabel("Reduccion de la demanda de oxigeno")

plt.grid()
# plt.show() # Grafica

# Convertir la dataframe a numpy

matriz = datos.to_numpy()
n = len(matriz)

sumatoria_x = np.sum(matriz[:,0])
sumatoria_y = np.sum(matriz[:,1])
sumatoria_producto = np.sum(matriz[:,0] * matriz[:,1])
sumatoria_cuadrado_x = np.sum(matriz[:,0] * matriz[:,0])

print("n: ",n)
print("sumatoria x ", sumatoria_x)
print("sumatoria y ", sumatoria_y)
print("sumatoria xy ", sumatoria_producto)
print("sumatoria x^2 ", sumatoria_cuadrado_x)

b1 = (n * sumatoria_producto - sumatoria_x * sumatoria_y) / (n * sumatoria_cuadrado_x - sumatoria_x * sumatoria_x)

b0 = (sumatoria_y - b1 * sumatoria_x) / n

print("b1 ", b1)
print("b0 ", b0)

# Resultado:  Y = b0 + b1*X




