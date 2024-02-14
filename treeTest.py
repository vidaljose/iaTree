"""
¿Debería ir a ver un programa protagonizado por un comediante
 estadounidense de 40 años, con 10 años de experiencia y 
 un ranking de comedia de 7?
"""

 
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.simplefilter('ignore') # Ignora los warnings 

# Lee la tabla
df = pandas.read_csv("data.csv")
print(df)

# Se transforma los datos de texto en numeros
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

# Variables de la tabla
features = ['Age', 'Experience', 'Rank', 'Nationality']
X_test = df[features] # Datos de entrada
y_test = df['Go']     # Respuesta

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_test, y_test) # Entrena

print(dtree.predict([[40, 10, 7, 1]])) # Evalua 1 true 0 false

print("[1] means 'GO'")
print("[0] means 'NO'")

