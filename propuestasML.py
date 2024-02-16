"""
Posible maquina aprendiendo, posibles propuestas ganadas y perdidas
"""
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.simplefilter('ignore') # Ignora los warnings 

# Lee la tabla
df = pandas.read_csv("a.csv")
print(df)

cliente = {'Publico':0,'Privado':1}
df['C-TipoCliente'] = df['C-TipoCliente'].map(cliente)
sector = {
    "Endocrinología": 1,
    "Software": 2,
    "Inmunología": 3,
    "Hemostasia": 4,
    "Coagulación": 5,
    "Química Clínica": 6,
    "Quimica Clinica": 7,
    "Hematología": 8,
    "POC/Gases": 9,
    "POC/Urgencia": 10,
    "Orina": 11,
    "POC/Coagulación": 12,
    "Electrolitos": 13,
    "Laboratorio": 14,
    "Sedimento": 15,
    "Iones": 16,
    "Gases": 17,
    "Gases en Sangre": 18,
    "Bacteriología": 19,
    "Microplacas Elisa": 20,
    "Eritro": 21,
    "Electroforesis": 22
    }
df['A-Sector'] = df['A-Sector'].map(sector)

# Variables de la tabla
# features = ['Age', 'Experience', 'Rank', 'Nationality']
features = ["C-CodigoCliente","C-TipoCliente","C-Sucursal","A-Equipo","A-Sector","PC-CostoAccesorios","PC-CostoAlquiler","PC-CostoCCC","PC-CostoReactivos","PC-CostoServicioTecnico","PC-CostoTotal","PC-TotalDeterminaciones","PC-ImporteAccesorios","PC-ImporteAlquiler","PC-ImporteCCC","PC-ImporteReactivos","PC-ImporteServicioTecnico","PC-UtilidadAccesorios","PC-UtilidadAlquiler","PC-UtilidadCCC","PC-UtilidadReactivos","PC-UtilidadServicioTecnico","PC-UtilidadPropuesta"]
X_test = df[features] # Datos de entrada
y_test = df['PC-Ganado']     # Respuesta

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_test, y_test) # Entrena
dtree = dtree.fit(X_test, y_test)

tree.plot_tree(dtree, feature_names=features)

print(dtree.predict([[   1975, 0, 1834, 336, 1, 0, 4564755, 1491099.6375000002, 6650811.585, 1597664.25, 17529984.45, 5120, 0, 0, 3429529.17, 13301623.16, 798832.12, 1, 0, 2.3, 2, 0.5, 1.5999649104081208 ]])) # Evalua 1 true 0 false
#print(dtree.predict([[ 1903, 1, 1834, 275, 1, 0, 795600, 355364.658, 3147444.135, 278460, 5802851.87, 1600, 0, 795600, 479742.29, 4249049.58, 278460, 1, 1, 1.35, 1.35, 1, 1.6543122426788 ]])) # Evalua 1 true 0 false
#print(dtree.predict([[ 1903, 1, 1834, 275, 1, 0, 795600, 355364.658, 3147444.135, 278460, 58000000000028051.87, 1600, 0, 795600, 479742.29, 4249049.58, 278460, 1, 1, 1.35, 1.35, 1, 3 ]])) # Evalua 1 true 0 false

print("[1] significa posible ganado")
print("[0] significa posible perdido")

