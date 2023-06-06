# Karla Ornelas Gamero

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import base64
from IPython.display import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for

import pickle

# 1) Objetivo
print("\nPredicción de pérdida de clientes")

# 2) Extracción
path = 'C:/xampp/htdocs/proyecto_datos/WA_Fn-UseC_-Telco-Customer-Churn (1).csv'
df_clientes = pd.read_csv(path)

# 3) Análisis
print("Cantidad de datos:", df_clientes.shape)
print("Estadísticas del Dataset:", df_clientes.describe())
print("Cantidad de personas con churn:\n", df_clientes['Churn'].value_counts())

plt.figure(figsize=(6, 4))
df_clientes['Churn'].value_counts().plot(kind='bar')
plt.title('Distribución de Churn')
plt.xlabel('Churn')
plt.ylabel('Cantidad de clientes')
#plt.savefig('churn_distribution.png')
plt.show()

plt.figure(figsize=(6, 4))
df_clientes['MonthlyCharges'].hist(bins=30)
plt.title('Distribución de MonthlyCharges')
plt.xlabel('MonthlyCharges')
plt.ylabel('Frecuencia')
#plt.savefig('monthly_charges_histogram.png')
plt.show()

mean = df_clientes['MonthlyCharges'].mean()
median = df_clientes['MonthlyCharges'].median()
std = df_clientes['MonthlyCharges'].std()

print("Estadísticas descriptivas de MonthlyCharges (cargos mensuales):")
print("Media:", mean)
print("Mediana:", median)
print("Desviación estándar:", std)

correlation_matrix = df_clientes[['MonthlyCharges', 'tenure']].corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.savefig('correlation_heatmap.png')
plt.show()

df_gender = df_clientes[['gender', 'Churn']]
contingency_table = pd.crosstab(df_gender['gender'], df_gender['Churn'])
sns.heatmap(contingency_table, annot=True, fmt='d')
plt.xlabel('Churn')
plt.ylabel('Gender')
plt.title('Tabla de contingencia: Churn vs. Gender')
#plt.savefig('gender_churn_contingency_table.png')
plt.show()
print("Esta tabla muestra la relación entre el género y las personas que cancelaron la suscripción.")

fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
sns.countplot(x='Partner', hue='Churn', data=df_clientes, ax=axes1[0, 0])
sns.countplot(x='Dependents', hue='Churn', data=df_clientes, ax=axes1[0, 1])
sns.countplot(x='InternetService', hue='Churn', data=df_clientes, ax=axes1[1, 0])
sns.countplot(x='OnlineSecurity', hue='Churn', data=df_clientes, ax=axes1[1, 1])
#plt.savefig('subplots1.png')
plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
sns.countplot(x='OnlineBackup', hue='Churn', data=df_clientes, ax=axes2[0, 0])
sns.countplot(x='DeviceProtection', hue='Churn', data=df_clientes, ax=axes2[0, 1])
sns.countplot(x='TechSupport', hue='Churn', data=df_clientes, ax=axes2[1, 0])
sns.countplot(x='StreamingTV', hue='Churn', data=df_clientes, ax=axes2[1, 1])

#plt.savefig('subplots2.png')
plt.tight_layout()
plt.show()

# 4) Pre-Procesamiento

null_values = df_clientes.isnull().sum()
print(null_values)

df_clientes_sin_faltantes = df_clientes.drop('SeniorCitizen', axis=1)

missing_values_after = df_clientes_sin_faltantes.isnull().sum()
print(missing_values_after)

columnas_categoricas = df_clientes_sin_faltantes.select_dtypes(include=['object']).columns

print("Columnas categóricas:")
print(columnas_categoricas)

print("Valores faltantes en columnas categóricas:")
print(df_clientes[columnas_categoricas].isnull().sum())

df_clientes_codificado = pd.get_dummies(df_clientes, columns=columnas_categoricas)

print(df_clientes_codificado.head())

X = df_clientes_codificado.drop(['Churn_No', 'Churn_Yes'], axis=1)
Y = df_clientes_codificado[['Churn_No', 'Churn_Yes']]

escalador = StandardScaler()
X_normalizado = escalador.fit_transform(X)

df_clientes_normalizado = pd.DataFrame(X_normalizado, columns=X.columns)
df_clientes_normalizado[['Churn_No', 'Churn_Yes']] = Y
print(df_clientes_normalizado.head())

# 6) Modelado

X_entrenamiento, X_pruebas, Y_entrenamiento, Y_pruebas = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

clf = DecisionTreeClassifier()
clf.fit(X_entrenamiento, Y_entrenamiento)
Y_pred = clf.predict(X_pruebas)

accuracy = accuracy_score(Y_pruebas, Y_pred)
print("Accuracy:", accuracy)

with open('modelo_predictivo.pkl', 'wb') as file: 
    pickle.dump(clf, file)

classification_report = classification_report(Y_pruebas, Y_pred)
print("Classification Report:")
print(classification_report)

new_customer = pd.DataFrame([['1236-YPRLE', 'Male', 0, 'No', 'No', 1, 'No', 'No phone service', 'DSL', 'No', 'Yes', 'No', 'No', 'No', 'No', 'Month-to-month', 'Yes', 'Credit card (automatic)', 59.85, 120.85, 'No']], columns=df_clientes.columns)

new_customer_encoded = pd.get_dummies(new_customer, columns=columnas_categoricas)
new_customer_encoded = new_customer_encoded.reindex(columns=X.columns, fill_value=0)


# Verificar que los nombres de las columnas sean iguales
columnas_df_clientes_codificado = df_clientes_codificado.columns.tolist()
columnas_new_customer_encoded = new_customer_encoded.columns.tolist()

if columnas_df_clientes_codificado == columnas_new_customer_encoded:
    print("Los nombres de las columnas son iguales.")
else:
    print("Hay diferencias en los nombres de las columnas.")

# Obtener los nombres de las columnas de df_clientes_codificado y new_customer_encoded
columnas_df_clientes_codificado = df_clientes_codificado.columns.tolist()
columnas_new_customer_encoded = new_customer_encoded.columns.tolist()

# Guardar los nombres de las columnas en un archivo de texto
with open('columnas.txt', 'w') as file:
    file.write("Columnas de df_clientes_codificado:\n")
    file.write(str(columnas_df_clientes_codificado))
    file.write("\n\n")
    file.write("Columnas de new_customer_encoded:\n")
    file.write(str(columnas_new_customer_encoded))

# Imprimir un mensaje indicando la ubicación del archivo
print("Se han guardado los nombres de las columnas en el archivo columnas.txt.")

X_new_normalized = escalador.transform(new_customer_encoded)
prediction = clf.predict(X_new_normalized)
print("Predicción de Churn para el nuevo cliente:", prediction)

if prediction.any() == 1:
    print("El nuevo cliente es propenso a cancelar el servicio (Churn).")
else:
    print("El nuevo cliente no es propenso a cancelar el servicio (No Churn).")

app = Flask(__name__, static_folder='static')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        customerID = request.form['customerID']
        gender = request.form['gender']
        SeniorCitizen = request.form['SeniorCitizen']
        Partner = request.form['Partner']
        Dependents = request.form['Dependents']
        tenure = request.form['tenure']
        PhoneService = request.form['PhoneService']
        MultipleLines = request.form['MultipleLines']
        InternetService = request.form['InternetService']
        OnlineSecurity = request.form['OnlineSecurity']
        OnlineBackup = request.form['OnlineBackup']
        DeviceProtection = request.form['DeviceProtection']
        TechSupport = request.form['TechSupport']
        StreamingTV = request.form['StreamingTV']
        StreamingMovies = request.form['StreamingMovies']
        Contract = request.form['Contract']
        PaperlessBilling = request.form['PaperlessBilling']
        PaymentMethod = request.form['PaymentMethod']
        MonthlyCharges = request.form['MonthlyCharges']
        TotalCharges = request.form['TotalCharges']
        Churn = request.form['Churn']

        new_customer = pd.DataFrame([[customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges,Churn]], columns=df_clientes.columns)

        
        new_customer_encoded = pd.get_dummies(new_customer, columns=columnas_categoricas)
        new_customer_encoded = new_customer_encoded.reindex(columns=X.columns, fill_value=0)


        #new_customer_encoded = pd.get_dummies(new_customer, columns=columnas_categoricas)

        # Asegurarse de que todas las columnas categóricas de df_clientes_codificado estén presentes en new_customer_encoded
        #new_customer_encoded = new_customer_encoded.reindex(columns=df_clientes_codificado.columns, fill_value=0)

      

        X_new_normalized = escalador.transform(new_customer_encoded)
        prediction = clf.predict(X_new_normalized)
        print("Predicción de Churn para el nuevo cliente:", prediction)

        if prediction.any() == 1:
            print("El nuevo cliente es propenso a cancelar el servicio (Churn).")
            prediction_text = "El nuevo cliente es propenso a cancelar el servicio (Churn)."
            print(prediction_text)
            return redirect(url_for('prediccion', resultado=prediction_text))
        else:
            print("El nuevo cliente no es propenso a cancelar el servicio (No Churn).")
            prediction_text = "El nuevo cliente no es propenso a cancelar el servicio (No Churn)."
            print(prediction_text)
            return redirect(url_for('prediccion', resultado=prediction_text))

    return render_template('index.html')

@app.route('/prediccion/<resultado>', methods=['GET'])
def prediccion(resultado):
    return render_template('prediccion.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
