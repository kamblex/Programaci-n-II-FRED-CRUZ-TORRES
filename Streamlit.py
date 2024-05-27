import streamlit as st  # Importa la biblioteca Streamlit y la renombra como 'st'
import pandas as pd  # Importa la biblioteca Pandas y la renombra como 'pd'
import seaborn as sns  # Importa la biblioteca Seaborn y la renombra como 'sns'
import matplotlib.pyplot as plt  # Importa la biblioteca Matplotlib y la renombra como 'plt'
from sklearn.linear_model import LinearRegression  # Importa el modelo de regresión lineal de la biblioteca Scikit-Learn
from sklearn.model_selection import train_test_split  # Importa la función train_test_split de Scikit-Learn
from sklearn.metrics import mean_squared_error, r2_score  # Importa las métricas de evaluación de Scikit-Learn

# Título de la aplicación
st.title("Aplicación de Análisis de Datos con Streamlit")

# 1. Lectura de Datos
st.header("Lectura de Datos")
st.text("Usamos el conjunto de datos Iris para esta demostración.")
iris = sns.load_dataset('iris')  # Carga el conjunto de datos Iris desde Seaborn
st.write(iris)  # Muestra los datos en la aplicación

# 2. Resumen de Datos
st.header("Resumen de Datos")
st.write(iris.describe())  # Muestra un resumen estadístico de los datos

# Selección de variables para la regresión
st.header("Visualización de Datos")
selected_x_var = st.selectbox("Selecciona la variable independiente (X)", iris.columns[:-1])  # Permite al usuario seleccionar una variable independiente
selected_y_var = st.selectbox("Selecciona la variable dependiente (Y)", iris.columns[:-1])  # Permite al usuario seleccionar una variable dependiente

# Visualización de los datos seleccionados
fig, ax = plt.subplots()  # Crea una figura y ejes para el gráfico
ax.scatter(iris[selected_x_var], iris[selected_y_var])  # Dibuja un gráfico de dispersión de las variables seleccionadas
ax.set_xlabel(selected_x_var)  # Etiqueta del eje x
ax.set_ylabel(selected_y_var)  # Etiqueta del eje y
ax.set_title(f'Dispersión de {selected_x_var} vs {selected_y_var}')  # Título del gráfico
st.pyplot(fig)  # Muestra el gráfico en la aplicación

# 3. Más Funcionalidades: Boxplot
st.header("Boxplot de las Variables")
selected_feature = st.selectbox("Selecciona una variable para visualizar su boxplot", iris.columns[:-1])  # Permite al usuario seleccionar una variable para visualizar su boxplot
fig, ax = plt.subplots()  # Crea una nueva figura y ejes para el boxplot
sns.boxplot(x=iris[selected_feature], ax=ax)  # Crea un boxplot de la variable seleccionada
ax.set_title(f'Boxplot de {selected_feature}')  # Título del boxplot
st.pyplot(fig)  # Muestra el boxplot en la aplicación

# 4. Técnica estadística: Regresión Lineal
st.header("Regresión Lineal Simple")
X = iris[[selected_x_var]]  # Variable independiente
y = iris[selected_y_var]  # Variable dependiente

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
reg = LinearRegression()
reg.fit(X_train, y_train)  # Entrenar el modelo

# Predicciones
y_pred = reg.predict(X_test)

# Mostrar coeficientes y métricas de la regresión
st.subheader("Resultados de la Regresión")
st.write(f"Coeficiente: {reg.coef_[0]}")  # Coeficiente de la variable independiente
st.write(f"Intercepción: {reg.intercept_}")  # Intercepción de la línea de regresión
st.write(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred)}")  # Error cuadrático medio
st.write(f"R^2: {r2_score(y_test, y_pred)}")  # Coeficiente de determinación

# Visualización de la regresión
fig, ax = plt.subplots()  # Crea una nueva figura y ejes para la visualización
ax.scatter(X_test, y_test, color='black', label='Datos de prueba')  # Dibuja los datos de prueba
ax.plot(X_test, y_pred, color='blue', linewidth=3, label='Línea de regresión')  # Dibuja la línea de regresión
ax.set_xlabel(selected_x_var)  # Etiqueta del eje x
ax.set_ylabel(selected_y_var)  # Etiqueta del eje y
ax.set_title(f'Regresión Lineal: {selected_x_var} vs {selected_y_var}')  # Título de la visualización
ax.legend()  # Muestra la leyenda
st.pyplot(fig)  # Muestra la visualización en la aplicación
