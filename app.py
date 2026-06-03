import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load("modelo_churn.pkl")
sc = joblib.load("escalador.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

st.title("Predicción de Riesgo de Abandono de Clientes")

st.write("Ingrese los datos del cliente para predecir si abandonará el servicio.")

gender = st.selectbox("Género", ["Female", "Male"])
SeniorCitizen = st.selectbox("Adulto mayor", [0, 1])
Partner = st.selectbox("Tiene pareja", ["Yes", "No"])
Dependents = st.selectbox("Tiene dependientes", ["Yes", "No"])
tenure = st.number_input("Meses con el servicio", min_value=0, value=1)
PhoneService = st.selectbox("Servicio telefónico", ["Yes", "No"])
MultipleLines = st.selectbox("Múltiples líneas", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Servicio de internet", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Seguridad en línea", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Respaldo en línea", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Protección del dispositivo", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Soporte técnico", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming películas", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Facturación electrónica", ["Yes", "No"])
PaymentMethod = st.selectbox("Método de pago", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Cargo mensual", min_value=0.0, value=50.0)
TotalCharges = st.number_input("Cargo total", min_value=0.0, value=100.0)

nuevo_cliente = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

nuevo_cliente = pd.get_dummies(nuevo_cliente)

nuevo_cliente = nuevo_cliente.reindex(
    columns=columnas_modelo,
    fill_value=0
)

if st.button("Predecir", key="boton_predecir_individual"):

    nuevo_cliente_scaled = sc.transform(nuevo_cliente)

    prediccion = modelo.predict(nuevo_cliente_scaled)
    probabilidad = modelo.predict_proba(nuevo_cliente_scaled)

    if prediccion[0] == 1:
        st.error("El cliente tiene riesgo de abandonar el servicio.")
        st.write("Probabilidad de abandono:", round(probabilidad[0][1] * 100, 2), "%")

        st.subheader("Recomendaciones")
        st.write("Contactar al cliente.")
        st.write("Ofrecer una promoción.")
        st.write("Revisar inconformidades.")
        st.write("Proponer cambio de contrato o beneficio.")

        resultado_texto = f"""
REPORTE DE PREDICCIÓN DE CHURN

Resultado:
El cliente tiene riesgo de abandonar el servicio.

Probabilidad:
{round(probabilidad[0][1] * 100, 2)} %

Recomendaciones:
- Contactar al cliente
- Ofrecer una promoción
- Revisar inconformidades
- Proponer cambio de contrato o beneficio
"""

        st.download_button(
            "📄 Descargar Diagnóstico",
            resultado_texto,
            "diagnostico_churn.txt"
        )

    else:
        st.success("El cliente no tiene riesgo alto de abandono.")
        st.write("Probabilidad de no abandono:", round(probabilidad[0][0] * 100, 2), "%")

        st.subheader("Recomendaciones")
        st.write("Mantener seguimiento normal.")
        st.write("Conservar beneficios actuales.")

        resultado_texto = f"""
REPORTE DE PREDICCIÓN DE CHURN

Resultado:
El cliente NO tiene riesgo alto de abandono.

Probabilidad:
{round(probabilidad[0][0] * 100, 2)} %

Recomendaciones:
- Mantener seguimiento normal
- Conservar beneficios actuales
"""

        st.download_button(
            "📄 Descargar Diagnóstico",
            resultado_texto,
            "diagnostico_churn.txt"
        )