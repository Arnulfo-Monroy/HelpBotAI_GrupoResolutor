import streamlit as st
import joblib

# ================================
# 1. Cargar modelo y vectorizador
# ================================
model = joblib.load("helpbot_grupo_model.pkl")
vectorizer = joblib.load("tfidf_grupo.pkl")

# ================================
# 2. Interfaz de la app
# ================================
st.set_page_config(page_title="HelpBot AI - Grupo Resolutor", page_icon="🤖")

st.title("HelpBot AI 🤖")
st.subheader("Predicción del Grupo Resolutor de tickets de Mesa de Ayuda")

st.markdown("Escribe el título y la descripción de un ticket, y el sistema predecirá el **Grupo resolutor** que debería atenderlo.")

# Inputs de usuario
titulo = st.text_input("📌 Ingresa el título del ticket:")
descripcion = st.text_area("📝 Ingresa la descripción del ticket:")

# Botón para clasificar
if st.button("🚀Clasificar Ticket"):
    if titulo.strip() == "" and descripcion.strip() == "":
        st.warning("⚠️ Por favor escribe un título o una descripción.")
    else:
        # Combinar título + descripción
        texto = titulo + " " + descripcion
        texto_vect = vectorizer.transform([texto])
        grupo = model.predict(texto_vect)[0]
        
        st.success(f"✅ Grupo resolutor predicho: **{grupo}**")
