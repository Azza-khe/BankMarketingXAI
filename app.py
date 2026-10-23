
import streamlit as st
from data.load_data import load_data

st.set_page_config(
    page_title="Bank Marketing - Classification",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Bank Marketing — Classification Explicable")

st.write(
    "Application de classification permettant de prédire "
    "si un client souscrira à un dépôt à terme."
)

try:
    df = load_data()

    st.success("Dataset chargé avec succès !")

    st.subheader("Aperçu des données")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Informations générales")

    col1, col2, col3 = st.columns(3)

    col1.metric("Nombre de lignes", df.shape[0])
    col2.metric("Nombre de colonnes", df.shape[1])
    col3.metric("Valeurs manquantes", int(df.isna().sum().sum()))

    st.subheader("Répartition de la variable cible")

    if "y" in df.columns:
        st.write(df["y"].value_counts())
        st.bar_chart(df["y"].value_counts())

    st.warning(
        "Attention : la variable duration devra être exclue "
        "de l'entraînement des modèles pour éviter la fuite de données."
    )

except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")