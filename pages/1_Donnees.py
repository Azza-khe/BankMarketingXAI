
import streamlit as st
import pandas as pd
import plotly.express as px

from data.load_data import load_data


# Configuration de la page
st.set_page_config(
    page_title="Exploration des données",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Module 1 — Exploration des données")

st.write(
    "Ce module permet de comprendre le dataset Bank Marketing, "
    "d'examiner ses variables et d'identifier les éventuels problèmes "
    "avant la préparation des données."
)


# Chargement des données
try:
    df = load_data()

    st.success("Dataset chargé avec succès !")

except Exception as e:
    st.error(f"Impossible de charger le dataset : {e}")
    st.stop()


# 1. Présentation générale
st.header("1. Présentation générale")

col1, col2, col3 = st.columns(3)

col1.metric("Nombre de lignes", df.shape[0])
col2.metric("Nombre de colonnes", df.shape[1])
col3.metric(
    "Valeurs manquantes",
    int(df.isna().sum().sum())
)

st.write(
    "Chaque ligne représente un client associé à une campagne "
    "marketing. Les colonnes décrivent ses caractéristiques, "
    "les informations de contact et le résultat de la souscription."
)


# 2. Aperçu du dataset
st.header("2. Aperçu du dataset")

nombre_lignes = st.slider(
    "Nombre de lignes à afficher",
    min_value=5,
    max_value=50,
    value=10
)

st.dataframe(
    df.head(nombre_lignes),
    use_container_width=True
)


# 3. Informations sur les variables
st.header("3. Informations sur les variables")

informations = pd.DataFrame({
    "Variable": df.columns,
    "Type": df.dtypes.astype(str).values,
    "Valeurs distinctes": [
        df[col].nunique() for col in df.columns
    ],
    "Valeurs manquantes": [
        df[col].isna().sum() for col in df.columns
    ]
})

st.dataframe(informations, use_container_width=True)


# 4. Distribution de la variable cible
st.header("4. Distribution de la variable cible")

if "y" in df.columns:

    distribution = (
        df["y"]
        .value_counts()
        .rename_axis("Résultat")
        .reset_index(name="Nombre de clients")
    )

    fig_cible = px.bar(
        distribution,
        x="Résultat",
        y="Nombre de clients",
        color="Résultat",
        title="Répartition des souscriptions",
        text="Nombre de clients"
    )

    st.plotly_chart(fig_cible, use_container_width=True)

    st.write(
        "Cette distribution permet de comparer le nombre de clients "
        "ayant souscrit à un dépôt à terme avec celui des clients "
        "n'ayant pas souscrit."
    )

    st.info(
        "Si les deux classes sont très déséquilibrées, nous devrons "
        "en tenir compte lors de l'évaluation des modèles."
    )


# 5. Distribution des variables numériques
st.header("5. Distribution des variables numériques")

colonnes_numeriques = df.select_dtypes(
    include="number"
).columns.tolist()

if colonnes_numeriques:

    variable = st.selectbox(
        "Choisir une variable numérique",
        colonnes_numeriques
    )

    fig_histogramme = px.histogram(
        df,
        x=variable,
        nbins=30,
        title=f"Distribution de {variable}",
        marginal="box"
    )

    st.plotly_chart(
        fig_histogramme,
        use_container_width=True
    )


# 6. Analyse des valeurs manquantes
st.header("6. Analyse des valeurs manquantes")

valeurs_manquantes = (
    df.isna()
    .sum()
    .reset_index()
)

valeurs_manquantes.columns = [
    "Variable",
    "Nombre de valeurs manquantes"
]

valeurs_manquantes = valeurs_manquantes[
    valeurs_manquantes["Nombre de valeurs manquantes"] > 0
]

if valeurs_manquantes.empty:
    st.success("Aucune valeur manquante détectée.")
else:
    st.dataframe(
        valeurs_manquantes,
        use_container_width=True
    )


# 7. Attention à la fuite de données
st.header("7. Vérification de la fuite de données")

if "duration" in df.columns:
    st.warning(
        "La variable duration représente la durée de l'appel en "
        "secondes. Elle ne sera pas utilisée pour entraîner les "
        "modèles, car elle n'est connue qu'après l'appel."
    )

st.success("Exploration des données terminée.")