
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer

from data.load_data import load_data
from src.preprocessing import split_data
from src.train_models import train_models

st.set_page_config(
    page_title="Explicabilité LIME",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Module 6 — Explicabilité avec LIME")

st.write(
    "LIME explique une prédiction individuelle en identifiant "
    "les variables qui influencent localement le résultat du modèle."
)

# Vérifier si les modèles sont disponibles
if (
    "trained_models" not in st.session_state
    or "X_test" not in st.session_state
    or "y_test" not in st.session_state
):
    st.warning(
        "Les modèles ne sont pas disponibles dans cette session. "
        "Entraînons-les avant de générer les explications."
    )

    if st.button("Charger les données et entraîner les modèles"):
        with st.spinner("Chargement des données et entraînement..."):
            df = load_data()

            X_train, X_test, y_train, y_test = split_data(
                df,
                test_size=0.2,
                random_state=42
            )

            models = train_models(X_train, y_train)

            st.session_state["trained_models"] = models
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["X_train"] = X_train

        st.rerun()

    st.stop()

models = st.session_state["trained_models"]
X_test = st.session_state["X_test"]

# Recréer X_train si nécessaire
if "X_train" not in st.session_state:
    df = load_data()
    X_train, _, _, _ = split_data(
        df,
        test_size=0.2,
        random_state=42
    )
    st.session_state["X_train"] = X_train

X_train = st.session_state["X_train"]

st.header("1. Choisir le modèle")

model_name = st.selectbox(
    "Modèle à expliquer",
    list(models.keys())
)

pipeline = models[model_name]
preprocessor = pipeline.named_steps["preprocessing"]
classifier = pipeline.named_steps["model"]

st.header("2. Choisir un client")

client_index = st.number_input(
    "Numéro du client dans le jeu de test",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1
)

st.write("Caractéristiques du client sélectionné :")
st.dataframe(
    X_test.iloc[[int(client_index)]],
    use_container_width=True
)

# Préparer les données dans le même format que le modèle
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

if hasattr(X_train_transformed, "toarray"):
    X_train_transformed = X_train_transformed.toarray()

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

X_train_transformed = np.asarray(X_train_transformed)
X_test_transformed = np.asarray(X_test_transformed)

feature_names = preprocessor.get_feature_names_out()

st.header("3. Générer l'explication LIME")

st.write(
    "LIME crée des exemples artificiels proches du client choisi, "
    "observe les prédictions du modèle, puis construit une explication "
    "locale simplifiée."
)

if st.button("Expliquer ce client avec LIME", type="primary"):

    with st.spinner("Calcul de l'explication LIME..."):

        explainer = LimeTabularExplainer(
            training_data=X_train_transformed,
            feature_names=list(feature_names),
            class_names=["Pas de souscription", "Souscription"],
            mode="classification",
            discretize_continuous=False,
            random_state=42
        )

        instance = X_test_transformed[int(client_index)]

        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=classifier.predict_proba,
            num_features=15,
            num_samples=1000
        )

        explanation_list = explanation.as_list()

        explanation_df = pd.DataFrame(
            explanation_list,
            columns=["Variable", "Contribution LIME"]
        )

        st.session_state["lime_explanation_df"] = explanation_df
        st.session_state["lime_client_index"] = int(client_index)
        st.session_state["lime_model_name"] = model_name

if (
    "lime_explanation_df" in st.session_state
    and st.session_state.get("lime_client_index") == int(client_index)
    and st.session_state.get("lime_model_name") == model_name
):

    explanation_df = st.session_state["lime_explanation_df"]

    st.header("4. Résultats de l'explication")

    fig = px.bar(
        explanation_df.sort_values("Contribution LIME"),
        x="Contribution LIME",
        y="Variable",
        orientation="h",
        color="Contribution LIME",
        color_continuous_midpoint=0,
        title="Influence locale des variables selon LIME"
    )

    st.plotly_chart(fig, use_container_width=True)

    probabilities = classifier.predict_proba(
        X_test_transformed[int(client_index):int(client_index) + 1]
    )[0]

    st.metric(
        "Probabilité estimée — pas de souscription",
        f"{probabilities[0]:.1%}"
    )

    st.metric(
        "Probabilité estimée — souscription",
        f"{probabilities[1]:.1%}"
    )

    st.subheader("Comment interpréter le résultat ?")

    st.write(
        "Une contribution LIME positive favorise localement la classe "
        "associée à l'explication, tandis qu'une contribution négative "
        "la défavorise. Les valeurs décrivent l'explication locale "
        "approximative de LIME et ne sont pas des probabilités."
    )

    st.caption(
        "L'explication LIME est une approximation locale. "
        "Elle peut varier selon les paramètres et les exemples générés."
    )

else:
    st.info(
        "Choisis un modèle et un client, puis clique sur "
        "« Expliquer ce client avec LIME »."
    )
