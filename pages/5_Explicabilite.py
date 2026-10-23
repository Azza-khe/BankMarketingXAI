
import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px

from data.load_data import load_data
from src.preprocessing import split_data
from src.train_models import train_models


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Explicabilité des modèles",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Module 5 — Explicabilité des modèles")
st.write(
    "Ce module permet de comprendre quelles variables influencent "
    "les prédictions des modèles de classification."
)


# ============================================================
# Charger les modèles et les données
# ============================================================

if (
    "trained_models" not in st.session_state
    or "X_test" not in st.session_state
    or "y_test" not in st.session_state
):

    st.warning(
        "Les modèles ne sont pas disponibles dans cette session. "
        "Il faut les entraîner avant de réaliser les explications."
    )

    if st.button("Charger les données et entraîner les modèles"):

        with st.spinner("Chargement et entraînement en cours..."):

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

        st.rerun()

    st.stop()


models = st.session_state["trained_models"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]


# ============================================================
# Sélection du modèle
# ============================================================

st.header("1. Choix du modèle")

selected_model = st.selectbox(
    "Choisir le modèle à expliquer",
    list(models.keys())
)

pipeline = models[selected_model]

# Récupérer les deux étapes du pipeline
preprocessor = pipeline.named_steps["preprocessing"]
classifier = pipeline.named_steps["model"]


# ============================================================
# Préparer les données pour SHAP
# ============================================================

st.header("2. Préparation des données")

st.write(
    "Les données passent d'abord par le même préprocesseur que celui "
    "utilisé pendant l'entraînement : traitement des valeurs manquantes, "
    "normalisation des variables numériques et encodage des variables "
    "catégorielles."
)

# Transformer les variables du jeu de test
X_transformed = preprocessor.transform(X_test)

# Convertir en tableau dense si le résultat est creux
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)

# Récupérer les noms des variables après transformation
feature_names = preprocessor.get_feature_names_out()

# Échantillon limité pour éviter des calculs trop longs
max_rows = min(30, len(X_transformed))
max_background = min(30, len(X_transformed))

st.write(f"Nombre de clients disponibles pour l'explication : {len(X_test)}")
st.write(f"Nombre de variables après transformation : {len(feature_names)}")

st.caption(
    "Pour limiter le temps de calcul, SHAP utilise ici un échantillon "
    "du jeu de test. Les explications sont donc estimées sur cet échantillon."
)


# ============================================================
# Choisir le type d'explication
# ============================================================

st.header("3. Générer les explications SHAP")

explanation_type = st.radio(
    "Type d'explication",
    [
        "Importance globale des variables",
        "Explication d'un client"
    ],
    horizontal=True
)

if st.button("Calculer les explications SHAP", type="primary"):

    with st.spinner(
        "Calcul des valeurs SHAP en cours. Cela peut prendre un moment..."
    ):

        # Choisir les observations de référence et à expliquer
        background = X_transformed[:max_background]
        data_to_explain = X_transformed[:max_rows]

        # Fonction donnant la probabilité de souscription
        def predict_probability(data):
            return classifier.predict_proba(data)[:, 1]

        # Explainer indépendant du type de classifieur
        explainer = shap.Explainer(
            predict_probability,
            background,
            algorithm="permutation"
        )

        max_evals = 2 * X_transformed.shape[1] + 1

        shap_values = explainer(
            data_to_explain,
            max_evals=max_evals
        )

        # Enregistrer les résultats dans la session
        st.session_state["shap_values"] = shap_values
        st.session_state["shap_feature_names"] = list(feature_names)
        st.session_state["shap_data"] = data_to_explain
        st.session_state["shap_model_name"] = selected_model

    st.success("Les explications SHAP ont été calculées !")


# ============================================================
# Afficher les résultats
# ============================================================

if (
    "shap_values" in st.session_state
    and st.session_state.get("shap_model_name") == selected_model
):

    shap_values = st.session_state["shap_values"]
    feature_names = st.session_state["shap_feature_names"]
    data_to_explain = st.session_state["shap_data"]

    st.header("4. Résultats et interprétation")

    if explanation_type == "Importance globale des variables":

        st.subheader("Variables les plus influentes")

        # Importance moyenne : moyenne des valeurs absolues SHAP
        importance = np.abs(shap_values.values).mean(axis=0)

        importance_df = pd.DataFrame({
            "Variable": feature_names,
            "Importance SHAP": importance
        })

        importance_df = importance_df.sort_values(
            "Importance SHAP",
            ascending=False
        ).head(15)

        fig = px.bar(
            importance_df.sort_values(
                "Importance SHAP",
                ascending=True
            ),
            x="Importance SHAP",
            y="Variable",
            orientation="h",
            title="Top 15 des variables les plus influentes"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            "Plus l'importance SHAP moyenne d'une variable est élevée, "
            "plus cette variable influence les probabilités prédites "
            "par le modèle sur l'échantillon analysé."
        )

        st.caption(
            "L'importance SHAP mesure l'ampleur de l'influence, "
            "pas si cette influence augmente ou diminue la probabilité."
        )

    else:

        st.subheader("Explication individuelle")

        client_index = st.slider(
            "Choisir un client dans l'échantillon expliqué",
            min_value=0,
            max_value=len(data_to_explain) - 1,
            value=0,
            step=1
        )

        values = shap_values.values[client_index]

        local_df = pd.DataFrame({
            "Variable": feature_names,
            "Contribution SHAP": values
        })

        local_df["Importance absolue"] = (
            local_df["Contribution SHAP"].abs()
        )

        local_df = local_df.sort_values(
            "Importance absolue",
            ascending=False
        ).head(15)

        local_df["Effet"] = np.where(
            local_df["Contribution SHAP"] > 0,
            "Augmente la probabilité prédite",
            np.where(
                local_df["Contribution SHAP"] < 0,
                "Diminue la probabilité prédite",
                "Effet nul"
            )
        )

        fig_local = px.bar(
            local_df.sort_values(
                "Contribution SHAP",
                ascending=True
            ),
            x="Contribution SHAP",
            y="Variable",
            color="Effet",
            orientation="h",
            title="Contributions SHAP pour le client sélectionné"
        )

        st.plotly_chart(
            fig_local,
            use_container_width=True
        )

        # Retrouver les données originales de ce client
        original_position = client_index
        client = X_test.iloc[original_position]

        probability = classifier.predict_proba(
            data_to_explain[client_index:client_index + 1]
        )[0, 1]

        st.metric(
            "Probabilité estimée de souscription",
            f"{probability:.1%}"
        )

        st.write("Caractéristiques du client :")

        st.dataframe(
            client.to_frame(name="Valeur").rename_axis("Variable"),
            use_container_width=True
        )

        st.write(
            "Une contribution SHAP positive augmente le score prédit "
            "par rapport à la valeur de référence de l'explication. "
            "Une contribution négative le diminue. Ces contributions "
            "expliquent le modèle ; elles ne prouvent pas une relation "
            "de causalité dans la réalité."
        )

else:

    st.info(
        "Clique sur « Calculer les explications SHAP » pour afficher "
        "les résultats du modèle sélectionné."
    )
