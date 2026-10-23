
import streamlit as st
import pandas as pd
import time

from data.load_data import load_data
from src.preprocessing import split_data
from src.train_models import train_models


st.set_page_config(
    page_title="Modélisation",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Module 3 — Modélisation et comparaison")

st.write(
    "Ce module entraîne cinq algorithmes de classification "
    "pour prédire si un client souscrira à un dépôt à terme."
)


# 1. Charger les données
try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()


# 2. Choisir la proportion du jeu de test
st.header("1. Paramètres")

test_percentage = st.slider(
    "Pourcentage des données réservé au test",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

test_size = test_percentage / 100


# 3. Afficher les algorithmes
st.header("2. Algorithmes étudiés")

st.write(
    "Nous allons comparer la régression logistique, "
    "l'arbre de décision, KNN, Random Forest et XGBoost."
)

with st.expander("Comprendre les cinq algorithmes"):

    st.markdown("""
    **1. Régression logistique**  
    Estime la probabilité qu'un client souscrive.

    **2. Arbre de décision**  
    Prend des décisions successives à partir des caractéristiques du client.

    **3. KNN**  
    Prédit en fonction des clients les plus proches dans l'espace des variables.

    **4. Random Forest**  
    Combine plusieurs arbres de décision pour produire une prédiction.

    **5. XGBoost**  
    Construit progressivement des arbres qui cherchent à corriger les erreurs précédentes.
    """)


# 4. Lancer l'entraînement
st.header("3. Entraînement des modèles")

if st.button("🚀 Entraîner les cinq modèles", type="primary"):

    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=test_size,
        random_state=42
    )

    st.write(f"Nombre de clients pour l'entraînement : {len(X_train)}")
    st.write(f"Nombre de clients pour le test : {len(X_test)}")

    trained_models = {}
    training_times = {}

    progress = st.progress(0)
    status = st.empty()

    try:
        # Entraîner les modèles un par un
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier

        model_definitions = {
            "Régression logistique": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),
            "Arbre de décision": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        }

        for i, (name, model) in enumerate(model_definitions.items()):

            status.info(f"Entraînement : {name}...")

            from src.preprocessing import create_preprocessor

            pipeline = Pipeline([
                ("preprocessing", create_preprocessor(X_train)),
                ("model", model)
            ])

            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            elapsed = time.perf_counter() - start

            trained_models[name] = pipeline
            training_times[name] = elapsed

            progress.progress((i + 1) / len(model_definitions))

        status.success("Entraînement terminé !")

        # 5. Afficher les durées d'entraînement
        st.header("4. Temps d'entraînement")

        time_df = pd.DataFrame({
            "Modèle": list(training_times.keys()),
            "Temps (secondes)": list(training_times.values())
        })

        st.dataframe(time_df, use_container_width=True)

        st.bar_chart(
            time_df.set_index("Modèle")["Temps (secondes)"]
        )

        # 6. Stocker les résultats dans la session Streamlit
        st.session_state["trained_models"] = trained_models
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["training_times"] = training_times

        st.success(
            "Les modèles sont prêts. Tu peux maintenant ouvrir "
            "le Module 4 pour les évaluer."
        )

    except Exception as e:
        status.error("L'entraînement a échoué.")
        st.exception(e)


# 7. Indiquer si des modèles sont déjà entraînés
if "trained_models" in st.session_state:
    st.info(
        "Des modèles sont disponibles dans la session actuelle. "
        "Passe au Module 4 pour examiner leurs performances."
    )
    