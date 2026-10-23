
import streamlit as st
import pandas as pd

from data.load_data import load_data
from src.preprocessing import split_data, create_preprocessor


st.set_page_config(
    page_title="Préparation des données",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 Module 2 — Préparation des données")

st.write(
    "Ce module prépare les données avant l'entraînement "
    "des modèles de Machine Learning."
)

# 1. Charger les données
try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()


# 2. Paramètres de séparation
st.header("1. Séparation des données")

st.write(
    "Choisissez la proportion des données réservée au test. "
    "Le reste sera utilisé pour entraîner les modèles."
)

test_percentage = st.slider(
    "Pourcentage des données pour le test",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

test_size = test_percentage / 100

# 3. Séparer les données
try:
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=test_size,
        random_state=42
    )

    st.success("Séparation effectuée avec succès !")

except Exception as e:
    st.error(f"Erreur pendant la préparation : {e}")
    st.stop()


# 4. Résumé de la séparation
st.header("2. Résumé de la séparation")

col1, col2, col3 = st.columns(3)

col1.metric("Données totales", len(df))
col2.metric("Entraînement", len(X_train))
col3.metric("Test", len(X_test))

st.write("**Répartition de la cible dans les données d'entraînement :**")

distribution_train = (
    y_train
    .value_counts()
    .reindex([0, 1], fill_value=0)
    .rename(index={0: "no (0)", 1: "yes (1)"})
    .rename_axis("Résultat")
    .reset_index(name="Nombre de clients")
)

st.dataframe(distribution_train, use_container_width=True)


# 5. Vérifier les features utilisées
st.header("3. Variables utilisées")

st.write(
    "Les variables suivantes seront utilisées pour la prédiction. "
    "La cible y et la variable duration sont exclues."
)

st.dataframe(
    pd.DataFrame({
        "Variable": X_train.columns,
        "Type": X_train.dtypes.astype(str).values
    }),
    use_container_width=True
)

if "duration" not in X_train.columns:
    st.success("La variable duration est bien exclue.")

if "y" not in X_train.columns:
    st.success("La cible y n'est pas présente dans les features.")


# 6. Préparation numérique et catégorielle
st.header("4. Transformation des variables")

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Variables numériques")
    st.write(numeric_features)
    st.caption(
        "Les valeurs manquantes sont remplacées par la médiane, "
        "puis les variables sont standardisées."
    )

with col2:
    st.subheader("Variables catégorielles")
    st.write(categorical_features)
    st.caption(
        "Les valeurs manquantes sont remplacées par la catégorie "
        "la plus fréquente, puis les catégories sont encodées."
    )


# 7. Démonstration de la transformation
st.header("5. Aperçu après transformation")

st.write(
    "Nous appliquons le préprocesseur aux données d'entraînement, "
    "puis nous utilisons les mêmes transformations sur les données "
    "de test."
)

preprocessor = create_preprocessor(X_train)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

col1, col2 = st.columns(2)

col1.metric(
    "Colonnes avant transformation",
    X_train.shape[1]
)

col2.metric(
    "Colonnes après transformation",
    X_train_transformed.shape[1]
)

st.write(
    f"Dimensions après transformation : "
    f"{X_train_transformed.shape[0]} lignes d'entraînement "
    f"et {X_test_transformed.shape[0]} lignes de test."
)

if st.checkbox("Afficher les premières lignes transformées"):

    feature_names = preprocessor.get_feature_names_out()

    transformed_df = pd.DataFrame(
        X_train_transformed[:10].toarray()
        if hasattr(X_train_transformed, "toarray")
        else X_train_transformed[:10],
        columns=feature_names
    )

    st.dataframe(transformed_df, use_container_width=True)


# 8. Résumé
st.header("6. Résumé de la préparation")

st.success(
    "Les données sont séparées et les transformations sont prêtes. "
    "Le jeu de test reste séparé de l'apprentissage des transformations."
)

st.info(
    "Dans le module de modélisation, chaque modèle utilisera "
    "son propre pipeline avec ce préprocesseur afin d'éviter "
    "les fuites de données pendant l'entraînement et la validation."
)