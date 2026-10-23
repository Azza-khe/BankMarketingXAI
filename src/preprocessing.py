
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_data(df, test_size=0.2, random_state=42):
    """
    Sépare les données en ensembles d'entraînement et de test.
    Exclut la cible et la variable duration des features.
    """

    if "y" not in df.columns:
        raise ValueError("La colonne cible 'y' est absente du dataset.")

    # La cible à prédire
    y = df["y"].map({"no": 0, "yes": 1})

    if y.isna().any():
        raise ValueError("La cible 'y' contient des valeurs inattendues.")

    # Les features : exclure la cible et duration
    X = df.drop(columns=["y", "duration"], errors="ignore")

    # Séparation avant toute transformation apprise sur les données
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor(X):
    """
    Crée un préprocesseur pour les variables numériques
    et catégorielles.
    """

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Préparation des variables numériques
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Préparation des variables catégorielles
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    # Appliquer le bon traitement selon le type de variable
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor