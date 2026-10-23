
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing import create_preprocessor


def train_models(X_train, y_train):
    """
    Entraîne cinq modèles de classification.
    Chaque modèle possède son propre pipeline de préparation.
    """

    models = {
        "Régression logistique": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Arbre de décision": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

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

    trained_models = {}

    for name, model in models.items():

        # Créer un préprocesseur indépendant
        preprocessor = create_preprocessor(X_train)

        # Associer la préparation et le modèle
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        # Entraîner le pipeline
        pipeline.fit(X_train, y_train)

        # Sauvegarder le modèle entraîné en mémoire
        trained_models[name] = pipeline

    return trained_models