
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data.load_data import load_data
from src.preprocessing import split_data
from src.train_models import train_models


app = FastAPI(
    title="BankMarketingXAI API",
    description=(
        "API de prédiction de souscription à un dépôt bancaire "
        "avec plusieurs modèles de classification."
    ),
    version="1.0.0",
)

models_cache = None
feature_columns = None
example_features = None


def initialize_models():
    """Charge les données et entraîne les modèles une seule fois."""
    global models_cache, feature_columns, example_features

    if models_cache is None:
        df = load_data()

        X_train, _, y_train, _ = split_data(
            df,
            test_size=0.2,
            random_state=42,
        )

        models_cache = train_models(X_train, y_train)
        feature_columns = list(X_train.columns)

        # Exemple de caractéristiques pour tester l'API
        example_features = X_train.iloc[0].to_dict()

    return models_cache


class PredictionRequest(BaseModel):
    model_name: str = Field(
        default="Random Forest",
        description="Nom exact du modèle à utiliser",
    )
    features: dict[str, Any]
    threshold: float = Field(default=0.5, ge=0, le=1)


@app.get("/")
def home():
    return {
        "message": "Bienvenue dans l'API BankMarketingXAI",
        "documentation": "/docs",
        "endpoints": ["/health", "/models", "/features", "/predict"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    models = initialize_models()
    return {"models": list(models.keys())}


@app.get("/features")
def get_features():
    initialize_models()

    return {
        "required_features": feature_columns,
        "example_features": {
                key: (
                     None
                    if pd.isna(value)
                    else value.item()
                    if hasattr(value, "item")
                    else value
                )
                for key, value in example_features.items()
       },
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        models = initialize_models()

        if request.model_name not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Modèle inconnu. Modèles disponibles : {list(models.keys())}",
            )

        missing = [
            col for col in feature_columns
            if col not in request.features
        ]

        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Des variables obligatoires sont manquantes.",
                    "missing_features": missing,
                },
            )

        # Garder les variables attendues dans le bon ordre
        row = {
            col: request.features[col]
            for col in feature_columns
        }
        X = pd.DataFrame([row], columns=feature_columns)

        pipeline = models[request.model_name]
        probabilities = pipeline.predict_proba(X)[0]

        probability_yes = float(probabilities[1])
        probability_no = float(probabilities[0])
        prediction = int(probability_yes >= request.threshold)

        return {
            "model": request.model_name,
            "probability_no_subscription": round(probability_no, 4),
            "probability_subscription": round(probability_yes, 4),
            "threshold": request.threshold,
            "prediction": prediction,
            "decision": (
                "Souscription probable"
                if prediction == 1
                else "Pas de souscription probable"
            ),
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(error)}",
        ) from error
