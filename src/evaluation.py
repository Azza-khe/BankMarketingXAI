
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def evaluate_models(models, X_test, y_test):
    """
    Évalue les modèles entraînés sur le même jeu de test.
    La classe positive correspond à y = 1 (yes).
    """

    results = []
    confusion_matrices = {}
    roc_curves = {}

    for name, model in models.items():

        # Prédictions des classes
        y_pred = model.predict(X_test)

        # Probabilité estimée de souscription
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calcul des métriques
        results.append({
            "Modèle": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "PR-AUC": average_precision_score(y_test, y_proba)
        })

        # Matrice de confusion
        confusion_matrices[name] = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        )

        # Scores nécessaires pour tracer la courbe ROC
        roc_curves[name] = {
            "y_true": y_test,
            "y_proba": y_proba
        }

    results_df = pd.DataFrame(results)

    return results_df, confusion_matrices, roc_curves