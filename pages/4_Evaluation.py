
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

from data.load_data import load_data
from src.preprocessing import split_data
from src.train_models import train_models
from src.evaluation import evaluate_models


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Évaluation des modèles",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Module 4 — Évaluation des modèles")

st.write(
    "Compare les performances des cinq algorithmes "
    "sur un jeu de test indépendant."
)


# ============================================================
# Récupérer les modèles de la session si disponibles
# ============================================================

if "trained_models" not in st.session_state:

    st.warning(
        "Aucun modèle n'est disponible dans cette session. "
        "Charge les données et entraîne les modèles ci-dessous."
    )

    if st.button("Entraîner les modèles maintenant"):

        with st.spinner("Chargement et entraînement des modèles..."):

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


# ============================================================
# Récupérer les données de test et les modèles
# ============================================================

models = st.session_state["trained_models"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]


# ============================================================
# Calculer les métriques et les probabilités
# ============================================================

results_df, confusion_matrices, roc_curves = evaluate_models(
    models,
    X_test,
    y_test
)


# ============================================================
# 1. Tableau comparatif des performances
# ============================================================

st.header("1. Comparaison des performances")

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-score": "{:.3f}",
        "ROC-AUC": "{:.3f}",
        "PR-AUC": "{:.3f}"
    }),
    use_container_width=True
)


# ============================================================
# 2. Comparaison graphique
# ============================================================

st.header("2. Comparaison graphique")

metric = st.selectbox(
    "Choisir une métrique",
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC"
    ]
)

fig_bar = px.bar(
    results_df,
    x="Modèle",
    y=metric,
    color="Modèle",
    title=f"Comparaison des modèles — {metric}",
    range_y=[0, 1]
)

st.plotly_chart(fig_bar, use_container_width=True)


# ============================================================
# 3. Matrice de confusion
# ============================================================

st.header("3. Matrice de confusion")

selected_model = st.selectbox(
    "Choisir un modèle",
    list(models.keys())
)

cm = confusion_matrices[selected_model]

fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=["Prédit : no", "Prédit : yes"],
    y=["Réel : no", "Réel : yes"],
    labels={
        "x": "Classe prédite",
        "y": "Classe réelle",
        "color": "Nombre de clients"
    },
    title=f"Matrice de confusion — {selected_model}",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig_cm, use_container_width=True)

st.caption(
    "Les lignes représentent les classes réelles et les colonnes "
    "les classes prédites."
)


# ============================================================
# 4. Courbes ROC superposées
# ============================================================

st.header("4. Courbes ROC")

fig_roc = go.Figure()

for name, values in roc_curves.items():

    fpr, tpr, _ = roc_curve(
        values["y_true"],
        values["y_proba"]
    )

    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=name
        )
    )

fig_roc.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Référence aléatoire",
        line=dict(dash="dash")
    )
)

fig_roc.update_layout(
    title="Comparaison des courbes ROC",
    xaxis_title="Taux de faux positifs",
    yaxis_title="Taux de vrais positifs",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig_roc, use_container_width=True)


# ============================================================
# 5. Courbes Precision-Recall
# ============================================================

st.header("5. Courbes Precision-Recall")

fig_pr = go.Figure()

for name, values in roc_curves.items():

    precision, recall, _ = precision_recall_curve(
        values["y_true"],
        values["y_proba"]
    )

    fig_pr.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=name
        )
    )

fig_pr.update_layout(
    title="Comparaison des courbes Precision-Recall",
    xaxis_title="Recall",
    yaxis_title="Precision",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig_pr, use_container_width=True)


# ============================================================
# 6. Résumé automatique
# ============================================================

st.header("6. Résumé automatique")

best_model = results_df.loc[
    results_df[metric].idxmax()
]

st.success(
    f"Meilleur modèle selon {metric} : "
    f"{best_model['Modèle']} "
    f"(score = {best_model[metric]:.3f})."
)

st.info(
    "Le modèle ayant le meilleur score n'est pas forcément le meilleur "
    "pour tous les objectifs. Le choix final doit tenir compte du coût "
    "des faux positifs, des faux négatifs et du besoin métier."
)


# ============================================================
# 7. Interprétation des résultats
# ============================================================

st.header("7. Interprétation des résultats")

st.subheader("Meilleur modèle selon chaque métrique")

for metric_name in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "PR-AUC"
]:

    best = results_df.loc[
        results_df[metric_name].idxmax()
    ]

    st.write(
        f"**{metric_name} :** le meilleur score est obtenu par "
        f"**{best['Modèle']}**, avec un score de "
        f"**{best[metric_name]:.3f}**."
    )

st.divider()

st.subheader("Interprétation de la matrice de confusion")

tn, fp, fn, tp = cm.ravel()

st.write(f"- Vrais négatifs (TN) : {tn} clients.")
st.write(f"- Faux positifs (FP) : {fp} clients.")
st.write(f"- Faux négatifs (FN) : {fn} clients.")
st.write(f"- Vrais positifs (TP) : {tp} clients.")

st.write(
    f"Le modèle a correctement identifié {tp} clients ayant souscrit "
    f"et {tn} clients n'ayant pas souscrit."
)

st.write(
    f"Il a prédit à tort une souscription pour {fp} clients "
    f"et n'a pas détecté {fn} clients ayant réellement souscrit."
)

st.divider()

st.subheader("Conclusion")

best_recall = results_df.loc[
    results_df["Recall"].idxmax()
]

best_f1 = results_df.loc[
    results_df["F1-score"].idxmax()
]

best_pr = results_df.loc[
    results_df["PR-AUC"].idxmax()
]

st.write(
    f"Le modèle ayant le meilleur Recall est "
    f"**{best_recall['Modèle']}**. "
    "Il détecte la plus grande proportion de clients ayant réellement "
    "souscrit, avec le seuil de classification par défaut."
)

st.write(
    f"Le meilleur F1-score est obtenu par **{best_f1['Modèle']}**, "
    "ce qui indique le meilleur équilibre entre Precision et Recall "
    "parmi les modèles comparés."
)

st.write(
    f"Le meilleur PR-AUC est obtenu par **{best_pr['Modèle']}**. "
    "Cette métrique est utile lorsque les classes sont déséquilibrées."
)

st.info(
    "Le choix final doit dépendre de l'objectif métier. "
    "Pour une campagne bancaire, privilégier le Recall peut permettre "
    "de manquer moins de clients susceptibles de souscrire, mais peut "
    "également augmenter le nombre de contacts inutiles."
)


# ============================================================
# 8. Curseur de seuil de décision interactif
# ============================================================

st.header("8. Seuil de décision interactif")

st.write(
    "Modifie le seuil de décision pour observer en direct son effet "
    "sur la précision, le rappel, le F1-score et la matrice de confusion."
)

threshold = st.slider(
    "Seuil de décision",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help=(
        "Une probabilité supérieure ou égale au seuil donne la classe yes."
    )
)

# Récupérer les probabilités du modèle sélectionné
y_true = roc_curves[selected_model]["y_true"]
y_proba = roc_curves[selected_model]["y_proba"]

# Prédire les classes avec le seuil choisi
y_pred_threshold = (y_proba >= threshold).astype(int)

# Recalculer les métriques au nouveau seuil
precision_threshold = precision_score(
    y_true,
    y_pred_threshold,
    zero_division=0
)

recall_threshold = recall_score(
    y_true,
    y_pred_threshold,
    zero_division=0
)

f1_threshold = f1_score(
    y_true,
    y_pred_threshold,
    zero_division=0
)

# Afficher les métriques actualisées
col1, col2, col3 = st.columns(3)

col1.metric(
    "Précision",
    f"{precision_threshold:.3f}"
)

col2.metric(
    "Rappel",
    f"{recall_threshold:.3f}"
)

col3.metric(
    "F1-score",
    f"{f1_threshold:.3f}"
)

st.caption(
    f"Modèle sélectionné : {selected_model} | "
    f"Seuil actuel : {threshold:.2f}"
)

# Recalculer la matrice de confusion au nouveau seuil
cm_threshold = confusion_matrix(
    y_true,
    y_pred_threshold,
    labels=[0, 1]
)

fig_threshold = px.imshow(
    cm_threshold,
    text_auto=True,
    x=["Prédit : no", "Prédit : yes"],
    y=["Réel : no", "Réel : yes"],
    labels={
        "x": "Classe prédite",
        "y": "Classe réelle",
        "color": "Nombre de clients"
    },
    title=(
        f"Matrice de confusion — {selected_model} "
        f"(seuil = {threshold:.2f})"
    ),
    color_continuous_scale="Blues"
)

st.plotly_chart(
    fig_threshold,
    use_container_width=True
)

st.write(
    f"Au seuil de **{threshold:.2f}**, le modèle prédit "
    f"**{int(y_pred_threshold.sum())} clients** comme susceptibles "
    "de souscrire."
)

st.info(
    "Interprétation : diminuer le seuil permet généralement de détecter "
    "davantage de clients susceptibles de souscrire, mais peut augmenter "
    "les faux positifs. Augmenter le seuil rend la prédiction yes plus "
    "exigeante et peut faire manquer davantage de clients intéressés."
)