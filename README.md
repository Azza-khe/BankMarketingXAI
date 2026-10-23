
# BankMarketingXAI

## 1. Présentation du projet

BankMarketingXAI est une application web interactive de classification explicable réalisée dans le cadre du module Data Mining à l'ENSI pour l'année universitaire 2026-2027.

L'objectif est de prédire si un client souscrira à un dépôt bancaire à terme à partir de ses caractéristiques et des informations relatives aux campagnes marketing.

L'application utilise Streamlit pour l'interface utilisateur et FastAPI pour exposer un service de prédiction.

## 2. Technologies utilisées

- Python
- Streamlit
- Pandas et NumPy
- Scikit-learn
- XGBoost
- Plotly
- SHAP
- LIME
- FastAPI et Uvicorn

## 3. Modèles de classification

Le projet compare plusieurs modèles :

- Régression logistique
- Arbre de décision
- K plus proches voisins (KNN)
- Random Forest
- XGBoost

Les performances peuvent être évaluées à l'aide de plusieurs métriques : Accuracy, Precision, Recall, F1-score, ROC-AUC et PR-AUC.

## 4. Explicabilité

Le projet utilise SHAP et LIME pour aider à comprendre les prédictions des modèles.

Ces méthodes permettent notamment d'étudier l'influence des variables sur une prédiction et de mieux interpréter les résultats obtenus.

## 5. Installation

Cloner le dépôt :

```bash
git clone URL_DU_DEPOT
cd BankMarketingXAI
```

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Sous Windows, activer l'environnement :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
python -m pip install -r requirements.txt
```

## 6. Lancement de l'application Streamlit

Depuis le dossier principal :

```bash
python -m streamlit run app.py
```

L'application est normalement accessible à l'adresse :

http://localhost:8501

## 7. Lancement de l'API FastAPI

Dans un autre terminal, depuis le dossier principal :

```bash
python -m uvicorn api.main:app --reload
```

L'API est normalement accessible à l'adresse :

http://127.0.0.1:8000

La documentation interactive est disponible ici :

http://127.0.0.1:8000/docs

### Principaux endpoints

- `GET /` : présentation de l'API
- `GET /health` : vérification du fonctionnement
- `GET /models` : liste des modèles disponibles
- `GET /features` : variables attendues et exemple
- `POST /predict` : prédiction pour un client

## 8. Données

Le projet utilise le jeu de données Bank Marketing disponible dans le dépôt UCI Machine Learning Repository.

La variable cible est `y` :

- `yes` : le client a souscrit au dépôt à terme
- `no` : le client n'a pas souscrit au dépôt à terme

La variable `duration` est exclue des variables explicatives afin d'éviter une fuite d'information liée à la durée de l'appel.

## 9. Limites

Les prédictions représentent des estimations statistiques et ne garantissent pas le comportement réel des clients.

Les explications SHAP et LIME décrivent le comportement du modèle ; elles ne démontrent pas nécessairement une relation de causalité.

Les résultats doivent également être interprétés en tenant compte du déséquilibre des classes et des risques de biais liés à certaines variables.

## 10. Auteur

Projet réalisé dans le cadre du module Data Mining 
