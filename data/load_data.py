
import pandas as pd
import streamlit as st
from ucimlrepo import fetch_ucirepo


@st.cache_data
def load_data():
    """
    Télécharge le dataset Bank Marketing depuis UCI
    et retourne les variables explicatives avec la cible.
    """

    dataset = fetch_ucirepo(id=222)

    # Récupérer les variables explicatives
    features = dataset.data.features.copy()

    # Récupérer la variable cible
    targets = dataset.data.targets.copy()

    # Assembler les données dans un seul DataFrame
    df = pd.concat([features, targets], axis=1)

    # Nettoyer les noms des colonnes
    df.columns = df.columns.str.strip()

    return df
