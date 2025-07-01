import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import mean_squared_error, r2_score

# Configuration de la page
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

@st.cache_data
def load_data():
    # Téléchargement des données depuis UCI
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features
    y = wine_quality.data.targets
    df = X.copy()
    df['quality'] = y
    return df

@st.cache_resource
def load_model():
    # Exemple avec un modèle simple si vous n'avez pas de modèle sauvegardé
    from sklearn.ensemble import RandomForestRegressor
    df = load_data()
    X = df.drop('quality', axis=1)
    y = df['quality']
    model = RandomForestClassifier(n_estimators = 200, min_samples_split = 2, min_samples_leaf = 2, max_depth= 20, bootstrap= False)
    model.fit(X, y)
    return model

# Chargement des données
df = load_data()
model = load_model()

# Sidebar
st.sidebar.header("Configuration")
show_data = st.sidebar.checkbox("Afficher les données brutes", value=False)
show_analysis = st.sidebar.checkbox("Afficher l'analyse exploratoire", value=True)

# En-tête
st.title("🍷 Prédiction de la Qualité du Vin")
st.markdown("""
Cette application prédit la qualité du vin (0-10) à partir de ses caractéristiques physico-chimiques.
Données provenant de l'[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).
""")

# Section principale
tab1, tab2 = st.tabs(["Prédiction", "Analyse"])

with tab1:
    st.header("Faire une prédiction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.slider("Acidité fixe", float(df['fixed_acidity'].min()), float(df['fixed_acidity'].max()), float(df['fixed_acidity'].median()))
        volatile_acidity = st.slider("Acidité volatile", float(df['volatile_acidity'].min()), float(df['volatile_acidity'].max()), float(df['volatile_acidity'].median()))
        citric_acid = st.slider("Acide citrique", float(df['citric_acid'].min()), float(df['citric_acid'].max()), float(df['citric_acid'].median()))
        residual_sugar = st.slider("Sucre résiduel", float(df['residual_sugar'].min()), float(df['residual_sugar'].max()), float(df['residual_sugar'].median()))
        chlorides = st.slider("Chlorures", float(df['chlorides'].min()), float(df['chlorides'].max()), float(df['chlorides'].median()))
        
    with col2:
        free_sulfur = st.slider("Dioxyde de soufre libre", float(df['free_sulfur_dioxide'].min()), float(df['free_sulfur_dioxide'].max()), float(df['free_sulfur_dioxide'].median()))
        total_sulfur = st.slider("Dioxyde de soufre total", float(df['total_sulfur_dioxide'].min()), float(df['total_sulfur_dioxide'].max()), float(df['total_sulfur_dioxide'].median()))
        density = st.slider("Densité", float(df['density'].min()), float(df['density'].max()), float(df['density'].median()))
        ph = st.slider("pH", float(df['pH'].min()), float(df['pH'].max()), float(df['pH'].median()))
        sulphates = st.slider("Sulfates", float(df['sulphates'].min()), float(df['sulphates'].max()), float(df['sulphates'].median()))
        alcohol = st.slider("Alcool", float(df['alcohol'].min()), float(df['alcohol'].max()), float(df['alcohol'].median()))
    
    # Préparation des données pour la prédiction
    input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, 
                              residual_sugar, chlorides, free_sulfur, 
                              total_sulfur, density, ph, sulphates, alcohol]],
                            columns=df.columns[:-1])
    
    if st.button("Prédire la qualité"):
        prediction = model.predict(input_data)[0]
        quality = round(prediction)
        
        st.subheader(f"Résultat de la prédiction : {quality}/10")
        
        # Afficher une interprétation
        if quality >= 7:
            st.success("Ce vin est prédit comme étant de haute qualité!")
        elif 5 <= quality < 7:
            st.warning("Qualité moyenne prédite")
        else:
            st.error("Qualité basse prédite")
        
        # Feature importance
        with st.expander("Importance des caractéristiques"):
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Caractéristique': df.columns[:-1],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots()
                sns.barplot(x='Importance', y='Caractéristique', data=importance, palette='viridis')
                plt.title("Importance des caractéristiques dans la prédiction")
                st.pyplot(fig)
            else:
                st.info("Ce modèle ne fournit pas d'importance des caractéristiques")

with tab2:
    st.header("Analyse Exploratoire des Données")
    
    if show_data:
        st.subheader("Données Brutes")
        st.dataframe(df)
    
    if show_analysis:
        st.subheader("Distribution de la Qualité")
        fig, ax = plt.subplots()
        sns.countplot(x='quality', data=df, palette='viridis')
        plt.xlabel("Note de Qualité")
        plt.ylabel("Nombre d'échantillons")
        st.pyplot(fig)
        
        st.subheader("Relation Alcool/Qualité")
        fig, ax = plt.subplots()
        sns.boxplot(x='quality', y='alcohol', data=df, palette='viridis')
        plt.xlabel("Note de Qualité")
        plt.ylabel("Taux d'Alcool")
        st.pyplot(fig)
        
        st.subheader("Matrice de Corrélation")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                   annot_kws={"size": 8}, ax=ax)
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**Projet de prédiction de la qualité du vin**  
*Données provenant de l'UCI Machine Learning Repository*  
[Accéder au jeu de données](https://archive.ics.uci.edu/ml/datasets/wine+quality)
""")