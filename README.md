# **Prédiction de la Qualité des Vins - README**

![Vin Rouge](https://img.freepik.com/free-photo/red-wine-glass_1409-6220.jpg)

## **📌 Aperçu du Projet**
Ce projet vise à prédire la qualité des vins à l'aide de techniques d'apprentissage automatique. La qualité du vin est un facteur essentiel pour les consommateurs et les producteurs, influençant sa valeur marchande et son goût. En utilisant des propriétés physico-chimiques des vins, nous construisons des modèles pour classer leur qualité, aidant ainsi les producteurs à optimiser leurs processus et les consommateurs à faire de meilleurs choix.

### **🔍 Jeu de Données**
Le jeu de données utilisé est le **Wine Quality Dataset** du [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality). Il contient des mesures physico-chimiques (acidité, sucre, alcool, etc.) et des notes de qualité (0-10) pour des vins rouges et blancs.

### **🎯 Objectif**
Construire un **modèle d'apprentissage automatique** capable de prédire la qualité du vin en fonction de ses caractéristiques physico-chimiques.

---

## **📊 Prétraitement des Données**
### **1. Gestion des Valeurs Manquantes**
- Aucune valeur manquante détectée (`sns.heatmap(df.isnull())`).

### **2. Détection et Suppression des Valeurs Aberrantes**
- Utilisation de **l'IQR (Intervalle Interquartile)** et d'une **troncature basée sur les percentiles** pour éliminer les valeurs extrêmes.
- Application d'une **transformation de Box-Cox** pour normaliser les caractéristiques asymétriques.

### **3. Équilibrage des Données**
- **SMOTE (Technique de Suréchantillonnage des Minoritaires Synthétiques)** a été utilisé pour équilibrer le jeu de données, car les données originales présentaient des classes déséquilibrées.

### **4. Mise à l'Échelle des Caractéristiques**
- **StandardScaler** appliqué pour normaliser les caractéristiques avant l'entraînement des modèles.

---

## **🤖 Modèles d'Apprentissage Automatique**
Plusieurs modèles ont été entraînés et évalués :

| Modèle | Précision | ROC AUC |
|--------|----------|---------|
| **Régression Logistique** | 0.78 | 0.93 |
| **Forêt Aléatoire** | 0.88 | 0.98 |
| **Arbre de Décision** | 0.82 | 0.95 |
| **Gradient Boosting** | 0.86 | 0.97 |
| **K-Nearest Neighbors (KNN)** | 0.84 | 0.96 |
| **Machine à Vecteurs de Support (SVM)** | 0.80 | 0.94 |

### **🔝 Meilleur Modèle**
- La **Forêt Aléatoire** a obtenu la meilleure précision (**88%**) et le meilleur ROC AUC (**0.98**).

---

## **📈 Évaluation des Modèles**
### **1. Courbes ROC-AUC**
- Évaluation des performances en classification multi-classes.
- La Forêt Aléatoire a obtenu le plus haut AUC (0.98), indiquant une excellente séparabilité entre les classes.

![Courbes ROC](https://miro.medium.com/v2/resize:fit:1400/1*4PdJ2owkDYQw6Q5fE-5l3A.png)

### **2. Matrices de Confusion**
- La **Forêt Aléatoire** a eu le moins de mauvaises classifications.
  
![Matrice de Confusion](https://www.researchgate.net/publication/336402347/figure/fig5/AS:812472659349505@1570719985505/Confusion-matrix-for-Random-Forest-classifier.png)

### **3. Rapports de Classification**
- **Précision, Rappel et F1-Score** ont été calculés pour chaque modèle.
- La **Forêt Aléatoire** a obtenu le meilleur F1-score (0.88).

---

## **🚀 Comment Exécuter le Code**
### **1. Installer les Dépendances**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost imbalanced-learn ucimlrepo
```

### **2. Exécuter le Notebook Jupyter**
```bash
jupyter notebook Wine_Quality_Prediction.ipynb
```

### **3. Bibliothèques Clés Utilisées**
- `numpy`, `pandas` (Manipulation des Données)
- `matplotlib`, `seaborn` (Visualisation)
- `scikit-learn` (Modèles d'Apprentissage Automatique)
- `imbalanced-learn` (SMOTE pour l'Équilibrage)
- `xgboost` (Gradient Boosting)

---

## **📂 Structure du Projet**
```
Prédiction-Qualité-Vins/
│
├── Wine_Quality_Prediction.ipynb  # Notebook Jupyter Principal
├── README.md                      # Documentation du Projet
├── requirements.txt               # Dépendances Python
└── images/                        # Visualisations (optionnel)
```

---

## **📝 Conclusion**
- La **Forêt Aléatoire** est le meilleur modèle pour prédire la qualité du vin (88% de précision).
- **Importance des caractéristiques** : L'alcool, les sulfates et les niveaux d'acidité étaient les facteurs les plus influents.
- Des améliorations futures pourraient inclure **l'optimisation des hyperparamètres** et l'utilisation de **réseaux neuronaux**.

---

## **📜 Licence**
Ce projet est sous licence **MIT**. Voir [LICENSE](LICENSE) pour plus de détails.

---

**👨‍💻 Auteur** : [sobjiolagnol]  
**📅 Dernière Mise à Jour** : Juillet 2024  

---
### **🔗 Liens**
- [Dépôt GitHub](https://github.com/lagnolsobjio/prediction-qualite-vins)
- [Jeu de Données UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---
