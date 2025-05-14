# 🏡 ImmoEliza: Application de Prédiction du Prix de l'Immobilier

## Aperçu
ImmoEliza est une application web conviviale qui prédit le prix des biens immobiliers en Belgique en utilisant un modèle d'apprentissage automatique (XGBoost Regressor). Les utilisateurs peuvent saisir les caractéristiques d'une propriété via une interface simple et recevoir instantanément une estimation du prix.

## Pile technologique:
* XGBoost pour la régression
* scikit-learn
* Streamlit pour l'interface web
* pandas, numpy

## 🚀 Démarrage
1.  **Cloner le dépôt**
    ```bash
    git clone [https://github.com/yourusername/immoeliza.git](https://github.com/yourusername/immoeliza.git)
    cd immoeliza
    ```

2.  **Installer les dépendances**
    Nous recommandons d'utiliser un environnement virtuel (facultatif mais encouragé) :
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
    ```
    Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
    (Si vous n'avez pas de `requirements.txt`, voir ci-dessous pour une liste pip-ready.)

3.  **Préparer les données**
    Placez votre fichier CSV de données principal nommé `data.csv` dans le dossier du projet (il doit contenir les données de propriété attendues par `cleaning_dataset.py`).

4.  **Entraîner le modèle**
    Entraîner et sérialiser le modèle et le scaler (créera `model.pkl` et `scaler.pkl`) :
    ```bash
    python XGBoost_model.py
    ```

5.  **Lancer l'application web**
    ```bash
    streamlit run app.py
    ```
    Visitez l'URL fournie par Streamlit (généralement `http://localhost:8501/`) dans votre navigateur.

## 🖥️ Structure du Projet
immoeliza/
│
├── app.py                 # Application web Streamlit (point d'entrée principal)
├── XGBoost_model.py       # Script d'entraînement et de sérialisation du modèle
├── cleaning_dataset.py    # Nettoyage des données et ingénierie des fonctionnalités
├── model.pkl              # (Généré) Modèle XGBoost sérialisé
├── scaler.pkl             # (Généré) Scaler scikit-learn sérialisé
├── data.csv               # Votre source de données CSV
├── requirements.txt       # (Recommandé) Dépendances Python
└── README.md              # Ce fichier !


## 🏗 Fonctionnalités
* Prédit le prix de l'immobilier en fonction de dizaines de caractéristiques (localisation, nombre de pièces, surfaces, énergie, extras, ...)
* Interface utilisateur en temps réel utilisant Streamlit — très facile à utiliser !
* Prétraitement intelligent des données :
    * Gère les valeurs manquantes
    * Applique l'ingénierie des fonctionnalités et les mappings
    * Utilise uniquement les fonctionnalités pertinentes pour le modèle

## ⚙️ Prérequis
Exigences minimales :
```
streamlit
pandas
numpy
xgboost
scikit-learn
```

## 📝 Comment ça marche ?
1.  **Saisie Utilisateur :** Vous remplissez le formulaire web décrivant votre propriété.
2.  **Prétraitement :** Les caractéristiques saisies sont formatées, transformées et nettoyées selon les besoins du modèle.
3.  **Mise à l'échelle :** Les données sont mises à l'échelle avec le même scaler que lors de l'entraînement du modèle.
4.  **Prédiction :** Le modèle XGBoost entraîné produit un prix estimé.
5.  **Résultat :** Le prix estimé apparaît instantanément à l'écran !

## 🧑‍💻 Notes du Développeur
* Vous pouvez réentraîner ou ajuster le modèle en modifiant `XGBoost_model.py`.
* Ajoutez de nouvelles fonctionnalités/données — alignez simplement les noms de colonnes et les types de données avec le code de prétraitement dans `cleaning_dataset.py`.
* Besoin d'ajouter des pages ou d'étendre l'interface utilisateur ? Streamlit facilite la création d'applications multi-pages (voir la documentation) !

## 📬 Contribution
Les *pull requests* sont les bienvenues ! Pour les changements majeurs, veuillez d'abord ouvrir une *issue* pour discuter d
