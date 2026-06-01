# FoodLens — Analyse Nutritionnelle & IA

Projet IPSSI Master 2 IA & Big Data.  
Application web d'analyse de produits alimentaires basée sur Open Food Facts, avec prédiction du Nutri-Score par un modèle XGBoost.

---

## Prérequis

- **Python 3.10+** via Anaconda
- **Git**

---

## Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd "Open Data"
```

### 2. Créer un environnement Conda

```bash
conda create -n foodlens python=3.11
conda activate foodlens
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
copy .env.example .env
```

Ouvrir `.env` et renseigner les credentials MongoDB (demander à Lila) et une `SECRET_KEY` aléatoire :

```env
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
DB_NAME=foodlens
SECRET_KEY=une_cle_longue_et_aleatoire
```

Pour générer une `SECRET_KEY` solide :
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Lancer l'application

```bash
python run_web.py
```

Ouvrir [http://localhost:8000](http://localhost:8000) dans le navigateur.  
Le serveur se relance automatiquement à chaque modification dans `web/`.

---

## Entraîner le modèle ML

Le modèle XGBoost prédit le Nutri-Score (A/B/C/D/E) à partir des valeurs nutritionnelles.

```bash
python src/train_model.py
```

Durée : 3 à 5 minutes sur le jeu complet (447k produits).  
Le modèle est sauvegardé dans `models/nutriscore_xgb.pkl`.

---

## Évaluer le modèle

```bash
# Échantillon 50 000 produits (défaut, ~30 secondes)
python evaluate_model.py

# Taille personnalisée
python evaluate_model.py -n 20000

# Jeu de test complet (~2 minutes)
python evaluate_model.py --all
```

Affiche dans le terminal : accuracy, matrice de confusion, rapport de classification, importance des features.

**Résultats actuels (jeu complet) : 88.36% accuracy**

---

## Pipeline de données (optionnel)

> Le cluster MongoDB est déjà alimenté — cette étape n'est pas nécessaire pour faire tourner l'app.

Télécharger le CSV Open Food Facts (~3 Go) sur [world.openfoodfacts.org/data](https://world.openfoodfacts.org/data) et le placer dans `data/raw/fr.openfoodfacts.org.products.csv`.

```bash
python src/data_pipeline.py
```

Le pipeline lit le CSV par chunks, filtre les produits incomplets, convertit kJ en kcal si nécessaire, puis importe dans MongoDB par lots de 2000. Il s'arrête automatiquement avant d'atteindre la limite Atlas (512 MB).

---

## Structure du projet

```
Open Data/
├── web/
│   ├── main.py                 # Routes FastAPI, sessions, logique métier
│   ├── templates/              # Pages HTML (Jinja2)
│   │   ├── base.html           # Layout commun (navbar, footer)
│   │   ├── index.html          # Page d'accueil
│   │   ├── search.html         # Recherche produits
│   │   ├── product.html        # Fiche produit
│   │   ├── substitution.html   # Suggestions de substituts
│   │   ├── dashboard.html      # Statistiques globales
│   │   ├── admin.html          # Panel d'administration
│   │   ├── auth.html           # Connexion / inscription
│   │   └── profile.html        # Profil utilisateur
│   └── static/
│       ├── css/style.css       # Styles personnalisés (Tailwind)
│       └── js/app.js           # Scanner code-barres, animations
│
├── src/
│   ├── database.py             # Connexion MongoDB
│   ├── data_pipeline.py        # Import CSV → MongoDB
│   ├── off_api.py              # Client API Open Food Facts
│   ├── substitution.py         # Algorithme de substitution
│   ├── train_model.py          # Entraînement XGBoost
│   └── ml_predictor.py         # Prédiction Nutri-Score
│
├── models/
│   └── nutriscore_xgb.pkl      # Modèle entraîné (non versionné — relancer train_model.py)
│
├── data/
│   ├── raw/                    # CSV brut OFF (non versionné, ~3 Go)
│   └── processed/              # CSV nettoyé intermédiaire
│
├── run_web.py                  # Lance le serveur
├── evaluate_model.py           # Évaluation du modèle en terminal
├── requirements.txt
├── .env.example                # Modèle de configuration
└── .env                        # Variables d'environnement (ne pas committer)
```

---

## Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| Recherche | Par nom ou scan code-barres (BarcodeDetector API) |
| Fiche produit | Nutri-Score, NOVA, Eco-Score, allergènes, additifs, substituts |
| Prédiction IA | Nutri-Score estimé par XGBoost si absent de la base |
| Substitution | 3 passes : catégorie + mots-clés → catégorie seule → catégorie parente |
| Dashboard | Répartition grades, distribution NOVA, additifs |
| Favoris | Sauvegarde par utilisateur |
| Allergènes | Profil utilisateur avec alertes sur les fiches produit |
| Admin | Gestion des rôles, stats globales, top favoris |
| Auth | Sessions signées (itsdangerous) + mots de passe bcrypt |

---

## Rôles

- **Premier compte créé** → admin automatiquement
- **Admin** : accès à `/admin`, peut promouvoir ou rétrograder les autres comptes
- **User** : accès à toutes les fonctionnalités sauf le panel admin

---

## Stack

| Composant | Technologie |
|---|---|
| Backend | FastAPI + Jinja2 |
| Serveur | Uvicorn |
| Base de données | MongoDB Atlas (Free Tier, 512 MB) |
| ML | XGBoost + scikit-learn |
| CSS | Tailwind CSS (CDN) |
| JS | Alpine.js, AOS.js, Chart.js |
| Auth | itsdangerous + bcrypt |
| Données | Open Food Facts — 447 000 produits |
