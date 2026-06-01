# -*- coding: utf-8 -*-
"""
Chargement du modèle XGBoost et prédiction du Nutri-Score.
Utilisé dans web/main.py pour les produits sans grade officiel.
"""
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "models" / "nutriscore_xgb.pkl"

# Le modèle est chargé une seule fois au premier appel et mis en cache
_cache: dict = {}


def _charger_modele() -> dict:
    if _cache:
        return _cache
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    import joblib
    payload = joblib.load(MODEL_PATH)
    _cache.update(payload)
    return _cache


def predire_grade(produit: dict) -> str | None:
    """
    Prédit le Nutri-Score d'un produit à partir de ses valeurs nutritionnelles.
    Retourne une lettre (a/b/c/d/e) ou None si les données sont trop incomplètes.
    On refuse de prédire si moins de 3 features sur 7 sont disponibles.
    """
    try:
        payload  = _charger_modele()
        model    = payload["model"]
        le       = payload["label_encoder"]
        features = payload["features"]

        # Convertir kJ en kcal si le produit n'a pas de valeur kcal directe
        produit = dict(produit)
        if not produit.get("energy-kcal_100g") and produit.get("energy_100g"):
            try:
                produit["energy-kcal_100g"] = float(produit["energy_100g"]) / 4.184
            except (TypeError, ValueError):
                pass

        valeurs   = []
        n_present = 0
        for f in features:
            v = produit.get(f)
            try:
                v = float(v)
                if v < 0 or v > 100:
                    v = np.nan
                else:
                    n_present += 1
            except (TypeError, ValueError):
                v = np.nan
            valeurs.append(v)

        if n_present < 3:
            return None

        pred  = model.predict(np.array([valeurs]))[0]
        grade = le.inverse_transform([pred])[0]
        return str(grade).lower()

    except Exception:
        return None


def modele_disponible() -> bool:
    return MODEL_PATH.exists()
