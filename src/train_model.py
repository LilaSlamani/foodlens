# -*- coding: utf-8 -*-
"""
Entraînement du modèle XGBoost — prédiction Nutri-Score (A/B/C/D/E).
On lit directement depuis MongoDB, pas besoin du CSV.

Usage :
    python src/train_model.py
"""
import sys, io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

from src.database import get_collection

# Les 7 valeurs nutritionnelles qu'on utilise comme features
FEATURES = [
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
]
GRADES     = ["a", "b", "c", "d", "e"]
MODEL_PATH = Path(__file__).parent.parent / "models" / "nutriscore_xgb.pkl"


def charger_donnees() -> pd.DataFrame:
    print("Chargement des données depuis MongoDB...")
    col = get_collection("products")

    projection = {f: 1 for f in FEATURES}
    projection["nutriscore_grade"] = 1
    projection["energy_100g"]      = 1   # kJ pour convertir si kcal absent
    projection["_id"]              = 0

    docs = list(col.find({"nutriscore_grade": {"$in": GRADES}}, projection))
    df   = pd.DataFrame(docs)
    print(f"  {len(df):,} produits chargés")

    for col_name in FEATURES + ["energy_100g"]:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    # Valeurs hors plage (>100 ou négatives) → NaN, XGBoost les gère nativement
    for col_name in FEATURES:
        if col_name in df.columns:
            df.loc[df[col_name] < 0,   col_name] = np.nan
            df.loc[df[col_name] > 100, col_name] = np.nan

    # Convertir kJ en kcal pour les produits qui n'ont pas de kcal direct
    if "energy_100g" in df.columns and "energy-kcal_100g" in df.columns:
        masque = df["energy-kcal_100g"].isna() & df["energy_100g"].notna()
        df.loc[masque, "energy-kcal_100g"] = df.loc[masque, "energy_100g"] / 4.184
        print(f"  {masque.sum():,} valeurs kcal complétées depuis kJ")
        df.drop(columns=["energy_100g"], inplace=True)

    return df


def entrainer(df: pd.DataFrame):
    # Le label encoder convertit a/b/c/d/e en 0/1/2/3/4
    le = LabelEncoder()
    le.fit(GRADES)
    y = le.transform(df["nutriscore_grade"])

    cols_features = [c for c in FEATURES if c in df.columns]
    X = df[cols_features].values

    print(f"\nDataset : {len(X):,} exemples, {len(cols_features)} features")
    for f in cols_features:
        taux = df[f].notna().mean() * 100
        print(f"  {f:<30} {taux:.1f}% renseigné")

    # Split stratifié 80/20 — on conserve la même répartition de grades dans train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nEntraînement — {len(X_train):,} train / {len(X_test):,} test")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist",   # gère les NaN sans preprocessing
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {acc:.4f} ({acc*100:.2f}%)")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=[g.upper() for g in GRADES]))

    # Sauvegarde du modèle, de l'encodeur et de la liste des features
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "features": cols_features}, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")

    print("\nImportance des features :")
    for feat, imp in sorted(zip(cols_features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<30} {imp:.4f}  ({int(imp * 100)}%)")

    return model, le, cols_features


def main():
    print("=" * 60)
    print("ENTRAÎNEMENT XGBoost — Nutri-Score")
    print("=" * 60)
    df = charger_donnees()
    entrainer(df)
    print("\nEntraînement terminé.")


if __name__ == "__main__":
    main()
