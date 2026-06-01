#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation du modèle XGBoost Nutri-Score — affichage terminal.

Usage :
    python evaluate_model.py            # échantillon 50 000 produits
    python evaluate_model.py --all      # jeu complet (~447k, ~2 min)
    python evaluate_model.py -n 20000   # taille au choix
"""

import sys, io, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.database import get_collection

# Codes couleur ANSI pour le terminal
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[96m"
GRAY  = "\033[90m"

GRADE_ANSI = {
    "a": "\033[1;32m",
    "b": "\033[32m",
    "c": "\033[33m",
    "d": "\033[91m",
    "e": "\033[31m",
}

MODEL_PATH = Path(__file__).parent / "models" / "nutriscore_xgb.pkl"
GRADES     = ["a", "b", "c", "d", "e"]

# Noms lisibles pour l'affichage des features
NOMS_FEATURES = {
    "energy-kcal_100g":  "Énergie (kcal)",
    "fat_100g":           "Lipides",
    "saturated-fat_100g": "Graisses saturées",
    "sugars_100g":        "Sucres",
    "fiber_100g":         "Fibres",
    "proteins_100g":      "Protéines",
    "salt_100g":          "Sel",
}


def sep(char="─", n=70, color=GRAY):
    print(color + char * n + RESET)


def titre(texte):
    print()
    print(BOLD + CYAN + f"  {texte}" + RESET)
    sep()


def gc(g):
    return GRADE_ANSI.get(str(g).lower(), "")


def charger_modele():
    if not MODEL_PATH.exists():
        print(f"\033[31mModèle introuvable → {MODEL_PATH}\033[0m")
        print("Lancer d'abord : python src/train_model.py")
        sys.exit(1)
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["label_encoder"], payload["features"]


def charger_donnees(features, limit=None):
    print(f"\n{DIM}Connexion MongoDB...{RESET}", end="", flush=True)
    col = get_collection("products")

    projection = {f: 1 for f in features}
    projection["nutriscore_grade"] = 1
    projection["energy_100g"]      = 1
    projection["_id"]              = 0

    cursor = col.find({"nutriscore_grade": {"$in": GRADES}}, projection)
    if limit:
        cursor = cursor.limit(limit)

    docs = list(cursor)
    print(f"\r{DIM}Chargé : {len(docs):,} produits.{' ' * 20}{RESET}")

    df = pd.DataFrame(docs)
    for f in features + ["energy_100g"]:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    for f in features:
        if f in df.columns:
            df.loc[df[f] < 0,   f] = np.nan
            df.loc[df[f] > 100, f] = np.nan

    # Convertir kJ en kcal si le champ kcal est absent
    if "energy_100g" in df.columns and "energy-kcal_100g" in df.columns:
        masque = df["energy-kcal_100g"].isna() & df["energy_100g"].notna()
        df.loc[masque, "energy-kcal_100g"] = df.loc[masque, "energy_100g"] / 4.184
        df.drop(columns=["energy_100g"], inplace=True)

    return df


def afficher_distribution(df):
    titre("DISTRIBUTION DU JEU DE DONNÉES")
    counts = df["nutriscore_grade"].value_counts()
    total  = len(df)
    max_c  = counts.max()
    BAR    = 30

    print(f"  {'Grade':<8} {'Produits':>10}  {'%':>6}   Distribution")
    sep("·", 70)

    for g in GRADES:
        cnt   = counts.get(g, 0)
        pct   = cnt / total * 100 if total > 0 else 0
        n_bar = int(cnt / max_c * BAR) if max_c > 0 else 0
        bar   = "█" * n_bar + "░" * (BAR - n_bar)
        print(f"  {gc(g)}{BOLD}{g.upper():<8}{RESET}"
              f" {cnt:>10,}"
              f"  {pct:>5.1f}%"
              f"   {gc(g)}{bar}{RESET}")

    sep("·", 70)
    print(f"  {'TOTAL':<8} {total:>10,}  100.0%")


def afficher_accuracy(y_test, y_pred, n_test):
    titre("ACCURACY GLOBALE")
    acc    = accuracy_score(y_test, y_pred)
    pct    = acc * 100
    n_bars = int(pct / 100 * 50)
    bar    = "█" * n_bars + "░" * (50 - n_bars)

    # Vert si >80%, jaune entre 60-80%, rouge sinon
    if pct >= 80:
        color = "\033[1;32m"
    elif pct >= 60:
        color = "\033[33m"
    else:
        color = "\033[31m"

    print(f"  {color}{BOLD}{pct:.2f}%{RESET}   sur {n_test:,} exemples de test (split 80/20, seed=42)")
    print(f"  {color}{bar}{RESET}")


def afficher_confusion(y_test, y_pred):
    titre("MATRICE DE CONFUSION")
    cm       = confusion_matrix(y_test, y_pred)
    grades_u = [g.upper() for g in GRADES]
    max_val  = cm.max()

    print(f"  {GRAY}                    ← Prédit{RESET}")

    hdr = "  " + " " * 14
    for g in grades_u:
        hdr += f"  {BOLD}{gc(g.lower())}{g:>6}{RESET}"
    print(hdr)
    sep("·", 70)

    for i, g_reel in enumerate(grades_u):
        line = f"  {GRAY}Réel{RESET} {gc(g_reel.lower())}{BOLD}{g_reel:<6}{RESET}  {GRAY}│{RESET}  "
        for j in range(len(grades_u)):
            val  = cm[i][j]
            frac = val / max_val if max_val > 0 else 0
            if i == j:
                cell_c = "\033[1;32m"   # bien classé
            elif frac > 0.25:
                cell_c = "\033[1;31m"   # grosse erreur
            elif frac > 0.05:
                cell_c = "\033[33m"     # erreur modérée
            else:
                cell_c = GRAY
            line += f"{cell_c}{val:>6,}{RESET}  "

        rappel = cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0
        line  += f"  {GRAY}│ rappel {rappel:.1%}{RESET}"
        print(line)

    sep("·", 70)

    prec_line = "  " + " " * 14 + "   "
    for j in range(len(GRADES)):
        col_sum = cm[:, j].sum()
        p = cm[j][j] / col_sum if col_sum > 0 else 0
        prec_line += f"{GRAY}{p:>5.1%}   {RESET}"
    prec_line += f"  {GRAY}← précision{RESET}"
    print(prec_line)


def afficher_rapport(y_test, y_pred):
    titre("RAPPORT DE CLASSIFICATION")
    grades_u = [g.upper() for g in GRADES]
    report   = classification_report(
        y_test, y_pred,
        target_names=grades_u,
        output_dict=True,
        zero_division=0,
    )

    W = 10
    print(f"  {'Grade':<10} {'Précision':>{W}} {'Rappel':>{W}} {'F1-Score':>{W}} {'Support':>{W}}")
    sep("·", 70)

    for g in grades_u:
        row  = report.get(g, {})
        prec = row.get("precision", 0)
        rec  = row.get("recall",    0)
        f1   = row.get("f1-score",  0)
        sup  = int(row.get("support", 0))

        f1_c = "\033[32m" if f1 >= 0.80 else "\033[33m" if f1 >= 0.60 else "\033[31m"

        print(f"  {gc(g.lower())}{BOLD}{g:<10}{RESET}"
              f" {prec:>{W}.3f}"
              f" {rec:>{W}.3f}"
              f" {f1_c}{f1:>{W}.3f}{RESET}"
              f" {sup:>{W},}")

    sep("·", 70)
    macro = report.get("macro avg", {})
    wa    = report.get("weighted avg", {})

    print(f"  {'Macro moy.':<10}"
          f" {macro.get('precision',0):>{W}.3f}"
          f" {macro.get('recall',0):>{W}.3f}"
          f" {macro.get('f1-score',0):>{W}.3f}"
          f" {int(macro.get('support',0)):>{W},}")
    print(f"  {'Pondérée':<10}"
          f" {wa.get('precision',0):>{W}.3f}"
          f" {wa.get('recall',0):>{W}.3f}"
          f" {wa.get('f1-score',0):>{W}.3f}"
          f" {int(wa.get('support',0)):>{W},}")


def afficher_importance(model, features):
    titre("IMPORTANCE DES FEATURES  (gain XGBoost)")
    imps  = model.feature_importances_
    ordre = np.argsort(imps)[::-1]
    max_i = imps.max()
    BAR   = 36

    for rank, idx in enumerate(ordre):
        feat  = features[idx]
        imp   = imps[idx]
        nom   = NOMS_FEATURES.get(feat, feat)
        n_bar = int(imp / max_i * BAR) if max_i > 0 else 0
        bar   = "█" * n_bar + "░" * (BAR - n_bar)
        pct   = imp * 100

        if rank == 0:
            c = "\033[1;32m"
        elif rank <= 2:
            c = "\033[32m"
        elif rank <= 4:
            c = "\033[33m"
        else:
            c = "\033[90m"

        medal = ["🥇", "🥈", "🥉", "  ", "  ", "  ", "  "][rank]
        print(f"  {medal} {nom:<22} {c}{bar}{RESET}  {BOLD}{pct:5.1f}%{RESET}")


def afficher_couverture(df, features):
    titre("COUVERTURE DES DONNÉES (valeurs non-nulles)")
    total = len(df)

    print(f"  {'Feature':<26} {'Renseigné':>12}  {'%':>6}   Couverture")
    sep("·", 70)

    for f in features:
        nom   = NOMS_FEATURES.get(f, f)
        n_ok  = df[f].notna().sum()
        pct   = n_ok / total * 100 if total > 0 else 0
        n_bar = int(pct / 100 * 30)
        bar   = "█" * n_bar + "░" * (30 - n_bar)
        color = "\033[32m" if pct >= 80 else "\033[33m" if pct >= 50 else "\033[31m"
        print(f"  {nom:<26} {n_ok:>12,}  {pct:>5.1f}%   {color}{bar}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Evaluation XGBoost Nutri-Score")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true",
                       help="Jeu complet (~447k produits, lent)")
    group.add_argument("-n", type=int, default=50_000, metavar="N",
                       help="Taille de l'échantillon (défaut : 50 000)")
    args  = parser.parse_args()
    limit = None if args.all else args.n

    print()
    print(BOLD + CYAN + "╔══════════════════════════════════════════════════════════════════════╗" + RESET)
    print(BOLD + CYAN + "║      ÉVALUATION MODÈLE · XGBoost Nutri-Score (A / B / C / D / E)   ║" + RESET)
    print(BOLD + CYAN + "║                       FoodLens — IPSSI M2 IA & Big Data              ║" + RESET)
    print(BOLD + CYAN + "╚══════════════════════════════════════════════════════════════════════╝" + RESET)

    print(f"\n{DIM}Chargement : {MODEL_PATH}{RESET}")
    model, le, features = charger_modele()
    print(f"{DIM}Features  : {', '.join(features)}{RESET}")
    print(f"{DIM}Mode      : {'jeu complet' if args.all else f'échantillon {limit:,}'}{RESET}")

    df = charger_donnees(features, limit=limit)
    if df.empty:
        print("\033[31mAucun produit labellisé trouvé dans MongoDB.\033[0m")
        sys.exit(1)

    afficher_distribution(df)
    afficher_couverture(df, features)

    # On refait le même split qu'à l'entraînement pour évaluer sur des données inédites
    y_all = le.transform(df["nutriscore_grade"])
    X_all = df[features].values
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    print(f"\n{DIM}Prédiction sur {len(X_test):,} exemples...{RESET}", end="", flush=True)
    y_pred = model.predict(X_test)
    print(f"\r{DIM}Prédictions terminées.{' ' * 30}{RESET}")

    afficher_accuracy(y_test, y_pred, len(X_test))
    afficher_confusion(y_test, y_pred)
    afficher_rapport(y_test, y_pred)
    afficher_importance(model, features)

    acc = accuracy_score(y_test, y_pred)
    print()
    sep("═", 70)
    print(f"  {GRAY}Accuracy : {acc*100:.2f}%  |  Test : {len(X_test):,} ex.  |  Modèle : {MODEL_PATH.name}{RESET}")
    sep("═", 70)
    print()


if __name__ == "__main__":
    main()
