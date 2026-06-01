# -*- coding: utf-8 -*-
"""
Algorithme de substitution produit.
Pour un produit donné, on cherche dans MongoDB des alternatives
avec un meilleur Nutri-Score, dans la même catégorie alimentaire.
"""
import re
from src.database import get_collection

# Ordre des grades — on s'en sert pour comparer et scorer
ORDRE_GRADES = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

# Mots à ignorer pour l'extraction de mots-clés du nom produit
_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "avec", "sans", "et", "à",
    "au", "aux", "en", "un", "une", "bio", "pour", "par", "ou",
    "the", "and", "with", "100", "g", "ml", "cl", "kg", "gr",
}


def _meilleur_grade(grade_actuel: str) -> list[str]:
    """Retourne la liste des grades strictement meilleurs que le grade actuel."""
    rang = ORDRE_GRADES.get(grade_actuel, 4)
    return [g for g, r in ORDRE_GRADES.items() if r < rang]


def _keywords(nom: str) -> list[str]:
    """Extrait les mots significatifs du nom pour affiner la recherche."""
    mots = re.sub(r"[^\w\s]", " ", nom.lower()).split()
    return [m for m in mots if len(m) > 3 and m not in _STOPWORDS][:5]


def _score(original: dict, candidat: dict) -> float:
    """
    Score de tri — plus c'est bas, mieux c'est.
    On privilégie dans l'ordre : le grade, le score numérique,
    le nombre d'additifs, et la proximité calorique.
    """
    score = 0.0

    # Le grade reste le critère principal
    g = str(candidat.get("nutriscore_grade") or "e").lower()
    score += ORDRE_GRADES.get(g, 4) * 10

    # Le score numérique sert à départager les produits du même grade
    ns = candidat.get("nutriscore_score")
    if ns is not None:
        score += float(ns) * 0.5

    # On favorise les produits avec peu d'additifs
    score += int(candidat.get("additives_n") or 0) * 0.5

    # Pénalise si les calories sont très différentes de l'original
    orig_kcal = float(original.get("energy-kcal_100g") or 0)
    cand_kcal = float(candidat.get("energy-kcal_100g") or 0)
    if orig_kcal and cand_kcal:
        score += abs(cand_kcal - orig_kcal) / 200

    return score


def trouver_substituts(produit: dict, limit: int = 3) -> list[dict]:
    """
    Retourne jusqu'à `limit` produits de substitution triés par pertinence.

    La recherche se fait en 3 passes successives :
    1. Même catégorie + mots-clés du nom (résultats les plus proches)
    2. Même catégorie seule (si pas assez de résultats en passe 1)
    3. Catégorie parente pnns_groups_1 (filet de sécurité)
    """
    categorie = produit.get("pnns_groups_2")
    grade     = str(produit.get("nutriscore_grade") or "e").lower()
    code      = produit.get("code")
    nom       = produit.get("product_name") or ""

    grades_cibles = _meilleur_grade(grade)
    if not grades_cibles:
        return []  # produit déjà en grade A, rien à améliorer

    col    = get_collection("products")
    base_q = {
        "nutriscore_grade": {"$in": grades_cibles},
        "code":             {"$ne": code},
        "product_name":     {"$exists": True, "$ne": None},
    }

    resultats = []

    # Passe 1 : même catégorie + mots-clés du nom
    keywords = _keywords(nom)
    if categorie and keywords:
        kw_regex = "|".join(re.escape(k) for k in keywords)
        q = {**base_q, "pnns_groups_2": categorie,
             "product_name": {"$regex": kw_regex, "$options": "i"}}
        resultats = [{k: v for k, v in p.items() if k != "_id"}
                     for p in col.find(q, limit=limit * 4)]

    # Passe 2 : même catégorie sans filtre sur le nom
    if len(resultats) < limit and categorie:
        codes_deja = {r.get("code") for r in resultats} | {code}
        q2 = {**base_q, "pnns_groups_2": categorie,
              "code": {"$nin": list(codes_deja)}}
        resultats += [{k: v for k, v in p.items() if k != "_id"}
                      for p in col.find(q2, limit=limit * 4)]

    # Passe 3 : catégorie parente (pnns_groups_1) si toujours insuffisant
    if len(resultats) < limit:
        cat1 = produit.get("pnns_groups_1")
        if cat1:
            codes_deja = {r.get("code") for r in resultats} | {code}
            q3 = {**base_q, "pnns_groups_1": cat1,
                  "code": {"$nin": list(codes_deja)}}
            resultats += [{k: v for k, v in p.items() if k != "_id"}
                          for p in col.find(q3, limit=limit * 4)]

    # Dédoublonnage et tri par score
    seen   = set()
    unique = []
    for r in resultats:
        c = r.get("code")
        if c and c not in seen:
            seen.add(c)
            unique.append(r)

    unique.sort(key=lambda r: _score(produit, r))
    return unique[:limit]


def trouver_substitut(produit: dict) -> dict | None:
    subs = trouver_substituts(produit, limit=1)
    return subs[0] if subs else None
