# -*- coding: utf-8 -*-
"""
Client Open Food Facts API.
Utilisé en fallback quand un produit n'est pas dans notre base MongoDB.
"""
import requests

OFF_API_URL    = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"

# On ne demande à l'API que les champs dont on a besoin — ça accélère la réponse
CHAMPS_UTILES = [
    "code", "product_name", "brands", "categories_tags",
    "pnns_groups_1", "pnns_groups_2",
    "nutriscore_grade", "nutriscore_score",
    "nova_group", "additives_n", "additives_tags",
    "nutriments", "allergens_tags", "allergens",
    "image_front_small_url", "image_url",
    "ingredients_text_fr", "ingredients_text",
    "ecoscore_grade", "ecoscore_score",
    "labels_tags", "labels",
    "ingredients_analysis_tags",
    "nutrient_levels",
    "serving_size",
    "packaging_tags", "packaging",
    "origins",
    "manufacturing_places",
]

# Correspondance tag label → (emoji, texte) pour l'affichage
LABELS_DISPLAY = {
    "en:organic":                   ("🌱", "Bio"),
    "en:eu-organic":                ("🌍", "Bio EU"),
    "fr:ab-agriculture-biologique": ("🌱", "AB Bio"),
    "en:fair-trade":                ("🤝", "Commerce équitable"),
    "en:no-palm-oil":               ("🌴", "Sans huile de palme"),
    "en:rainforest-alliance":       ("🦋", "Rainforest Alliance"),
    "en:vegan":                     ("🌿", "Vegan"),
    "en:vegetarian":                ("🥗", "Végétarien"),
    "en:gluten-free":               ("🌾", "Sans gluten"),
    "en:halal":                     ("☪️", "Halal"),
    "en:kosher":                    ("✡️", "Casher"),
    "en:non-gmo":                   ("🧬", "Sans OGM"),
    "en:low-fat":                   ("⚖️", "Allégé"),
    "en:no-artificial-flavors":     ("✨", "Sans arômes artif."),
    "en:no-preservatives":          ("🚫", "Sans conservateurs"),
}

# Correspondance tag analyse ingrédients → (emoji, texte, bg, couleur texte)
ANALYSIS_DISPLAY = {
    "en:vegan":                     ("🌿", "Vegan",                    "#D1FAE5", "#059669"),
    "en:non-vegan":                 ("🌿", "Non vegan",                "#FEE2E2", "#DC2626"),
    "en:vegan-status-unknown":      ("🌿", "Vegan ?",                  "#F3F4F6", "#9CA3AF"),
    "en:vegetarian":                ("🥗", "Végétarien",               "#D1FAE5", "#059669"),
    "en:non-vegetarian":            ("🥗", "Non végé",                 "#FEE2E2", "#DC2626"),
    "en:vegetarian-status-unknown": ("🥗", "Végé ?",                   "#F3F4F6", "#9CA3AF"),
    "en:palm-oil-free":             ("🌴", "Sans huile de palme",      "#D1FAE5", "#059669"),
    "en:palm-oil":                  ("🌴", "Contient huile de palme",  "#FEE2E2", "#DC2626"),
    "en:may-contain-palm-oil":      ("🌴", "Peut contenir huile de palme", "#FEF3C7", "#D97706"),
}


def _extraire_nutriments(nutriments: dict) -> dict:
    """
    Remapping des nutriments depuis la structure imbriquée de l'API OFF
    vers les noms de champs plats qu'on utilise dans l'app.
    """
    mapping = {
        "energy-kcal_100g":   "energy-kcal_100g",
        "fat_100g":            "fat_100g",
        "saturated-fat_100g":  "saturated-fat_100g",
        "sugars_100g":         "sugars_100g",
        "fiber_100g":          "fiber_100g",
        "proteins_100g":       "proteins_100g",
        "salt_100g":           "salt_100g",
    }
    return {cle: nutriments.get(src) for cle, src in mapping.items()}


def chercher_par_code(code: str) -> dict | None:
    """
    Récupère un produit par code-barres depuis l'API Open Food Facts.
    Timeout à 5s pour ne pas bloquer l'affichage si l'API est lente.
    Retourne None si le produit n'existe pas ou si la requête échoue.
    """
    try:
        response = requests.get(
            OFF_API_URL.format(code=code),
            params={"fields": ",".join(CHAMPS_UTILES)},
            timeout=5,
        )
        data = response.json()
        if data.get("status") != 1:
            return None

        p             = data["product"]
        nutriments    = _extraire_nutriments(p.get("nutriments", {}))
        additives_raw = p.get("additives_tags", [])
        labels_raw    = p.get("labels_tags", [])
        analysis_raw  = p.get("ingredients_analysis_tags", [])

        return {
            "code":                      code,
            "product_name":              p.get("product_name"),
            "brands":                    p.get("brands"),
            "pnns_groups_1":             p.get("pnns_groups_1"),
            "pnns_groups_2":             p.get("pnns_groups_2"),
            "nutriscore_grade":          p.get("nutriscore_grade"),
            "nutriscore_score":          p.get("nutriscore_score"),
            "nova_group":                p.get("nova_group"),
            "additives_n":               p.get("additives_n"),
            "additives_tags":            additives_raw,
            "allergens":                 ", ".join(p.get("allergens_tags", [])),
            "image_url":                 p.get("image_front_small_url") or p.get("image_url"),
            "ingredients":               p.get("ingredients_text_fr") or p.get("ingredients_text"),
            "ecoscore_grade":            p.get("ecoscore_grade"),
            "ecoscore_score":            p.get("ecoscore_score"),
            "labels_tags":               labels_raw,
            "ingredients_analysis_tags": analysis_raw,
            "nutrient_levels":           p.get("nutrient_levels", {}),
            "serving_size":              p.get("serving_size"),
            "packaging":                 p.get("packaging") or ", ".join(p.get("packaging_tags", [])),
            "origins":                   p.get("origins"),
            "manufacturing_places":      p.get("manufacturing_places"),
            **nutriments,
            "source": "api",  # permet au template de savoir que c'est un résultat live
        }
    except Exception:
        return None


def chercher_par_nom(nom: str, page: int = 1, taille: int = 20) -> list[dict]:
    """
    Recherche des produits par nom via l'API Open Food Facts.
    Timeout à 8s (la recherche est plus lente que le lookup par code).
    """
    try:
        response = requests.get(
            OFF_SEARCH_URL,
            params={
                "search_terms": nom,
                "search_simple": 1,
                "action":        "process",
                "json":          1,
                "page":          page,
                "page_size":     taille,
                "fields":        ",".join(CHAMPS_UTILES),
            },
            timeout=8,
        )
        data     = response.json()
        produits = []
        for p in data.get("products", []):
            nutriments = _extraire_nutriments(p.get("nutriments", {}))
            produits.append({
                "code":                      p.get("code"),
                "product_name":              p.get("product_name"),
                "brands":                    p.get("brands"),
                "pnns_groups_2":             p.get("pnns_groups_2"),
                "nutriscore_grade":          p.get("nutriscore_grade"),
                "nova_group":                p.get("nova_group"),
                "additives_n":               p.get("additives_n"),
                "image_url":                 p.get("image_front_small_url") or p.get("image_url"),
                "ecoscore_grade":            p.get("ecoscore_grade"),
                "labels_tags":               p.get("labels_tags", []),
                "ingredients_analysis_tags": p.get("ingredients_analysis_tags", []),
                **nutriments,
                "source": "api",
            })
        return produits
    except Exception:
        return []
