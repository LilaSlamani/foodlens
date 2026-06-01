# -*- coding: utf-8 -*-
"""
FoodLens — backend FastAPI avec rendu Jinja2.
Toutes les routes, la logique métier et la gestion des sessions sont ici.
"""
import os, sys, json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, Response, Form, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv
import bcrypt

from src.database import get_collection
from src.off_api import chercher_par_nom, chercher_par_code, LABELS_DISPLAY, ANALYSIS_DISPLAY
from src.substitution import trouver_substituts
from src.ml_predictor import predire_grade, modele_disponible

load_dotenv()

BASE   = Path(__file__).parent
SECRET = os.getenv("SECRET_KEY", "foodlens-dev-secret")

# Le serializer signe les cookies avec la clé secrète — si le cookie est
# modifié côté client, la vérification échoue et la session est rejetée
_ser = URLSafeTimedSerializer(SECRET)

app = FastAPI(title="FoodLens")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))


# ── Gestion des sessions ───────────────────────────────────────────────────────

def get_user(request: Request) -> dict | None:
    """Lit et vérifie le cookie de session — retourne None si absent ou invalide."""
    cookie = request.cookies.get("fl_session")
    if not cookie:
        return None
    try:
        # max_age=7 jours : au-delà, le cookie est expiré même si la signature est valide
        return _ser.loads(cookie, max_age=86400 * 7)
    except Exception:
        return None


def is_admin(user: dict | None) -> bool:
    return bool(user and user.get("role") == "admin")


def login_user(response: Response, user_doc: dict):
    """Crée le cookie de session signé avec les infos de l'utilisateur."""
    data = {
        "id":        str(user_doc["_id"]),
        "username":  user_doc["username"],
        "email":     user_doc["email"],
        "allergenes": user_doc.get("allergenes", []),
        "role":      user_doc.get("role", "user"),
    }
    # httponly : JavaScript ne peut pas lire ce cookie (protection XSS)
    # samesite=lax : le cookie n'est pas envoyé sur les requêtes cross-site (protection CSRF)
    response.set_cookie("fl_session", _ser.dumps(data),
                        httponly=True, max_age=86400 * 7, samesite="lax")


def logout_user(response: Response):
    response.delete_cookie("fl_session")


# ── Couleurs et labels ─────────────────────────────────────────────────────────

# Palette officielle Nutri-Score (fonds et couleurs de texte)
GRADE_COLORS = {
    "a": {"bg": "#1a9641", "text": "#fff"},
    "b": {"bg": "#a6d96a", "text": "#1A1F36"},
    "c": {"bg": "#ffffbf", "text": "#1A1F36"},
    "d": {"bg": "#fdae61", "text": "#1A1F36"},
    "e": {"bg": "#d7191c", "text": "#fff"},
}

ECO_COLORS = {
    "a": {"bg": "#1a9641", "text": "#fff"},
    "b": {"bg": "#a6d96a", "text": "#1A1F36"},
    "c": {"bg": "#f0c419", "text": "#1A1F36"},
    "d": {"bg": "#f07819", "text": "#fff"},
    "e": {"bg": "#d7191c", "text": "#fff"},
}

NOVA_LABELS = {
    1: "Non transformé",
    2: "Ingrédient culinaire",
    3: "Transformé",
    4: "Ultra-transformé",
}

# Correspondance tags Open Food Facts → noms français des allergènes
# (les tags arrivent au format "en:milk", "fr:lait", etc.)
ALLERGEN_NAMES = {
    "en:milk":            "Lait",
    "en:eggs":            "Œufs",
    "en:gluten":          "Gluten",
    "en:wheat":           "Blé",
    "en:peanuts":         "Cacahuètes",
    "en:nuts":            "Fruits à coque",
    "en:almonds":         "Amandes",
    "en:hazelnuts":       "Noisettes",
    "en:walnuts":         "Noix",
    "en:cashews":         "Noix de cajou",
    "en:pistachios":      "Pistaches",
    "en:soybeans":        "Soja",
    "en:fish":            "Poisson",
    "en:shellfish":       "Crustacés",
    "en:molluscs":        "Mollusques",
    "en:celery":          "Céleri",
    "en:mustard":         "Moutarde",
    "en:sesame-seeds":    "Sésame",
    "en:sesame":          "Sésame",
    "en:sulphur-dioxide": "Sulfites",
    "en:sulphites":       "Sulfites",
    "en:lupin":           "Lupin",
    "fr:lait":            "Lait",
    "fr:gluten":          "Gluten",
    "fr:oeufs":           "Œufs",
}


def _format_allergens(raw: str) -> list[str]:
    """
    Convertit la chaîne brute d'allergènes ("en:milk,en:gluten") en liste lisible.
    Pour les tags inconnus, on nettoie le préfixe et on capitalise.
    """
    if not raw:
        return []
    tags   = [t.strip() for t in raw.replace(";", ",").split(",")]
    result = []
    for tag in tags:
        key  = tag.lower()
        name = ALLERGEN_NAMES.get(key)
        if not name:
            # Tag non référencé : on retire le préfixe langue et on formate
            name = key.replace("en:", "").replace("fr:", "").replace("-", " ").strip()
            name = name.capitalize() if name else None
        if name:
            result.append(name)
    return result


def _fmt_additive(tag: str) -> str:
    """Formate un tag additif "en:e150d-caramel" en "E150D — Caramel"."""
    clean = tag.replace("en:", "").replace("fr:", "")
    parts = clean.split("-", 1)
    e_num = parts[0].upper()
    name  = parts[1].replace("-", " ").title() if len(parts) > 1 else ""
    return f"{e_num}{' — ' + name if name else ''}"


def _off_image_url(code: str) -> str:
    """
    Construit l'URL de l'image produit depuis le CDN Open Food Facts.
    Le chemin est dérivé du code-barres : les 13 chiffres sont découpés en 3+3+3+reste.
    Ex: 3017620422003 → 301/762/042/2003/front_fr.3.400.jpg
    """
    c = str(code).strip()
    if len(c) >= 9:
        p = f"{c[0:3]}/{c[3:6]}/{c[6:9]}/{c[9:]}"
    else:
        p = c
    return f"https://images.openfoodfacts.org/images/products/{p}/front_fr.3.400.jpg"


def _enrich(p: dict) -> dict:
    """
    Enrichit un document produit avec les champs calculés dont les templates ont besoin.
    On préfixe ces champs par "_" pour les distinguer des données brutes MongoDB.
    """
    grade = str(p.get("nutriscore_grade") or "").lower()

    # Si le produit n'a pas de grade officiel, on tente une prédiction ML
    if grade in ("", "none", "?") and modele_disponible():
        predicted = predire_grade(p)
        if predicted:
            p["nutriscore_grade"] = predicted
            p["grade_predicted"]  = True  # le template affichera un badge "Prédit par IA"
            grade = predicted

    p["_grade_color"] = GRADE_COLORS.get(grade, {"bg": "#E5E7EB", "text": "#6B7280"})

    eco = str(p.get("ecoscore_grade") or "").lower()
    p["_eco_color"] = ECO_COLORS.get(eco, {"bg": "#E5E7EB", "text": "#6B7280"})

    # nova_group est stocké en float dans MongoDB (4.0, pas 4) — on convertit
    nova = p.get("nova_group")
    p["_nova_label"] = NOVA_LABELS.get(int(nova), "Inconnu") if nova else None

    # Analyse ingrédients (vegan, végétarien, huile de palme…)
    # On déduplique par type de propriété pour éviter "Vegan" et "Non-vegan" ensemble
    analysis = p.get("ingredients_analysis_tags", []) or []
    chips    = []
    seen     = set()
    for tag in analysis:
        if tag in ANALYSIS_DISPLAY:
            base = tag.replace("en:", "").replace("non-", "").replace("-status-unknown", "")
            if base not in seen:
                seen.add(base)
                chips.append(ANALYSIS_DISPLAY[tag])
    p["_analysis_chips"] = chips

    # Labels certifications (bio, commerce équitable…) — max 5 pour ne pas surcharger l'UI
    labels    = p.get("labels_tags", []) or []
    p["_labels"] = [LABELS_DISPLAY[t] for t in labels if t in LABELS_DISPLAY][:5]

    # Additifs : on formate chaque tag en "E150D — Caramel"
    add_tags          = p.get("additives_tags", []) or []
    p["_additives_fmt"] = [_fmt_additive(t) for t in add_tags if ":" in t]

    p["_allergens_list"] = _format_allergens(p.get("allergens") or "")

    # Aucun produit de notre base MongoDB n'a d'image_url stockée —
    # on construit l'URL CDN à la volée depuis le code-barres
    if not p.get("image_url") and p.get("code"):
        p["image_url"] = _off_image_url(str(p["code"]))

    return p


def _load_product(code: str) -> dict | None:
    """
    Charge un produit depuis MongoDB. Si absent, on tente l'API Open Food Facts.
    Retourne None si le produit n'existe nulle part.
    """
    col = get_collection("products")
    p   = col.find_one({"code": code})
    if p:
        p.pop("_id", None)
    else:
        p = chercher_par_code(code)  # fallback API OFF
    if p:
        _enrich(p)
    return p


# Valeurs de référence pour les barres nutritionnelles (en g/100g)
# max_v sert à calculer le pourcentage de remplissage de la barre
NUTRITION_CFG = [
    ("Graisses",      "fat_100g",          100, "#E07B54", "fat"),
    ("dont saturées", "saturated-fat_100g", 100, "#F4A261", "saturated-fat"),
    ("Sucres",        "sugars_100g",        100, "#EAB308", "sugars"),
    ("Fibres",        "fiber_100g",          15, "#3A7D44", ""),
    ("Protéines",     "proteins_100g",       40, "#4361EE", ""),
    ("Sel",           "salt_100g",            6, "#9CA3AF", "salt"),
]


def _nutrition_rows(p: dict) -> list:
    """Prépare les données des barres nutritionnelles pour le template."""
    levels = p.get("nutrient_levels", {}) or {}
    rows   = []
    for label, key, max_v, color, lvl_key in NUTRITION_CFG:
        val = p.get(key)
        if val is None:
            continue
        pct   = min(100, max(0, float(val) / max_v * 100))
        level = levels.get(lvl_key, "")  # "low", "moderate", "high" ou vide
        rows.append({"label": label, "val": round(float(val), 1),
                     "pct": pct, "color": color, "level": level})
    return rows


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = get_user(request)
    try:
        col = get_collection("products")
        # On affiche 8 produits de bonne qualité triés par score croissant
        # (score bas = meilleur Nutri-Score)
        populaires = [
            {k: v for k, v in p.items() if k != "_id"}
            for p in col.find(
                {"nutriscore_grade": {"$in": ["a", "b"]},
                 "product_name":     {"$exists": True, "$ne": ""}},
                limit=8, sort=[("nutriscore_score", 1)]
            )
        ]
        for p in populaires:
            _enrich(p)
    except Exception:
        populaires = []

    try:
        total  = get_collection("products").estimated_document_count()
        nb_fmt = f"{total:,}".replace(",", " ")
    except Exception:
        nb_fmt = "—"

    return templates.TemplateResponse(request, "index.html", {
        "user":         user,
        "populaires":   populaires,
        "total":        nb_fmt,
        "grade_colors": GRADE_COLORS,
    })


@app.get("/recherche", response_class=HTMLResponse)
async def recherche(
    request:  Request,
    q:        str       = Query(default=""),
    grade:    list[str] = Query(default=[]),
    nova:     list[int] = Query(default=[]),
    kcal_max: int       = Query(default=0),
):
    user      = get_user(request)
    resultats = []

    if q and len(q) >= 2:
        try:
            col   = get_collection("products")
            query: dict = {"product_name": {"$regex": q, "$options": "i"}}

            if grade:
                query["nutriscore_grade"] = {"$in": [g.lower() for g in grade]}
            if nova:
                query["nova_group"] = {"$in": nova}
            if kcal_max and kcal_max < 1000:
                query["energy-kcal_100g"] = {"$lte": float(kcal_max)}

            mongo_res = [{k: v for k, v in p.items() if k != "_id"}
                         for p in col.find(query, limit=30)]

            # Si MongoDB renvoie peu de résultats, on complète avec l'API OFF
            # (utile pour des produits récents pas encore dans notre base)
            if len(mongo_res) < 5:
                api_res   = chercher_par_nom(q, taille=20)
                codes_ok  = {r.get("code") for r in mongo_res}
                mongo_res += [r for r in api_res if r.get("code") not in codes_ok]

            resultats = [_enrich(p) for p in mongo_res]
        except Exception:
            pass

    return templates.TemplateResponse(request, "search.html", {
        "user":         user,
        "q":            q,
        "resultats":    resultats,
        "grade_filter": grade,
        "nova_filter":  nova,
        "kcal_max":     kcal_max,
        "grade_colors": GRADE_COLORS,
    })


@app.get("/produit/{code}", response_class=HTMLResponse)
async def produit(request: Request, code: str):
    user = get_user(request)
    p    = _load_product(code)
    if not p:
        return RedirectResponse("/recherche", status_code=302)
    return templates.TemplateResponse(request, "product.html", {
        "user":         user,
        "p":            p,
        "nutrition":    _nutrition_rows(p),
        "grade_colors": GRADE_COLORS,
        "eco_colors":   ECO_COLORS,
    })


@app.get("/substitution/{code}", response_class=HTMLResponse)
async def substitution(request: Request, code: str):
    user = get_user(request)
    p    = _load_product(code)
    if not p:
        return RedirectResponse("/recherche", status_code=302)

    subs = trouver_substituts(p, limit=3)
    for s in subs:
        _enrich(s)  # les substituts ont aussi besoin des couleurs et labels

    return templates.TemplateResponse(request, "substitution.html", {
        "user":         user,
        "original":     p,
        "substituts":   subs,
        "grade_colors": GRADE_COLORS,
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_user(request)
    try:
        col = get_collection("products")

        # Répartition par grade Nutri-Score
        grades_raw = {d["_id"]: d["count"] for d in col.aggregate([
            {"$group": {"_id": "$nutriscore_grade", "count": {"$sum": 1}}},
        ]) if d["_id"]}

        # nova_group est stocké en float (4.0) dans MongoDB — on convertit les clés en int
        nova_raw = {}
        for d in col.aggregate([{"$group": {"_id": "$nova_group", "count": {"$sum": 1}}}]):
            if d["_id"] is not None:
                try:
                    nova_raw[int(d["_id"])] = d["count"]
                except (TypeError, ValueError):
                    pass

        # Top 8 catégories alimentaires (pnns_groups_2) par nombre de produits
        cats_raw = [(d["_id"], d["count"]) for d in col.aggregate([
            {"$match": {"pnns_groups_2": {"$nin": [None, "", "unknown"]}}},
            {"$group": {"_id": "$pnns_groups_2", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}, {"$limit": 8},
        ])]

        # Distribution des additifs en 4 tranches : 0 / 1-3 / 4-9 / 10+
        # On utilise $cond pour faire le bucketing directement en agregation Mongo
        add_agg = list(col.aggregate([
            {"$match": {"additives_n": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": None,
                "avg":  {"$avg": "$additives_n"},
                "zero": {"$sum": {"$cond": [{"$eq": ["$additives_n", 0]}, 1, 0]}},
                "low":  {"$sum": {"$cond": [{"$and": [{"$gt": ["$additives_n", 0]},  {"$lte": ["$additives_n", 3]}]}, 1, 0]}},
                "med":  {"$sum": {"$cond": [{"$and": [{"$gt": ["$additives_n", 3]},  {"$lte": ["$additives_n", 9]}]}, 1, 0]}},
                "high": {"$sum": {"$cond": [{"$gt":  ["$additives_n", 9]}, 1, 0]}},
            }},
        ]))

        total      = col.estimated_document_count()
        total_nova = sum(nova_raw.values()) or 1
        avg_add    = round(add_agg[0]["avg"], 1) if add_agg else 0
        add_dist   = [add_agg[0]["zero"], add_agg[0]["low"],
                      add_agg[0]["med"],  add_agg[0]["high"]] if add_agg else [0] * 4

        stats = {
            "total":        f"{total:,}".replace(",", " "),
            "pct_ab":       round((grades_raw.get("a", 0) + grades_raw.get("b", 0)) / max(total, 1) * 100, 1),
            "pct_nova4":    round(nova_raw.get(4, 0) / total_nova * 100, 1),
            "avg_additifs": avg_add,
            "nutri_dist":   [grades_raw.get(g, 0) for g in ["a", "b", "c", "d", "e"]],
            "nova_dist":    [nova_raw.get(n, 0) for n in [1, 2, 3, 4]],
            "cat_labels":   [c[0][:22] for c in cats_raw],  # tronqué pour les graphiques
            "cat_values":   [c[1] for c in cats_raw],
            "add_dist":     add_dist,
        }
    except Exception:
        stats = {
            "total": "—", "pct_ab": 0, "pct_nova4": 0, "avg_additifs": 0,
            "nutri_dist": [0] * 6, "nova_dist": [0] * 4,
            "cat_labels": [], "cat_values": [], "add_dist": [0] * 4,
        }

    return templates.TemplateResponse(request, "dashboard.html", {
        "user":  user,
        "stats": stats,
    })


@app.get("/connexion", response_class=HTMLResponse)
async def connexion_page(request: Request):
    # Si l'utilisateur est déjà connecté, on le redirige directement
    if get_user(request):
        return RedirectResponse("/profil", status_code=302)
    return templates.TemplateResponse(request, "auth.html", {
        "user": None, "error": None, "success": None,
    })


@app.post("/connexion", response_class=HTMLResponse)
async def do_login(
    request:  Request,
    email:    str = Form(""),
    password: str = Form(""),
):
    user_doc = get_collection("users").find_one({"email": email.lower().strip()})
    # bcrypt.checkpw compare le mot de passe en clair avec le hash stocké
    if user_doc and bcrypt.checkpw(password.encode(), user_doc["password"]):
        r = RedirectResponse("/", status_code=302)
        login_user(r, user_doc)
        return r
    return templates.TemplateResponse(request, "auth.html", {
        "user": None, "error": "Email ou mot de passe incorrect.", "success": None,
    })


@app.post("/inscription", response_class=HTMLResponse)
async def do_register(
    request:    Request,
    username:   str       = Form(""),
    email:      str       = Form(""),
    password:   str       = Form(""),
    allergenes: list[str] = Form(default=[]),
):
    error = None
    if not username or not email or not password:
        error = "Tous les champs sont obligatoires."
    elif len(password) < 6:
        error = "Mot de passe trop court (6 caractères minimum)."
    elif get_collection("users").find_one({"email": email.lower().strip()}):
        error = "Cet email est déjà utilisé."

    if error:
        return templates.TemplateResponse(request, "auth.html", {
            "user": None, "error": error, "success": None,
        })

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    # Le tout premier compte créé devient automatiquement admin
    is_first = get_collection("users").estimated_document_count() == 0
    doc = {
        "username":  username.strip(),
        "email":     email.lower().strip(),
        "password":  hashed,
        "allergenes": allergenes,
        "role":      "admin" if is_first else "user",
        "created":   datetime.utcnow(),
    }
    result    = get_collection("users").insert_one(doc)
    doc["_id"] = result.inserted_id
    r = RedirectResponse("/", status_code=302)
    login_user(r, doc)
    return r


@app.post("/deconnexion")
async def do_logout():
    r = RedirectResponse("/", status_code=302)
    logout_user(r)
    return r


@app.get("/profil", response_class=HTMLResponse)
async def profil(request: Request):
    user = get_user(request)
    if not user:
        return RedirectResponse("/connexion", status_code=302)
    try:
        favs_raw = list(get_collection("favorites").find(
            {"user_id": user["id"]}, sort=[("date", -1)], limit=20
        ))
        favs = [{k: v for k, v in f.items() if k != "_id"} for f in favs_raw]
    except Exception:
        favs = []
    return templates.TemplateResponse(request, "profile.html", {
        "user": user, "favoris": favs,
    })


@app.post("/favoris/{code}")
async def add_favori(request: Request, code: str):
    user = get_user(request)
    if not user:
        return RedirectResponse("/connexion", status_code=302)
    p = _load_product(code)
    if p:
        # upsert=True : si le favori existe déjà, on le met à jour sans créer de doublon
        get_collection("favorites").update_one(
            {"user_id": user["id"], "code": code},
            {"$set": {
                "code":             code,
                "user_id":          user["id"],
                "date":             datetime.utcnow(),
                "product_name":     p.get("product_name"),
                "brands":           p.get("brands"),
                "nutriscore_grade": p.get("nutriscore_grade"),
                "image_url":        p.get("image_url"),
            }},
            upsert=True,
        )
    return RedirectResponse(f"/produit/{code}?saved=1", status_code=302)


@app.post("/profil/allergenes")
async def update_allergenes(
    request:    Request,
    allergenes: list[str] = Form(default=[]),
):
    user = get_user(request)
    if not user:
        return RedirectResponse("/connexion", status_code=302)
    from bson import ObjectId
    get_collection("users").update_one(
        {"_id": ObjectId(user["id"])},
        {"$set": {"allergenes": allergenes}},
    )
    # On met aussi à jour le cookie pour que la session reflète le changement immédiatement
    user["allergenes"] = allergenes
    r = RedirectResponse("/profil", status_code=302)
    r.set_cookie("fl_session", _ser.dumps(user), httponly=True, max_age=86400 * 7, samesite="lax")
    return r


# ── Panel d'administration ─────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    user = get_user(request)
    if not is_admin(user):
        return RedirectResponse("/", status_code=302)

    try:
        col_u = get_collection("users")
        col_p = get_collection("products")
        col_f = get_collection("favorites")

        users_raw  = list(col_u.find({}, sort=[("created", -1)], limit=200))
        users_list = [
            {
                "id":        str(u["_id"]),
                "username":  u.get("username"),
                "email":     u.get("email"),
                "role":      u.get("role", "user"),
                "created":   u.get("created"),
                "allergenes": u.get("allergenes", []),
            }
            for u in users_raw
        ]

        total_products  = col_p.estimated_document_count()
        total_users     = col_u.estimated_document_count()
        total_favorites = col_f.estimated_document_count()

        grades = {d["_id"]: d["count"] for d in col_p.aggregate([
            {"$group": {"_id": "$nutriscore_grade", "count": {"$sum": 1}}}
        ]) if d["_id"]}

        # Top 5 produits les plus ajoutés en favoris par les utilisateurs
        top_favs_raw = list(col_f.aggregate([
            {"$group": {"_id": "$code", "count": {"$sum": 1},
                        "name": {"$first": "$product_name"}}},
            {"$sort": {"count": -1}}, {"$limit": 5},
        ]))

        db_stats = {
            "total_products":  total_products,
            "total_users":     total_users,
            "total_favorites": total_favorites,
            "nb_admins":       sum(1 for u in users_list if u["role"] == "admin"),
            "grade_a":         grades.get("a", 0),
            "grade_e":         grades.get("e", 0),
        }

    except Exception:
        users_list, db_stats, top_favs_raw = [], {}, []

    return templates.TemplateResponse(request, "admin.html", {
        "user":       user,
        "users_list": users_list,
        "db_stats":   db_stats,
        "top_favs":   top_favs_raw,
    })


@app.post("/admin/role")
async def set_role(
    request:  Request,
    user_id:  str = Form(""),
    new_role: str = Form("user"),
):
    user = get_user(request)
    if not is_admin(user):
        return RedirectResponse("/", status_code=302)

    # On valide le rôle pour éviter d'injecter une valeur arbitraire en base
    if new_role not in ("user", "admin"):
        new_role = "user"

    from bson import ObjectId
    try:
        get_collection("users").update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"role": new_role}},
        )
    except Exception:
        pass

    return RedirectResponse("/admin", status_code=302)
