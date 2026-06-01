# -*- coding: utf-8 -*-
"""
Connexion MongoDB — accès aux collections users, favorites, products.
"""
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
import os

load_dotenv()

# Un seul client MongoDB partagé dans toute l'app — on évite d'ouvrir
# une nouvelle connexion à chaque requête (ça coûte cher en latence)
_client = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGO_URI"))
    return _client


def get_db():
    return get_client()[os.getenv("DB_NAME", "foodlens")]


def get_collection(nom: str):
    return get_db()[nom]


def initialiser_index() -> None:
    """
    Crée les index MongoDB au premier démarrage.
    À appeler une seule fois — MongoDB les ignore s'ils existent déjà.
    """
    db = get_db()

    # Email et username uniques pour éviter les doublons à l'inscription
    db["users"].create_index("email",    unique=True)
    db["users"].create_index("username", unique=True)

    # Index composé sur favorites : un utilisateur ne peut pas mettre deux fois
    # le même produit en favori (contrainte gérée côté base)
    db["favorites"].create_index(
        [("user_id", ASCENDING), ("code", ASCENDING)], unique=True
    )

    # Sur products : code pour les lookups par code-barres,
    # pnns_groups_2 pour les recherches de substituts,
    # nutriscore_grade pour les agrégats du dashboard
    db["products"].create_index("code", unique=True)
    db["products"].create_index("pnns_groups_2")
    db["products"].create_index("nutriscore_grade")

    print("Index MongoDB créés.")
