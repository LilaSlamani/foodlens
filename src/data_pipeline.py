# -*- coding: utf-8 -*-
"""
Pipeline ETL — Open Food Facts → MongoDB.
Lecture du CSV par chunks, nettoyage, filtrage qualité, import Atlas.

Usage :
    python src/data_pipeline.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

RACINE          = Path(__file__).parent.parent
CSV_BRUT_RAW    = RACINE / "data" / "raw" / "fr.openfoodfacts.org.products.csv"
CSV_BRUT_RACINE = RACINE / "fr.openfoodfacts.org.products.csv"
CSV_PROPRE      = RACINE / "data" / "processed" / "openfoodfacts_clean.csv"

# On ne lit que ces colonnes du CSV (~180 colonnes au total dans le fichier brut)
# pour limiter la mémoire et ne garder que ce dont l'app a besoin
COLS_SOUHAITEES = [
    "code", "product_name", "brands",
    "pnns_groups_1", "pnns_groups_2",
    "nutriscore_score", "nutriscore_grade",
    "nova_group", "additives_n",
    "energy-kcal_100g",
    "energy-kj_100g",   # utilisé pour calculer kcal si absent, puis supprimé
    "fat_100g", "saturated-fat_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g",
    "allergens", "completeness",
]

COLS_NUMERIQUES = [
    "energy-kcal_100g", "energy-kj_100g",
    "fat_100g", "saturated-fat_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g",
    "nutriscore_score", "nova_group", "additives_n", "completeness",
]

# Sécurité : on s'arrête avant d'atteindre la limite du tier gratuit Atlas (512 MB)
LIMITE_DOCS = 650_000
QUOTA_OCTET = 480 * 1024 * 1024   # 480 MB = marge de 32 MB


def _trouver_csv() -> Path:
    """Cherche le CSV brut à deux emplacements possibles."""
    if CSV_BRUT_RAW.exists():
        return CSV_BRUT_RAW
    if CSV_BRUT_RACINE.exists():
        return CSV_BRUT_RACINE
    raise FileNotFoundError(
        f"Fichier CSV introuvable.\n"
        f"  Attendu : {CSV_BRUT_RAW}\n"
        f"  ou      : {CSV_BRUT_RACINE}"
    )


def detecter_colonnes_disponibles(chemin_csv: Path) -> list:
    """Vérifie quelles colonnes souhaitées sont effectivement présentes dans le CSV."""
    entetes     = pd.read_csv(chemin_csv, sep="\t", nrows=0).columns.tolist()
    disponibles = [c for c in COLS_SOUHAITEES if c in entetes]
    manquantes  = [c for c in COLS_SOUHAITEES if c not in entetes]
    print(f"Colonnes trouvées   : {len(disponibles)}/{len(COLS_SOUHAITEES)}")
    if manquantes:
        print(f"Colonnes absentes   : {manquantes}")
    return disponibles


def nettoyer_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un chunk : typage, conversion kJ→kcal, suppression valeurs aberrantes."""

    for col in COLS_NUMERIQUES:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # Convertir kJ en kcal pour les produits qui n'ont pas de valeur kcal directe
    if "energy-kj_100g" in chunk.columns and "energy-kcal_100g" in chunk.columns:
        masque = chunk["energy-kcal_100g"].isna() & chunk["energy-kj_100g"].notna()
        chunk.loc[masque, "energy-kcal_100g"] = chunk.loc[masque, "energy-kj_100g"] / 4.184

    # Normaliser le grade en minuscule (le CSV contient parfois "A", "B"…)
    if "nutriscore_grade" in chunk.columns:
        chunk["nutriscore_grade"] = chunk["nutriscore_grade"].str.lower().str.strip()

    # Supprimer les valeurs nutritionnelles impossibles (>100g/100g ou négatives)
    cols_100g = [c for c in COLS_NUMERIQUES if "_100g" in c and c in chunk.columns]
    for col in cols_100g:
        chunk.loc[chunk[col] > 100, col] = np.nan
        chunk.loc[chunk[col] < 0,   col] = np.nan

    # On n'a plus besoin du kJ maintenant que kcal est calculé
    if "energy-kj_100g" in chunk.columns:
        chunk = chunk.drop(columns=["energy-kj_100g"])

    return chunk


def charger_et_nettoyer(chemin_csv: Path, taille_chunk: int = 50_000) -> pd.DataFrame:
    """
    Lit le CSV par blocs de 50k lignes pour ne pas saturer la RAM.
    Le fichier Open Food Facts fait ~3-4 Go non compressé.
    """
    colonnes       = detecter_colonnes_disponibles(chemin_csv)
    chunks_propres = []
    total_lignes   = 0

    print(f"Lecture par chunks de {taille_chunk:,} lignes...")

    for i, chunk in enumerate(pd.read_csv(
        chemin_csv,
        sep="\t",
        usecols=colonnes,
        chunksize=taille_chunk,
        low_memory=False,
        on_bad_lines="skip",   # ignore les lignes malformées sans planter
    )):
        chunk_propre = nettoyer_chunk(chunk)
        chunks_propres.append(chunk_propre)
        total_lignes += len(chunk_propre)
        print(f"  Chunk {i + 1} — {total_lignes:,} lignes", end="\r")

    df = pd.concat(chunks_propres, ignore_index=True)
    print(f"\nChargement terminé : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    return df


def filtrer_dataset(df: pd.DataFrame, seuil_completude: float = 0.3) -> pd.DataFrame:
    """
    Filtre qualité : on ne garde que les produits exploitables.
    Un produit sans Nutri-Score ou sans nom n'apporte rien à l'app.
    """
    avant = len(df)

    # completeness est un score OFF de 0 à 1 — en dessous de 0.3 la fiche est trop vide
    if "completeness" in df.columns:
        df = df[df["completeness"] >= seuil_completude]

    # Sans grade officiel, le produit ne sert pas à l'entraînement du modèle
    if "nutriscore_grade" in df.columns:
        df = df[df["nutriscore_grade"].isin(["a", "b", "c", "d", "e"])]

    if "product_name" in df.columns:
        df = df[df["product_name"].notna()]

    # pnns_groups_2 est utilisé pour la substitution — sans ça l'algo ne peut pas chercher
    if "pnns_groups_2" in df.columns:
        df = df[df["pnns_groups_2"].notna() & (df["pnns_groups_2"].str.lower() != "unknown")]

    # On exige au moins 4 valeurs nutritionnelles sur 6 pour que les barres aient du sens
    cols_nutrition = ["fat_100g", "saturated-fat_100g", "sugars_100g",
                      "proteins_100g", "salt_100g", "fiber_100g"]
    cols_presentes = [c for c in cols_nutrition if c in df.columns]
    df = df[df[cols_presentes].notna().sum(axis=1) >= 4]

    # Colonnes utilitaires qu'on ne stocke pas en base
    cols_drop = ["completeness", "additives_tags", "stores",
                 "categories", "ingredients_text", "energy-kj_100g"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])

    print(f"Filtre qualité : {len(df):,} produits conservés (supprimé {avant - len(df):,})")
    return df.copy()


def importer_mongodb(df: pd.DataFrame, batch_size: int = 2_000) -> None:
    """
    Importe les produits dans MongoDB par lots de 2000 avec upsert.
    Surveille la taille de la base toutes les 10 itérations pour éviter
    de dépasser la limite du tier gratuit Atlas (512 MB).
    """
    import dns.resolver
    # Forcer l'utilisation de DNS publics pour éviter les problèmes de résolution
    # en réseau d'entreprise ou avec certains FAI
    dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
    dns.resolver.default_resolver.nameservers = ["8.8.8.8", "1.1.1.1"]

    from pymongo import MongoClient, UpdateOne

    uri = os.getenv("MONGO_URI")
    if not uri:
        print("MONGO_URI absent dans .env — import ignoré")
        return

    client     = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    db         = client[os.getenv("DB_NAME", "foodlens")]
    collection = db["products"]
    collection.create_index("code", unique=True)

    records = df.to_dict(orient="records")
    total   = len(records)
    inseres = 0
    arret   = False

    print(f"Import MongoDB — {total:,} produits à traiter...")

    for i in range(0, total, batch_size):
        if arret:
            break

        batch      = records[i: i + batch_size]
        operations = []
        for rec in batch:
            # pandas représente les NaN comme float('nan') — MongoDB préfère None
            rec_propre = {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in rec.items()
            }
            # upsert : mise à jour si le code existe déjà, insertion sinon
            operations.append(UpdateOne(
                {"code": rec_propre.get("code")},
                {"$set": rec_propre},
                upsert=True,
            ))

        collection.bulk_write(operations, ordered=False)
        inseres += len(batch)

        # Vérification du quota toutes les 10 itérations (pas besoin de vérifier à chaque fois)
        if (i // batch_size) % 10 == 0:
            stats      = db.command("dbStats")
            storage_mb = stats["storageSize"] / 1024**2
            total_mb   = (stats["dataSize"] + stats["indexSize"]) / 1024**2
            docs_total = collection.estimated_document_count()

            print(f"  {inseres:,}/{total:,} traités | DB: {total_mb:.0f} MB | Docs: {docs_total:,}")

            if stats["storageSize"] >= QUOTA_OCTET:
                print(f"  Quota approché ({storage_mb:.0f} MB) — import stoppé.")
                arret = True
            elif docs_total >= LIMITE_DOCS:
                print(f"  Limite {LIMITE_DOCS:,} documents atteinte — import stoppé.")
                arret = True

    docs_final     = collection.estimated_document_count()
    stats          = db.command("dbStats")
    total_final_mb = (stats["dataSize"] + stats["indexSize"]) / 1024**2
    print(f"\nImport terminé — {docs_final:,} documents | {total_final_mb:.1f} MB utilisés")
    client.close()


def main():
    print("=" * 60)
    print("PIPELINE OPEN FOOD FACTS")
    print("=" * 60)

    if CSV_PROPRE.exists():
        # Si le CSV nettoyé existe déjà on l'utilise directement — évite de re-lire 3 Go
        print(f"CSV propre détecté — chargement direct : {CSV_PROPRE}")
        df_propre = pd.read_csv(CSV_PROPRE, low_memory=False)
        print(f"Chargé : {df_propre.shape[0]:,} lignes, {df_propre.shape[1]} colonnes")
        df_propre = filtrer_dataset(df_propre, seuil_completude=0.0)
    else:
        chemin_csv = _trouver_csv()
        print(f"Source : {chemin_csv}\n")
        df        = charger_et_nettoyer(chemin_csv)
        df_propre = filtrer_dataset(df, seuil_completude=0.3)
        kcal_ok   = df_propre["energy-kcal_100g"].notna().sum()
        print(f"Couverture kcal : {kcal_ok:,} / {len(df_propre):,} ({kcal_ok/len(df_propre)*100:.1f}%)")
        CSV_PROPRE.parent.mkdir(parents=True, exist_ok=True)
        df_propre.to_csv(CSV_PROPRE, index=False)
        print(f"CSV propre exporté : {CSV_PROPRE}")

    importer_mongodb(df_propre)
    print("\nPipeline terminé.")


if __name__ == "__main__":
    main()
