#!/usr/bin/env python3
# trainer/train_item_neighbors.py
import os, time
from supabase import create_client
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

FACTORS = 64
REGULARIZATION = 0.01
ITERATIONS = 20
TOP_K = 50
MIN_SCORE = 0.01

def fetch_training_interactions(limit=None):
    q = supabase.table('interactions_trainview').select('*').execute()
    if q.error:
        # fallback to aggregating interactions if view not present
        q2 = supabase.table('interactions').select('user_id,session_id,product_id,event_type,weight,created_at').limit(100000).execute()
        if q2.error:
            raise Exception(q2.error.message)
        df = pd.DataFrame(q2.data)
        if df.empty:
            return df
        wmap = {}
        df['weight'] = df['weight'].fillna(1)
        df['user_key'] = df.apply(lambda r: str(r['user_id']) if r.get('user_id') else f"anon_{r.get('session_id')}", axis=1)
        df = df.groupby(['user_key','product_id'], as_index=False)['weight'].sum()
        df = df.rename(columns={'product_id':'item_id'})
        return df
    df = pd.DataFrame(q.data)
    return df

def build_matrices(df):
    user_codes, user_uniques = pd.factorize(df['user_key'])
    item_codes, item_uniques = pd.factorize(df['item_id'])
    data = df['weight'].astype(float).values
    matrix = coo_matrix((data, (item_codes, user_codes)))
    return matrix.tocsr(), item_uniques, user_uniques

def train_als(item_user_csr):
    model = AlternatingLeastSquares(factors=FACTORS, regularization=REGULARIZATION, iterations=ITERATIONS, calculate_training_loss=False)
    model.fit(item_user_csr)
    return model

def upsert_neighbors(neighbors):
    CHUNK = 500
    for i in range(0, len(neighbors), CHUNK):
        chunk = neighbors[i:i+CHUNK]
        res = supabase.table('itemneighbors').upsert(chunk).execute()
        if res.error:
            print("Upsert error:", res.error)
        else:
            print(f"Upserted {len(chunk)} rows")

def compute_and_upsert_neighbors(model, item_uniques):
    neighbors_to_upsert = []
    for item_idx in range(len(item_uniques)):
        item_id = int(item_uniques[item_idx])
        try:
            sims = model.similar_items(item_idx, N=TOP_K+1)
        except Exception as e:
            print("similar_items error", e)
            continue
        for sim_idx, score in sims:
            if sim_idx == item_idx: continue
            if score <= MIN_SCORE: continue
            neighbor_item = int(item_uniques[sim_idx])
            neighbors_to_upsert.append({
                "item_id": item_id,
                "neighbor_id": neighbor_item,
                "score": float(score),
                "source": "als",
                "metadata": {}
            })
        if len(neighbors_to_upsert) >= 500:
            upsert_neighbors(neighbors_to_upsert)
            neighbors_to_upsert = []
    if neighbors_to_upsert:
        upsert_neighbors(neighbors_to_upsert)

def main():
    print("Fetching training data...")
    df = fetch_training_interactions()
    print("Rows:", len(df))
    if df.empty:
        print("No data. Exiting.")
        return
    print("Building matrices...")
    item_user_csr, item_uniques, user_uniques = build_matrices(df)
    print("Training ALS...")
    model = train_als(item_user_csr)
    print("Computing neighbors & upserting...")
    compute_and_upsert_neighbors(model, item_uniques)
    print("Done.")

if __name__ == "__main__":
    main()
