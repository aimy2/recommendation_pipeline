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

# safe unwrap for supabase responses (place near imports)
def unwrap_response(resp):
    """
    Return (data, error) for different supabase client response shapes.
    Works with dict responses and object responses (APIResponse / pydantic).
    """
    # dict-like
    if isinstance(resp, dict):
        return resp.get("data"), resp.get("error")
    # object-like: prefer .data / .error, fallback to .json()
    data = getattr(resp, "data", None)
    error = getattr(resp, "error", None)
    # some versions expose a method .json() returning dict
    if (data is None or error is None) and hasattr(resp, "json"):
        try:
            j = resp.json()
            if isinstance(j, dict):
                data = j.get("data", data)
                error = j.get("error", error)
        except Exception:
            pass
    return data, error


def fetch_training_interactions(limit=None):
    # try reading the aggregated view first
    q = supabase.table('interactions_trainview').select('*').execute()
    data, error = unwrap_response(q)
    if error:
        # fallback: aggregate raw interactions
        print("interactions_trainview not available or returned error:", error)
        q2 = supabase.table('interactions').select('user_id,session_id,product_id,event_type,weight,created_at').limit(100000).execute()
        data2, error2 = unwrap_response(q2)
        if error2:
            raise Exception(f"Error reading interactions: {error2}")
        df = pd.DataFrame(data2)
        if df.empty:
            return df
        df['weight'] = df['weight'].fillna(1)
        df['user_key'] = df.apply(lambda r: str(r['user_id']) if r.get('user_id') else f"anon_{r.get('session_id')}", axis=1)
        df = df.groupby(['user_key','product_id'], as_index=False)['weight'].sum()
        df = df.rename(columns={'product_id':'item_id'})
        if limit and len(df) > limit:
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        return df

    # normal path
    df = pd.DataFrame(data or [])
    if df.empty:
        return df
    # optional sampling for CI / dev
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
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
        data, error = unwrap_response(res)
        if error:
            # Supabase may return error as dict/string — print for debugging and continue or raise
            print("Upsert error:", error)
            # raise Exception(error)   # uncomment to fail hard on upsert errors
        else:
            print(f"Upserted {len(chunk)} neighbors")

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
