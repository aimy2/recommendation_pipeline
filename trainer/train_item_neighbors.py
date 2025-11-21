#!/usr/bin/env python3
# trainer/train_item_neighbors.py
"""
Complete trainer that:
- computes/aggregates interactions_trainview from raw interactions and upserts it
- trains an ALS model (implicit) on the aggregated view
- computes item-item neighbors robustly and upserts them into itemneighbors
- persists user embeddings for user_keys that are UUID-like into user_embeddings.user_id (uuid)

This file is self-contained (uses SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars).
Only additions vs your original file:
 - safer upsert fallback (upsert -> insert)
 - robust neighbor normalization (fixes index vs id confusion)
 - persist user embeddings for user_key values that are UUID-like
Everything else and original behavior kept unchanged.
"""

import os
import time
import re
from supabase import create_client
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

# --- Config / env ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

FACTORS = int(os.environ.get("FACTORS", 64))
REGULARIZATION = float(os.environ.get("REGULARIZATION", 0.01))
ITERATIONS = int(os.environ.get("ITERATIONS", 20))
TOP_K = int(os.environ.get("TOP_K", 50))
MIN_SCORE = float(os.environ.get("MIN_SCORE", 0.01))
CHUNK = int(os.environ.get("UPSERT_CHUNK", 500))
RAW_FETCH_LIMIT = int(os.environ.get("RAW_FETCH_LIMIT", 200000))  # safety limit when reading raw interactions

# regex to detect UUID-like strings (lower/upper hex)
UUID_RE = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')

# --- Utilities ---
def unwrap_response(resp):
    """
    Return (data, error) for different supabase client response shapes.
    Works with dict responses and object responses (APIResponse / pydantic).
    """
    try:
        # dict-like
        if isinstance(resp, dict):
            return resp.get("data"), resp.get("error")
        # object-like: prefer .data / .error, fallback to .json()
        data = getattr(resp, "data", None)
        error = getattr(resp, "error", None)
        if (data is None or error is None) and hasattr(resp, "json"):
            try:
                j = resp.json()
                if isinstance(j, dict):
                    data = j.get("data", data)
                    error = j.get("error", error)
            except Exception:
                pass
        return data, error
    except Exception as e:
        print("unwrap_response failed:", e)
        return None, {"message": "unwrap failed", "exception": str(e)}

# --- Trainview aggregation & upsert ---
def compute_trainview_from_raw(df_raw, limit=None):
    """
    Given raw interactions df (columns: user_id, session_id, product_id, weight, created_at, event_type),
    compute aggregated train view rows with columns: user_key, item_id, weight.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=['user_key', 'item_id', 'weight'])

    # coerce weight numeric
    df_raw['weight'] = pd.to_numeric(df_raw.get('weight', 1), errors='coerce').fillna(1.0)

    # build user_key (user_id if present else anon_session)
    df_raw['user_key'] = df_raw.apply(
        lambda r: str(r['user_id']) if r.get('user_id') else f"anon_{r.get('session_id')}", axis=1
    )

    # rename product_id -> item_id if necessary
    if 'product_id' in df_raw.columns:
        df_raw = df_raw.rename(columns={'product_id': 'item_id'})

    # group and sum weights
    agg = df_raw.groupby(['user_key', 'item_id'], as_index=False)['weight'].sum()

    # try to cast item_id to integer when possible
    try:
        agg['item_id'] = agg['item_id'].astype('Int64')
    except Exception:
        # leave as-is if conversion fails (e.g., uuid strings)
        pass

    if limit and len(agg) > limit:
        agg = agg.sample(n=limit, random_state=42).reset_index(drop=True)

    return agg

def safe_upsert_or_insert(table_name, chunk):
    """
    Try upsert first; if it errors, try insert. Return approx number inserted.
    """
    try:
        res = getattr(supabase, "table")(table_name).upsert(chunk).execute()
    except Exception as e:
        print(f"safe_upsert_or_insert: upsert raised exception: {e}")
        try:
            res2 = getattr(supabase, "table")(table_name).insert(chunk).execute()
            data2, error2 = unwrap_response(res2)
            if error2:
                print("safe_upsert_or_insert: insert fallback error:", error2)
                return 0
            # best-effort count
            return len(data2) if isinstance(data2, list) else 1
        except Exception as e2:
            print("safe_upsert_or_insert: insert fallback exception:", e2)
            return 0

    data, error = unwrap_response(res)
    # if error present, try insert fallback
    if error:
        print("safe_upsert_or_insert: upsert response error:", error)
        try:
            res2 = getattr(supabase, "table")(table_name).insert(chunk).execute()
            data2, error2 = unwrap_response(res2)
            if error2:
                print("safe_upsert_or_insert: insert fallback error:", error2)
                return 0
            return len(data2) if isinstance(data2, list) else 1
        except Exception as e2:
            print("safe_upsert_or_insert: insert fallback exception:", e2)
            return 0
    # success
    return len(data) if isinstance(data, list) else 1

def upsert_trainview(rows, chunk_size=CHUNK):
    """
    Upsert aggregated rows into interactions_trainview table.
    Each row dict should have: user_key, item_id, weight, optionally updated_at.
    Uses safe_upsert_or_insert logic.
    """
    if rows is None or len(rows) == 0:
        print("upsert_trainview: nothing to upsert")
        return 0

    # convert rows to list of dicts if DataFrame
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient='records')

    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i+chunk_size]
        # ensure types: cast item_id to int if possible, weight to float
        for r in chunk:
            try:
                if r.get('item_id') is not None:
                    r['item_id'] = int(r['item_id'])
            except Exception:
                # leave as-is if can't coerce
                pass
            try:
                r['weight'] = float(r.get('weight', 1.0))
            except Exception:
                r['weight'] = 1.0
            # optionally set updated_at to now if not present
            if 'updated_at' not in r:
                r['updated_at'] = None

        inserted = safe_upsert_or_insert('interactions_trainview', chunk)
        total += inserted
        print(f"upsert_trainview: chunk attempted {len(chunk)}, approx inserted/updated: {inserted}")

    print(f"upsert_trainview: attempted rows {len(rows)}, approx inserted/updated total: {total}")
    return total

# --- Fetching training interactions (reads interactions_trainview if present) ---
def fetch_training_interactions(limit=None):
    # try reading the aggregated view first
    resp = supabase.table('interactions_trainview').select('*').execute()
    data, error = unwrap_response(resp)
    if error:
        # fallback: aggregate raw interactions
        print("interactions_trainview not available or returned error:", error)
        resp2 = supabase.table('interactions').select('user_id,session_id,product_id,event_type,weight,created_at').limit(RAW_FETCH_LIMIT).execute()
        data2, error2 = unwrap_response(resp2)
        if error2:
            raise Exception(f"Error reading interactions: {error2}")
        df = pd.DataFrame(data2 or [])
        if df.empty:
            return df
        # coerce numeric weight
        df['weight'] = pd.to_numeric(df.get('weight', 1), errors='coerce').fillna(1.0)
        df['user_key'] = df.apply(lambda r: str(r['user_id']) if r.get('user_id') else f"anon_{r.get('session_id')}", axis=1)
        df = df.groupby(['user_key','product_id'], as_index=False)['weight'].sum()
        df = df.rename(columns={'product_id':'item_id'})
        if limit and len(df) > limit:
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        return df

    # normal path: read trainview
    df = pd.DataFrame(data or [])
    if df.empty:
        return df
    # ensure numeric weight column present
    if 'weight' not in df.columns:
        df['weight'] = 1.0
    else:
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1.0)

    # optional sampling for CI / dev
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    return df

# --- Matrix building / ALS training ---
def build_matrices(df):
    """
    Expect df with columns: user_key, item_id, weight
    Returns CSR matrix (items x users), item_uniques, user_uniques
    """
    if 'user_key' not in df.columns or 'item_id' not in df.columns:
        raise ValueError("DataFrame must contain 'user_key' and 'item_id' columns")

    user_codes, user_uniques = pd.factorize(df['user_key'])
    item_codes, item_uniques = pd.factorize(df['item_id'])
    data = df['weight'].astype(np.float32).values
    # implicit expects an item x user matrix
    matrix = coo_matrix((data, (item_codes, user_codes)), shape=(len(item_uniques), len(user_uniques)))
    return matrix.tocsr(), item_uniques, user_uniques

def train_als(item_user_csr):
    # ensure float32
    item_user_csr = item_user_csr.astype(np.float32)
    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        calculate_training_loss=False
    )
    model.fit(item_user_csr)
    return model

# --- Neighbors computation & upsert ---
def upsert_neighbors(neighbors, chunk_size=CHUNK):
    """
    Upsert neighbors list into itemneighbors table, chunked, with dedup within a chunk.
    Keeps the maximum score for duplicate (item_id, neighbor_id) pairs.
    """
    if not neighbors:
        print("upsert_neighbors: nothing to upsert")
        return 0

    total = 0
    for i in range(0, len(neighbors), chunk_size):
        chunk = neighbors[i:i+chunk_size]

        # Coerce types and build dedupe map: (item_id, neighbor_id) -> record with max score
        dedupe = {}
        for r in chunk:
            # try coerce numeric ids where possible (leave as-is otherwise)
            try:
                if r.get('item_id') is not None:
                    r['item_id'] = int(r['item_id'])
            except Exception:
                pass
            try:
                if r.get('neighbor_id') is not None:
                    r['neighbor_id'] = int(r['neighbor_id'])
            except Exception:
                pass
            try:
                r['score'] = float(r.get('score', 0.0))
            except Exception:
                r['score'] = 0.0
            if 'metadata' not in r:
                r['metadata'] = {}

            key = (r.get('item_id'), r.get('neighbor_id'))
            # keep the record with the maximum score
            existing = dedupe.get(key)
            if existing is None or r['score'] > existing['score']:
                dedupe[key] = r

        deduped_chunk = list(dedupe.values())

        # If nothing left after dedupe, continue
        if not deduped_chunk:
            print("upsert_neighbors: chunk had no deduped rows, skipping")
            continue

        inserted = safe_upsert_or_insert('itemneighbors', deduped_chunk)
        total += inserted
        print(f"upsert_neighbors: chunk attempted {len(chunk)} -> deduped {len(deduped_chunk)}, approx inserted: {inserted}")

    print(f"upsert_neighbors: total approx inserted: {total}")
    return total

def compute_and_upsert_neighbors(model, item_uniques):
    """
    Robust neighbor computation that handles cases where the ALS model's
    item_factors/user_factors axis may be transposed relative to item_uniques.

    Strategy:
    - If model.item_factors.shape[0] == len(item_uniques) => use model.item_factors
    - elif model.user_factors.shape[0] == len(item_uniques) => use model.user_factors
    - else: attempt safe fallbacks (use model.item_factors if available, but warn)
    - Compute cosine similarity between item vectors and upsert top-K neighbors.
    """
    import numpy as _np
    from math import isfinite

    # attempt to get factor matrices (may vary by implicit version)
    item_factors = getattr(model, "item_factors", None)
    user_factors = getattr(model, "user_factors", None)

    n_itemuni = len(item_uniques)
    chosen = None
    factors = None

    # choose the factor matrix that corresponds to items
    if item_factors is not None and item_factors.shape[0] == n_itemuni:
        chosen = "item_factors"
        factors = item_factors
    elif user_factors is not None and user_factors.shape[0] == n_itemuni:
        chosen = "user_factors"
        factors = user_factors
    elif item_factors is not None:
        # fallback: use item_factors anyway but warn (may produce only a subset)
        chosen = "item_factors_fallback"
        factors = item_factors
        print("compute_and_upsert_neighbors: WARNING - item_factors length does not equal len(item_uniques). Using item_factors anyway.")
    elif user_factors is not None:
        chosen = "user_factors_fallback"
        factors = user_factors
        print("compute_and_upsert_neighbors: WARNING - using user_factors fallback (no item_factors present).")
    else:
        print("compute_and_upsert_neighbors: no item_factors or user_factors found on model; aborting.")
        return

    print(f"compute_and_upsert_neighbors: chosen factor matrix = {chosen}, factors.shape = {getattr(factors, 'shape', None)}, len(item_uniques) = {n_itemuni}")

    # coerce to numpy array (float32)
    try:
        mat = _np.asarray(factors, dtype=_np.float32)
    except Exception as e:
        print("compute_and_upsert_neighbors: could not convert factors to numpy array:", e)
        return

    n_items_mat = mat.shape[0]
    dim = mat.shape[1] if mat.ndim > 1 else 1

    # If mat has fewer rows than item_uniques, we will only compute neighbors for the available rows.
    # But if n_items_mat == n_itemuni we cover everything.
    if n_items_mat < 1:
        print("compute_and_upsert_neighbors: factor matrix has zero rows; aborting.")
        return

    # normalize vectors for cosine similarity
    norms = _np.linalg.norm(mat, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0] = 1.0
    mat_norm = mat / norms

    # compute similarity matrix in chunks if needed (n_items small usually)
    # For n up to a few thousands, full mat @ mat.T is fine.
    try:
        sims_full = mat_norm @ mat_norm.T  # shape (n_items_mat, n_items_mat)
    except Exception as e:
        print("compute_and_upsert_neighbors: failed to compute similarity matrix:", e)
        return

    neighbors_to_upsert = []

    # For mapping: when we chose a matrix that has same length as item_uniques, 
    # index i in mat => item_uniques[i]. If mat is smaller, we only process indices up to n_items_mat.
    # Build mapping function from mat_index -> item_id (try int conversion)
    def mat_index_to_item_id(idx):
        if idx < 0 or idx >= n_itemuni:
            return None
        raw = item_uniques[idx]
        try:
            return int(raw)
        except Exception:
            return raw

    # For each item index in matrix, find top K neighbors (excluding self)
    for i in range(n_items_mat):
        item_id = mat_index_to_item_id(i)
        if item_id is None:
            # skip if cannot map
            continue

        row = sims_full[i]
        # row is numpy array length n_items_mat
        # exclude self by setting -inf
        row_i = row.copy()
        row_i[i] = -_np.inf

        # get top K indices using argpartition for speed
        k = min(TOP_K, len(row_i) - 1)
        if k <= 0:
            continue

        # get candidate indices (unsorted)
        try:
            idx_part = _np.argpartition(-row_i, k)[:k]
            # sort these by descending score
            idx_sorted = idx_part[_np.argsort(-row_i[idx_part])]
        except Exception:
            # fallback to argsort full
            idx_sorted = _np.argsort(-row_i)[:k]

        for j in idx_sorted:
            score = float(row[j])
            if not isfinite(score):
                continue
            if score <= MIN_SCORE:
                continue

            neighbor_item = mat_index_to_item_id(j)
            if neighbor_item is None:
                continue

            neighbors_to_upsert.append({
                "item_id": item_id,
                "neighbor_id": neighbor_item,
                "score": float(score),
                "source": "als",
                "metadata": {}
            })

        # flush periodically to avoid huge memory usage
        if len(neighbors_to_upsert) >= CHUNK:
            upsert_neighbors(neighbors_to_upsert, chunk_size=CHUNK)
            neighbors_to_upsert = []

    # final flush
    if neighbors_to_upsert:
        upsert_neighbors(neighbors_to_upsert, chunk_size=CHUNK)

    print("compute_and_upsert_neighbors: finished. attempted upsert count (last chunk may be partial).")



# --- NEW: persist user embeddings keyed by user_id uuid when user_key is UUID-like ---
def upsert_user_embeddings_from_userkeys(user_uniques, user_factors, chunk_size=CHUNK):
    """
    user_uniques: array-like mapping index -> user_key (text)
    user_factors: numpy array shape (n_users, dim)
    Only write rows where user_key matches UUID pattern; user_id will be that UUID string.
    """
    if user_factors is None or len(user_uniques) == 0:
        print("upsert_user_embeddings: no user factors to upsert")
        return 0

    rows = []
    for uid, vec in zip(user_uniques, user_factors):
        # uid here is 'user_key' from the training DataFrame
        if uid is None:
            continue
        uid_str = str(uid)
        # only persist if user_key is UUID-like
        if not UUID_RE.match(uid_str):
            # skip anonymous or non-uuid user_keys
            continue
        user_id = uid_str  # it's already UUID-like; Supabase/psql will accept as uuid string
        rows.append({
            "user_id": user_id,
            "embedding": list(map(float, vec.tolist())),
            "updated_at": None
        })

    if not rows:
        print("upsert_user_embeddings: nothing to upsert (no UUID-like user_keys found)")
        return 0

    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i+chunk_size]
        inserted = safe_upsert_or_insert('user_embeddings', chunk)
        total += inserted
        print(f"upsert_user_embeddings: chunk attempted {len(chunk)}, approx inserted: {inserted}")

    print("upsert_user_embeddings: total approx inserted:", total)
    return total

# --- Main flow (compute trainview, train, compute neighbors, persist user embeddings) ---
def main():
    print("TRAINER STARTED:", time.asctime())
    # Step 1: Fetch raw interactions and compute+upsert trainview
    try:
        resp_raw = supabase.table('interactions').select('user_id,session_id,product_id,event_type,weight,created_at').limit(RAW_FETCH_LIMIT).execute()
        raw_data, raw_err = unwrap_response(resp_raw)
        if raw_err:
            print("Error reading raw interactions:", raw_err)
            # fallback to reading existing trainview (if raw read fails)
            df = fetch_training_interactions()
        else:
            df_raw = pd.DataFrame(raw_data or [])
            print("Raw interactions rows fetched:", len(df_raw))
            agg = compute_trainview_from_raw(df_raw)
            print("Aggregated trainview rows to upsert:", len(agg))
            upsert_trainview(agg.to_dict(orient='records'))
            # read back trainview for canonical training data
            resp_tv = supabase.table('interactions_trainview').select('user_key,item_id,weight').limit(RAW_FETCH_LIMIT).execute()
            tv_data, tv_err = unwrap_response(resp_tv)
            if tv_err:
                print("Error reading interactions_trainview after upsert:", tv_err)
                # fallback to using agg for training
                df = agg.rename(columns={'item_id': 'item_id'})
            else:
                df = pd.DataFrame(tv_data or [])
                # ensure weight numeric
                if 'weight' in df.columns:
                    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1.0)
                else:
                    df['weight'] = 1.0
                # keep names consistent
                df = df.rename(columns={'item_id': 'item_id'})
                print("Training rows loaded from interactions_trainview:", len(df))
    except Exception as e:
        print("Exception during trainview compute/read:", e)
        df = fetch_training_interactions()

    if df is None or df.empty:
        print("No training data. Exiting.")
        return

    # build matrices and train
    try:
        print("Building matrices...")
        item_user_csr, item_uniques, user_uniques = build_matrices(df)
        print("Matrix shape (items x users):", item_user_csr.shape)
    except Exception as e:
        print("Error building matrices:", e)
        return

    try:
        print("Training ALS... (factors=%s iterations=%s)" % (FACTORS, ITERATIONS))
        model = train_als(item_user_csr)
    except Exception as e:
        print("Error training ALS:", e)
        return

    # debug prints to verify mapping sizes
    try:
        print("model.item_factors.shape:", getattr(model, "item_factors", None).shape)
        print("len(item_uniques):", len(item_uniques))
    except Exception:
        pass

    # compute & upsert neighbors
    try:
        print("Computing neighbors & upserting...")
        compute_and_upsert_neighbors(model, item_uniques)
    except Exception as e:
        print("Error computing/upserting neighbors:", e)
        return

    # persist user embeddings (NEW: convert user_key -> user_id uuid when possible)
    try:
        user_factors = getattr(model, "user_factors", None)
        if user_factors is not None:
            print("Persisting user embeddings (UUID only) count:", len(user_uniques))
            upsert_user_embeddings_from_userkeys(user_uniques, user_factors)
        else:
            print("No user_factors found on model; skipping user embeddings persist.")
    except Exception as e:
        print("Error persisting user embeddings:", e)

    print("TRAINER FINISHED:", time.asctime())

if __name__ == "__main__":
    main()
