# trainer/train_item_neighbors.py
import os
import pandas as pd
from supabase import create_client

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_SERVICE_ROLE_KEY']
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_training_interactions():
    # reads a view Interactions_TrainView (pre-aggregated for training)
    q = supabase.table('Interactions_TrainView').select('*').execute()
    if q.error:
        raise Exception(q.error.message)
    return pd.DataFrame(q.data)

def compute_neighbors(df):
    # TODO: replace with your ALS training + neighbor extraction logic (implicit or lightfm)
    # Example return format: list of dicts: {'item_id': 'X', 'neighbor_id': 'Y', 'score': 0.78}
    neighbors = []
    # ... your ML code here ...
    return neighbors

def upsert_neighbors(neighbors):
    # batch upsert into ItemNeighbors table
    # expected columns in ItemNeighbors: item_id, neighbor_id, score, source, updated_at
    for chunk in [neighbors[i:i+500] for i in range(0, len(neighbors), 500)]:
        res = supabase.table('ItemNeighbors').upsert(chunk).execute()
        if res.error:
            print("Upsert error:", res.error)
        else:
            print("Upserted chunk, rows:", len(chunk))

def main():
    print("Fetching training data...")
    df = fetch_training_interactions()
    print("Training interactions:", len(df))
    neighbors = compute_neighbors(df)
    print("Computed neighbors:", len(neighbors))
    if neighbors:
        upsert_neighbors(neighbors)
    print("Done.")

if __name__ == "__main__":
    main()
