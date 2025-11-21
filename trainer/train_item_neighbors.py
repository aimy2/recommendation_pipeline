# run in trainer env (same SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)
from supabase import create_client
import os

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_SERVICE_ROLE_KEY']
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

test_row = {
    "item_id": 999999999,
    "neighbor_id": 111111111,
    "score": 0.5,
    "source": "test-upsert",
    "metadata": {}
}

print("Attempting test upsert...")
r = supabase.table('itemneighbors').upsert([test_row]).execute()
print("upsert.data:", getattr(r, "data", None))
print("upsert.error:", getattr(r, "error", None))

print("\nReading back test rows...")
rr = supabase.table('itemneighbors').select('*').eq('source', 'test-upsert').execute()
print("read.data:", getattr(rr, "data", None))
print("read.error:", getattr(rr, "error", None))
