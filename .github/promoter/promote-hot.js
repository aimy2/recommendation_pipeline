// promoter/promote-hot.js
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function fetch_recent_interactions(minutes = 60) {
  // expects Interactions table with created_at timestamp
  const since = new Date(Date.now() - minutes*60*1000).toISOString();
  const { data, error } = await supabase
    .from('Interactions')
    .select('*')
    .gte('created_at', since);

  if (error) throw error;
  return data || [];
}

function compute_promoted_neighbors(interactions) {
  // TODO: implement session co-occurrence
  // Return array of objects matching ItemNeighbors schema:
  // { item_id: 'A', neighbor_id: 'B', score: 0.9, source: 'promoter' }
  const out = [];
  // ... your promotion logic ...
  return out;
}

async function upsert_neighbors(neighbors) {
  if (!neighbors.length) return;
  // chunk to 500
  for (let i=0; i<neighbors.length; i+=500) {
    const chunk = neighbors.slice(i, i+500);
    const { error } = await supabase.from('ItemNeighbors').upsert(chunk);
    if (error) console.error('Upsert error:', error);
    else console.log('Upserted', chunk.length);
  }
}

(async () => {
  try {
    console.log('Fetching recent interactions...');
    const interactions = await fetch_recent_interactions(30);
    console.log('Interactions:', interactions.length);
    const neighbors = compute_promoted_neighbors(interactions);
    console.log('Promoted neighbors computed:', neighbors.length);
    await upsert_neighbors(neighbors);
    console.log('Promoter finished.');
  } catch (err) {
    console.error(err);
    process.exit(1);
  }
})();
