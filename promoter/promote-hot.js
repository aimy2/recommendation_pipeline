// promoter/promote-hot.js
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error("Missing SUPABASE envs");
  process.exit(1);
}
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const LOOKBACK_MINUTES = 60;
const TOP_K_PER_ITEM = 25;

async function fetch_recent_interactions(minutes = LOOKBACK_MINUTES) {
  const since = new Date(Date.now() - minutes * 60 * 1000).toISOString();
  const { data, error } = await supabase
    .from('interactions')
    .select('session_id, user_id, product_id, event_type, weight, created_at')
    .gte('created_at', since);
  if (error) throw error;
  return data || [];
}

function build_cooccurrence(interactions) {
  const bySession = {};
  for (const row of interactions) {
    const sid = row.session_id || (row.user_id ? `u_${row.user_id}` : null);
    if (!sid) continue;
    const pid = row.product_id;
    if (!pid) continue;
    bySession[sid] = bySession[sid] || new Set();
    bySession[sid].add(pid);
  }
  const co = {};
  for (const s of Object.keys(bySession)) {
    const items = Array.from(bySession[s]);
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < items.length; j++) {
        if (i === j) continue;
        const a = items[i], b = items[j];
        const key = `${a}::${b}`;
        co[key] = (co[key] || 0) + 1;
      }
    }
  }
  return co;
}

async function upsert_neighbors(neighbors) {
  if (!neighbors.length) return;
  const CHUNK = 500;
  for (let i = 0; i < neighbors.length; i += CHUNK) {
    const chunk = neighbors.slice(i, i + CHUNK);
    const { error } = await supabase.from('itemneighbors').upsert(chunk);
    if (error) console.error('Upsert error', error);
    else console.log('Upserted', chunk.length);
  }
}

(async () => {
  try {
    console.log('Fetching recent interactions...');
    const interactions = await fetch_recent_interactions();
    console.log('Rows:', interactions.length);
    const co = build_cooccurrence(interactions);
    const pairs = {};
    for (const k of Object.keys(co)) {
      const [a, b] = k.split('::').map(x => parseInt(x, 10));
      const count = co[k];
      pairs[a] = pairs[a] || [];
      pairs[a].push({ neighbor_id: b, count });
    }
    const neighbors = [];
    for (const a of Object.keys(pairs)) {
      const arr = pairs[a];
      arr.sort((x, y) => y.count - x.count);
      const top = arr.slice(0, TOP_K_PER_ITEM);
      const maxCount = top[0] ? top[0].count : 1;
      for (const p of top) {
        const score = p.count / maxCount;
        neighbors.push({
          item_id: parseInt(a, 10),
          neighbor_id: p.neighbor_id,
          score: parseFloat(score.toFixed(4)),
          source: 'promoter',
          metadata: { count: p.count }
        });
      }
    }
    await upsert_neighbors(neighbors);
    console.log('Promoter finished. Upserted:', neighbors.length);
    process.exit(0);
  } catch (err) {
    console.error('Promoter error', err);
    process.exit(1);
  }
})();
