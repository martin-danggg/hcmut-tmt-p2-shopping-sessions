#!/usr/bin/env python3
"""
Shopping Sessions: DuckDB pipeline (HCMUT TMT).

Reads  $DATA_DIR    (default /data)
Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
Temp   $SCRATCH_DIR (default /scratch)

Every rule below refers to rules.md. The CSV dialect is parsed by hand: DuckDB
reads each line as one opaque string and our SQL splits it, so DuckDB's own
quote / null / malformed-line handling never applies.
"""
import json
import os
from datetime import datetime, timezone

import duckdb
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = int(os.environ.get("NCPU", os.cpu_count() or 1))
MEM_GB = int(os.environ.get("MEM_GB", "8"))

DAY_MS = 86_400_000
GAP_MS = 1_800_000
WEEK_MS = 7 * DAY_MS

con = duckdb.connect()
con.execute(f"SET threads = {NCPU}")
con.execute(f"SET memory_limit = '{max(1, int(MEM_GB * 0.8))}GB'")
con.execute(f"SET temp_directory = '{os.path.join(SCRATCH, 'duckdb_tmp')}'")
con.execute("SET preserve_insertion_order = false")


def iso_ms(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp() * 1000)


with open(os.path.join(DATA, "manifest.json")) as f:
    manifest = json.load(f)
WIN_START = iso_ms(manifest["window_start"])
WIN_END = iso_ms(manifest["window_end"])


def q(s):
    return "'" + s.replace("'", "''") + "'"


# ---------------------------------------------------------------- CSV dialect
# Only "\n" ends a line; "\r" is an ordinary character. The first line of each
# file is its header. Fast path: DuckDB's reader in strict mode with quoting
# disabled and a delimiter that never occurs, so every line is one string. It
# refuses a "\r" (or a stray \x01), and then the exact slow path takes over:
# read each file whole and split on "\n" ourselves.
def raw_lines(table, folder):
    glob = os.path.join(DATA, folder, "*.csv")
    fast = f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT line FROM read_csv({q(glob)}, columns={{'line': 'VARCHAR'}}, delim=chr(1),
            quote='', escape='', header=true, auto_detect=false, new_line='\\n',
            strict_mode=true, null_padding=false)
    """
    slow = f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT unnest(string_split(content, chr(10))[2:]) AS line
        FROM read_text({q(glob)})
    """
    try:
        con.execute(fast)
    except duckdb.Error:
        con.execute(slow)


def parsed(table, folder, names):
    """Lines with exactly len(names) fields, each either bare or wholly quoted;
    quotes stripped, empty fields as NULL."""
    raw_lines(table + "_lines", folder)
    field = '(?:"[^",]*"|[^",]*)'
    pat = field + ("," + field) * (len(names) - 1)
    cols = ", ".join(f"f[{i + 1}] AS {n}" for i, n in enumerate(names))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT {cols} FROM (
            SELECT list_transform(string_split(line, ','),
                x -> CASE WHEN x = '' OR x = '""' THEN NULL
                          WHEN starts_with(x, '"') THEN x[2:-2]
                          ELSE x END) AS f
            FROM {table}_lines
            WHERE line IS NOT NULL AND regexp_full_match(line, {q(pat)})
        )
    """)
    con.execute(f"DROP TABLE {table}_lines")


def parsed_file(table, path, names):
    """Same dialect for products.csv / campaigns.csv (a single file)."""
    folder_tmp = path  # read_csv/read_text accept a plain path as glob
    field = '(?:"[^",]*"|[^",]*)'
    pat = field + ("," + field) * (len(names) - 1)
    cols = ", ".join(f"f[{i + 1}] AS {n}" for i, n in enumerate(names))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT {cols} FROM (
            SELECT list_transform(string_split(line, ','),
                x -> CASE WHEN x = '' OR x = '""' THEN NULL
                          WHEN starts_with(x, '"') THEN x[2:-2]
                          ELSE x END) AS f
            FROM (SELECT unnest(string_split(content, chr(10))[2:]) AS line FROM read_text({q(folder_tmp)}))
            WHERE regexp_full_match(line, {q(pat)})
        )
    """)


# ---------------------------------------------------------------- timestamps
# F1 2026-10-03T23:58:10Z, F2 ...-04:00, F3 epoch ms digits. Strict ASCII digits,
# strict calendar, offset hours 00-23 / minutes 00-59. Returns NULL when bad.
con.execute("""
CREATE MACRO dim(y, m) AS CASE
    WHEN m IN (1, 3, 5, 7, 8, 10, 12) THEN 31
    WHEN m IN (4, 6, 9, 11) THEN 30
    WHEN (y % 4 = 0 AND y % 100 <> 0) OR y % 400 = 0 THEN 29
    ELSE 28 END
""")
# days since 1970-01-01 (Howard Hinnant's days_from_civil, y >= 0)
con.execute("""
CREATE MACRO civ(y, m, d) AS
    ((y - (m <= 2)::INT) // 400) * 146097
    + ((y - (m <= 2)::INT) - ((y - (m <= 2)::INT) // 400) * 400) * 365
    + ((y - (m <= 2)::INT) - ((y - (m <= 2)::INT) // 400) * 400) // 4
    - ((y - (m <= 2)::INT) - ((y - (m <= 2)::INT) // 400) * 400) // 100
    + (153 * (m + CASE WHEN m > 2 THEN -3 ELSE 9 END) + 2) // 5 + d - 1
    - 719468
""")
con.execute("""
CREATE MACRO f12(y, mo, d, h, mi, se, z) AS CASE
    WHEN mo BETWEEN 1 AND 12 AND d BETWEEN 1 AND dim(y, mo) AND h <= 23 AND mi <= 59 AND se <= 59
     AND (z = 'Z' OR (TRY_CAST(z[2:3] AS INT) <= 23 AND TRY_CAST(z[5:6] AS INT) <= 59))
    THEN civ(y, mo, d)::BIGINT * 86400000 + (h * 3600 + mi * 60 + se)::BIGINT * 1000
         - CASE WHEN z = 'Z' THEN 0
                ELSE (CASE WHEN z[1] = '-' THEN -1 ELSE 1 END)
                     * (TRY_CAST(z[2:3] AS BIGINT) * 3600000 + TRY_CAST(z[5:6] AS BIGINT) * 60000) END
    END
""")
con.execute("""
CREATE MACRO tsms(s) AS CASE
    WHEN regexp_full_match(s, '[0-9]+') THEN
        CASE WHEN length(ltrim(s, '0')) <= 15 THEN TRY_CAST('0' || ltrim(s, '0') AS BIGINT) END
    WHEN regexp_full_match(s, '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(Z|[+-][0-9]{2}:[0-9]{2})') THEN
        f12(TRY_CAST(s[1:4] AS INT), TRY_CAST(s[6:7] AS INT), TRY_CAST(s[9:10] AS INT),
            TRY_CAST(s[12:13] AS INT), TRY_CAST(s[15:16] AS INT), TRY_CAST(s[18:19] AS INT), s[20:])
    END
""")

# ---------------------------------------------------------------- links -> person
parsed("links", "links", ["a", "b"])
con.execute("DELETE FROM links WHERE a IS NULL OR b IS NULL")
con.execute("""
    CREATE TEMP TABLE lid AS
    SELECT id, (row_number() OVER (ORDER BY id) - 1)::BIGINT AS ix
    FROM (SELECT a AS id FROM links UNION SELECT b FROM links)
""")
n_ids = con.execute("SELECT count(*) FROM lid").fetchone()[0]
if n_ids:
    ea, eb = con.execute("""
        SELECT x.ix, y.ix FROM links l JOIN lid x ON l.a = x.id JOIN lid y ON l.b = y.id
    """).fetchnumpy().values()
    g = coo_matrix((np.ones(len(ea), dtype=np.int8), (ea, eb)), shape=(n_ids, n_ids))
    _, label = connected_components(g, directed=False)
    # ix is the byte-wise rank, so the smallest ix of a component is its name
    root = np.full(label.max() + 1, n_ids, dtype=np.int64)
    np.minimum.at(root, label, np.arange(n_ids, dtype=np.int64))
    import pyarrow as pa
    rep = pa.table({"ix": np.arange(n_ids, dtype=np.int64), "rix": root[label]})
    con.register("rep_arrow", rep)
    con.execute("""
        CREATE TEMP TABLE pmap AS
        SELECT a.id, b.id AS person
        FROM rep_arrow r JOIN lid a ON a.ix = r.ix JOIN lid b ON b.ix = r.rix
        WHERE a.id <> b.id
    """)
    con.unregister("rep_arrow")
else:
    con.execute("CREATE TEMP TABLE pmap (id VARCHAR, person VARCHAR)")

# ---------------------------------------------------------------- reference data
parsed_file("products", os.path.join(DATA, "products.csv"), ["product_id", "category"])
parsed_file("campaigns", os.path.join(DATA, "campaigns.csv"), ["campaign_id", "channel"])

# ---------------------------------------------------------------- events
parsed("ev_raw", "events", ["eid", "uid", "ts", "ing", "typ", "pid", "price"])
# dedup on event_id before validation: the copy with the smallest ingest_ts.
# ingest_ts is always valid F1 (fixed width), so string order is time order.
con.execute(f"""
    CREATE TEMP TABLE ev AS
    WITH k AS (
        SELECT arg_min(struct_pack(uid, ts, ing, typ, pid, price), ing) AS r
        FROM ev_raw WHERE eid IS NOT NULL GROUP BY eid
    ), v AS (
        SELECT r.uid AS uid, tsms(r.ts) AS ts, tsms(r.ing) AS ing, r.typ AS typ, r.pid AS pid, r.price AS price
        FROM k
    )
    SELECT coalesce(m.person, v.uid) AS person, v.ts, v.typ, v.pid,
           CASE WHEN v.typ = 'purchase' THEN TRY_CAST('0' || ltrim(v.price, '0') AS BIGINT) ELSE 0 END AS rev
    FROM v LEFT JOIN pmap m ON m.id = v.uid
    WHERE v.uid IS NOT NULL AND v.pid IS NOT NULL
      AND v.ts IS NOT NULL AND v.ts >= {WIN_START} AND v.ts < {WIN_END}
      AND v.typ IN ('view', 'cart', 'purchase')
      AND (v.typ <> 'purchase' OR (regexp_full_match(coalesce(v.price, ''), '[0-9]+')
                                   AND length(ltrim(v.price, '0')) BETWEEN 1 AND 8
                                   AND TRY_CAST('0' || ltrim(v.price, '0') AS BIGINT) <= 10000000))
      AND v.ing - v.ts <= {DAY_MS}
""")
con.execute("DROP TABLE ev_raw")

# sessionize per person on distinct timestamps
con.execute(f"""
    CREATE TEMP TABLE tsmap AS
    WITH d AS (SELECT DISTINCT person, ts FROM ev),
    f AS (
        SELECT person, ts,
               CASE WHEN lag(ts) OVER w IS NULL OR ts - lag(ts) OVER w > {GAP_MS} THEN ts END AS st
        FROM d WINDOW w AS (PARTITION BY person ORDER BY ts)
    )
    SELECT person, ts, max(st) OVER (PARTITION BY person ORDER BY ts
                                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS start
    FROM f
""")
con.execute(f"""
    CREATE TEMP TABLE evs AS
    SELECT e.person, t.start, (DATE '1970-01-01' + TRY_CAST(t.start // {DAY_MS} AS INT)) AS day,
           e.typ, e.pid, e.rev
    FROM ev e JOIN tsmap t ON e.person = t.person AND e.ts = t.ts
""")
con.execute("""
    CREATE TEMP TABLE sess AS
    SELECT person, start, any_value(day) AS day, count(*) AS n, sum(rev) AS rev,
           bool_or(typ = 'purchase') AS has_p
    FROM evs GROUP BY person, start
""")

# ---------------------------------------------------------------- clicks
parsed("ck_raw", "clicks", ["cid", "uid", "camp", "ts"])
con.execute(f"""
    CREATE TEMP TABLE ck AS
    WITH k AS (
        SELECT cid, any_value(struct_pack(uid, camp, ts)) AS r FROM ck_raw WHERE cid IS NOT NULL GROUP BY cid
    ), v AS (SELECT cid, r.uid AS uid, r.camp AS camp, tsms(r.ts) AS ts FROM k)
    SELECT coalesce(m.person, v.uid) AS person, v.ts, v.cid, v.camp
    FROM v LEFT JOIN pmap m ON m.id = v.uid
    WHERE v.uid IS NOT NULL AND v.ts IS NOT NULL
      AND v.ts >= {WIN_START - WEEK_MS} AND v.ts < {WIN_END}
      AND v.camp IN (SELECT campaign_id FROM campaigns)
""")
con.execute("DROP TABLE ck_raw")

# credit: latest click at or before the start within 7 days; tie -> smallest click_id
con.execute(f"""
    CREATE TEMP TABLE cbest AS
    SELECT person, ts, arg_min(camp, cid) AS camp FROM ck GROUP BY person, ts
""")
con.execute(f"""
    CREATE TEMP TABLE credit AS
    SELECT s.person, s.start, s.day, s.rev, s.has_p,
           CASE WHEN c.ts IS NOT NULL AND s.start - c.ts <= {WEEK_MS} THEN c.camp ELSE 'organic' END AS camp
    FROM sess s ASOF LEFT JOIN cbest c ON s.person = c.person AND s.start >= c.ts
""")
# assists: every campaign with a click in [start - 7d, start]
con.execute(f"""
    CREATE TEMP TABLE assist AS
    SELECT DISTINCT s.person, s.start, s.day, c.camp
    FROM sess s JOIN (SELECT DISTINCT person, camp, ts FROM ck) c
      ON s.person = c.person AND c.ts <= s.start AND c.ts >= s.start - {WEEK_MS}
""")

# ---------------------------------------------------------------- outputs
os.makedirs(os.path.join(OUT, "user_daily"), exist_ok=True)
os.makedirs(os.path.join(OUT, "campaign_daily"), exist_ok=True)
con.execute(f"""
    COPY (
        WITH u AS (
            SELECT person, day, count(*) AS sessions, sum(n) AS events, sum(rev) AS revenue_cents
            FROM sess GROUP BY person, day
        ), c AS (
            SELECT e.person, e.day, coalesce(p.category, 'unknown') AS cat, sum(e.rev) AS amt
            FROM evs e LEFT JOIN products p ON p.product_id = e.pid
            WHERE e.typ = 'purchase' GROUP BY ALL
        ), t2 AS (
            SELECT person, day, cat FROM c
            QUALIFY row_number() OVER (PARTITION BY person, day ORDER BY amt DESC, cat ASC) = 1
        )
        SELECT u.person AS user_id, u.day, u.sessions::BIGINT AS sessions, u.events::BIGINT AS events,
               u.revenue_cents::BIGINT AS revenue_cents, t2.cat AS top_category
        FROM u LEFT JOIN t2 ON u.person = t2.person AND u.day = t2.day
    ) TO {q(os.path.join(OUT, "user_daily", "part-0.parquet"))} (FORMAT parquet)
""")
con.execute(f"""
    COPY (
        WITH cr AS (
            SELECT camp, day, count(*) AS sessions, count(*) FILTER (WHERE has_p) AS ps, sum(rev) AS rev
            FROM credit GROUP BY camp, day
        ), a AS (
            SELECT camp, day, count(*) AS asess FROM assist GROUP BY camp, day
        ), j AS (
            SELECT coalesce(cr.camp, a.camp) AS camp, coalesce(cr.day, a.day) AS day,
                   coalesce(cr.sessions, 0) AS sessions, coalesce(cr.ps, 0) AS ps,
                   coalesce(cr.rev, 0) AS rev, coalesce(a.asess, 0) AS asess
            FROM cr FULL OUTER JOIN a ON cr.camp = a.camp AND cr.day = a.day
        )
        SELECT j.camp AS campaign_id,
               CASE WHEN j.camp = 'organic' THEN 'organic' ELSE k.channel END AS channel,
               j.day, j.sessions::BIGINT AS sessions, j.ps::BIGINT AS purchasing_sessions,
               j.rev::BIGINT AS revenue_cents, j.asess::BIGINT AS assisted_sessions
        FROM j LEFT JOIN campaigns k ON k.campaign_id = j.camp
    ) TO {q(os.path.join(OUT, "campaign_daily", "part-0.parquet"))} (FORMAT parquet)
""")
print("done")
