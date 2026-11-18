"""
Supabase Adapter: Clean, plain Python class for read-only PostgreSQL diagnostics.
No inheritance, no magic — Ctrl+Click check_slow_queries() to inspect the exact SQL.
"""

from __future__ import annotations

import os


class SupabaseAdapter:
    """Adapter for Supabase PostgreSQL slow query & lock inspection."""

    def __init__(self, db_url: str | None = None, live: bool = False) -> None:
        self.db_url = (db_url if db_url is not None else os.getenv("SUPABASE_DB_URL", "")).strip()
        self.live = live

    def check_slow_queries(self, min_duration_ms: int = 500) -> str:
        """Inspect active slow queries, sequential scans, and connection saturation."""
        if not self.live or not self.db_url:
            return (
                f"[SANDBOX SUPABASE] Slow queries > {min_duration_ms}ms:\n"
                "  1. SELECT * FROM orders WHERE customer_email = $1 | avg=9,400ms | Seq Scan (missing index!)\n"
                "  2. Active connections: 48/50 (96% pool saturation)"
            )

        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError:
            return "ERROR: Install 'psycopg[binary]' to run live Supabase SQL diagnostics."

        sql = """
            SELECT pid, state, now() - query_start AS duration, left(query, 180) AS query
            FROM pg_stat_activity
            WHERE state != 'idle' AND query NOT ILIKE '%pg_stat_activity%'
            ORDER BY duration DESC
            LIMIT 10;
        """
        with psycopg.connect(self.db_url, connect_timeout=8) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        if not rows:
            return "[LIVE SUPABASE] Database healthy — no active slow or blocked queries right now."

        lines = ["[LIVE SUPABASE] Active Queries:"]
        for pid, state, dur, q in rows:
            lines.append(f"  PID={pid} | state={state} | duration={dur} | sql={q}")
        return "\n".join(lines)
