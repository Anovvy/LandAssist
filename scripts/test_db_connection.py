"""Test and verify Supabase PostGIS connection for LandAssist.

Usage:
    python scripts/test_db_connection.py
    python scripts/test_db_connection.py --url "postgresql://..."
    python scripts/test_db_connection.py --encode-password "MyP@ssw0rd!"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

# Fix Windows console UTF-8 output encoding for emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path if needed
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(filepath: Path) -> dict[str, str]:
    """Simple parser for .env file without external dependencies."""
    env_vars: dict[str, str] = {}
    if not filepath.exists():
        return env_vars
    for raw_line in filepath.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            env_vars[key.strip()] = val.strip().strip("'\"")
    return env_vars


def escape_password(raw_password: str) -> str:
    """URL-encode password for PostgreSQL connection string."""
    return urllib.parse.quote_plus(raw_password)


def parse_connection_info(url: str) -> dict[str, str | int]:
    """Parse DSN string into human-readable components."""
    try:
        parsed = urllib.parse.urlsplit(url)
        return {
            "scheme": parsed.scheme,
            "username": parsed.username or "",
            "hostname": parsed.hostname or "",
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/") or "postgres",
        }
    except Exception as exc:
        return {"error": str(exc)}


async def test_connection(dsn: str) -> bool:
    """Test connection, verify PostGIS, and inspect LandAssist schema."""
    try:
        import asyncpg
    except ImportError:
        print("\n❌ Error: 'asyncpg' module is not installed.")
        print("👉 Run: pip install asyncpg\n")
        return False

    info = parse_connection_info(dsn)
    port = info.get("port", 5432)
    host = info.get("hostname", "")
    user = info.get("username", "")
    db = info.get("database", "postgres")

    print("\n" + "=" * 65)
    print("🌍 LandAssist — Supabase PostGIS Connection Diagnostic")
    print("=" * 65)
    print(f"📡 Target Host : {host}")
    print(f"🔌 Port        : {port}")
    print(f"👤 User        : {user}")
    print(f"🗄️  Database    : {db}")

    is_pooler = "pooler.supabase.com" in str(host) or port == 6543
    if is_pooler:
        print("ℹ️  Mode        : Supavisor Connection Pooler (IPv4 Transaction Mode)")
    else:
        print("ℹ️  Mode        : Direct Connection (Port 5432, IPv6)")

    print("-" * 65)
    print("⏳ Connecting to PostgreSQL...", end="", flush=True)

    start_time = time.perf_counter()
    conn = None
    try:
        # For transaction pooler (port 6543), disable statement caching
        kwargs = {"timeout": 15.0}
        if port == 6543:
            kwargs["statement_cache_size"] = 0

        conn = await asyncpg.connect(dsn, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print(f" ✅ SUCCESS! ({elapsed_ms:.1f} ms)")

        # 1. Check Server Version
        pg_version = await conn.fetchval("SELECT version();")
        short_version = pg_version.split(",")[0] if pg_version else "Unknown"
        print(f"\n📦 PostgreSQL Version : {short_version}")

        # 2. Check PostGIS Extension
        try:
            postgis_ver = await conn.fetchval("SELECT postgis_full_version();")
            pgis_match = re.search(r"POSTGIS=\"([^\"]+)\"", postgis_ver or "")
            clean_pgis = pgis_match.group(1) if pgis_match else postgis_ver[:40]
            print(f"🗺️  PostGIS Extension   : ✅ Active (Version: {clean_pgis})")
        except Exception as e:
            print("🗺️  PostGIS Extension   : ❌ Not active!")
            print(f"   Error: {e}")
            print("   👉 Solution: Execute 'CREATE EXTENSION postgis;' in Supabase SQL Editor.")

        # 3. Check Table `landuse_predictions`
        table_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'landuse_predictions'
            );
            """
        )

        if table_exists:
            row_count = await conn.fetchval("SELECT count(*) FROM landuse_predictions;")
            print(f"📊 Table 'landuse_predictions': ✅ Found ({row_count} rows)")

            # Check Spatial Index
            has_spatial_idx = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE tablename = 'landuse_predictions'
                    AND indexname = 'idx_landuse_predictions_geom'
                );
                """
            )
            idx_status = "✅ Active (GIST)" if has_spatial_idx else "⚠️ Not created"
            print(f"⚡ Spatial Index (GIST)        : {idx_status}")

            # Check Helper Function get_landuse_mvt
            func_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_proc
                    WHERE proname = 'get_landuse_mvt'
                );
                """
            )
            func_status = "✅ Ready (Vector Tiles)" if func_exists else "⚠️ Not created"
            print(f"🧩 MVT Function (get_landuse_mvt): {func_status}")
        else:
            print("📊 Table 'landuse_predictions': ⚠️ Not found in public schema.")
            print("   👉 Solution: Execute 'sql/001_init_postgis.sql' in Supabase SQL Editor.")

        # 4. Geometry Sanity Test
        try:
            sample_point = await conn.fetchval(
                "SELECT ST_AsText(ST_GeomFromText('POINT(106.8456 -6.2088)', 4326));"
            )
            print(f"📍 PostGIS Point Sanity Test   : ✅ Passed ({sample_point})")
        except Exception as e:
            print(f"📍 PostGIS Point Sanity Test   : ❌ Failed ({e})")

        print("\n" + "=" * 65)
        print("🎉 Database connection & PostGIS verified 100%! Ready for deployment!")
        print("=" * 65 + "\n")
        return True

    except asyncpg.exceptions.InvalidPasswordError:
        print(" ❌ FAILED!")
        print("\n❌ Error: Invalid database password!")
        print("💡 Troubleshooting Tips:")
        print("  1. Verify that the password matches your Supabase project password.")
        print("  2. If your password contains special characters (@, #, $, %, etc.), it MUST be URL-encoded.")
        print("     Run encoder helper: python scripts/test_db_connection.py --encode-password \"your_password\"")
        return False

    except (OSError, TimeoutError) as exc:
        print(" ❌ FAILED!")
        print(f"\n❌ Network / Timeout Error: {exc}")
        print("💡 Diagnostic Tips:")
        if not is_pooler and port == 5432:
            print("  1. Direct connection 'db.<project>.supabase.co:5432' requires IPv6 routing.")
            print("     Some ISPs / local networks do not have full IPv6 support.")
            print("  2. Solution: Use Supavisor Connection Pooler (Port 6543, IPv4):")
            print("     postgresql://postgres.<project_ref>:[PASSWORD]@aws-0-<region>.pooler.supabase.com:6543/postgres")
        else:
            print("  1. Check your internet connection.")
            print("  2. Ensure your Supabase project is not paused.")
        return False

    except Exception as exc:
        print(" ❌ FAILED!")
        print(f"\n❌ Error detail: {type(exc).__name__}: {exc}")
        return False

    finally:
        if conn is not None:
            await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LandAssist Supabase PostGIS Connection Tester & Diagnostics"
    )
    parser.add_argument(
        "--url",
        "-u",
        help="PostgreSQL connection string (DSN). Default: loaded from .env (DATABASE_URL)",
    )
    parser.add_argument(
        "--encode-password",
        "-e",
        help="Helper to URL-encode database passwords containing special characters.",
    )

    args = parser.parse_args()

    # Password encoder utility
    if args.encode_password:
        raw_pw = args.encode_password
        encoded = escape_password(raw_pw)
        print("\n🔐 URL-Encoded Password Helper:")
        print(f"Original : {raw_pw}")
        print(f"Encoded  : {encoded}")
        print("\nExample usage in DATABASE_URL:")
        print(f"postgresql://postgres.xxx:{encoded}@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres\n")
        return

    # Determine DATABASE_URL
    target_url = args.url
    if not target_url:
        env_path = REPO_ROOT / ".env"
        env_vars = load_env_file(env_path)
        target_url = env_vars.get("DATABASE_URL") or os.environ.get("DATABASE_URL")

    if not target_url or "password@db.xxxx" in target_url:
        print("\n⚠️  DATABASE_URL is not configured!")
        print("You can provide it using one of the following methods:")
        print("1. Set DATABASE_URL in your .env file (copied from .env.example)")
        print("2. Or pass it directly via CLI:")
        print('   python scripts/test_db_connection.py --url "postgresql://postgres:password@host:5432/postgres"')
        print("\nIf your password contains special characters:")
        print('   python scripts/test_db_connection.py --encode-password "MyP@ss#123"')
        sys.exit(1)

    # Run async test
    success = asyncio.run(test_connection(target_url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
