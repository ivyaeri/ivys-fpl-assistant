# config.py
import os
import pytz

TZ = pytz.timezone("Europe/London")
MODEL_NAME = os.getenv("FPL_LLM_MODEL", "gpt-4o")

# Database:
#  - For Postgres set DATABASE_URL like:
#    postgresql+psycopg2://user:pass@host:5432/dbname
#  - Fallback to local sqlite file if not set.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/fpl.db")

# Season label (key in DB rows)
SEASON = os.getenv("FPL_SEASON", "2025-26")
