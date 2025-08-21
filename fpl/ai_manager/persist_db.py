# fpl/ai_manager/persist_db.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Integer, String, DateTime, JSON, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.sql import func

# ---- Resolve DATABASE_URL, defaulting to repo-local SQLite ----
try:
    # Prefer your config if present
    from config import DATABASE_URL as CONFIG_DB_URL, SEASON
except Exception:
    CONFIG_DB_URL = None
    SEASON = os.getenv("SEASON", "2025-26")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = REPO_ROOT / "data" / "fpl_local.db"
DEFAULT_SQLITE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
print(DEFAULT_SQLITE_PATH,DEFAULT_SQLITE_URL )
DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or CONFIG_DB_URL
    or DEFAULT_SQLITE_URL
)

IS_SQLITE = DATABASE_URL.startswith("sqlite:///")

# Ensure parent directory exists (for local SQLite)
if IS_SQLITE:
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- Engine config: SQLite (local) vs Neon/Postgres ----
if IS_SQLITE:
    # Streamlit-friendly sqlite: allow cross-thread use of the same connection
    print(DATABASE_URL)
    engine = create_engine(
        DATABASE_URL,
        future=True,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
    )
# else:
#     # Neon/Postgres-friendly small pool
#     engine = create_engine(
#         DATABASE_URL,
#         future=True,
#         pool_pre_ping=True,
#         pool_size=5,
#         max_overflow=0,
#     )

class Base(DeclarativeBase):
    pass

class SeasonState(Base):
    __tablename__ = "season_states"
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    season:  Mapped[str] = mapped_column(String, primary_key=True, default=SEASON)
    state:   Mapped[dict] = mapped_column(JSON, nullable=False)
    # SQLite doesn't have TZ; func.now() is fine for both backends
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True),
                                            server_default=func.now(),
                                            onupdate=func.now())

class GwLog(Base):
    __tablename__ = "gw_logs"
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    season:  Mapped[str] = mapped_column(String, primary_key=True, default=SEASON)
    gw:      Mapped[int] = mapped_column(Integer, primary_key=True)
    entry:   Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(engine)

def load_state(user_id: str, season: str = SEASON) -> Optional[dict]:
    with Session(engine) as s:
        row = s.get(SeasonState, {"user_id": user_id, "season": season})
        return row.state if row else None

def save_state(user_id: str, state: dict, season: str = SEASON) -> None:
    with Session(engine) as s:
        row = s.get(SeasonState, {"user_id": user_id, "season": season})
        if row:
            row.state = state
        else:
            row = SeasonState(user_id=user_id, season=season, state=state)
        s.merge(row)
        s.commit()

def append_gw_log(user_id: str, gw: int, entry: dict, season: str = SEASON) -> None:
    with Session(engine) as s:
        s.merge(GwLog(user_id=user_id, season=season, gw=gw, entry=entry))
        s.commit()

def get_gw_logs(user_id: str, season: str = SEASON) -> list[dict]:
    with Session(engine) as s:
        rows = (
            s.execute(
                select(GwLog)
                .where(GwLog.user_id == user_id, GwLog.season == season)
                .order_by(GwLog.gw.asc())
            )
            .scalars()
            .all()
        )
        return [r.entry for r in rows]

# Optional utilities (handy in admin tab)
def list_users() -> List[str]:
    with Session(engine) as s:
        users = s.execute(select(SeasonState.user_id).distinct()).scalars().all()
    return users

def raw_query(sql: str) -> list[Dict[str, Any]]:
    with engine.connect() as conn:
        res = conn.exec_driver_sql(sql)
        cols = res.keys()
        return [dict(zip(cols, row)) for row in res.fetchall()]
