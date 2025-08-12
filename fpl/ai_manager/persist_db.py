# fpl/ai_manager/persist_db.py
from __future__ import annotations
import os, pathlib
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Integer, String, DateTime, JSON, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.sql import func
from config import DATABASE_URL, SEASON

# Ensure parent directory for sqlite file exists (if using SQLite locally)
if DATABASE_URL.startswith("sqlite:///"):
    path = DATABASE_URL.replace("sqlite:///", "", 1)
    if path not in (":memory:", ""):
        parent = os.path.dirname(path)
        if parent:
            pathlib.Path(parent).mkdir(parents=True, exist_ok=True)

# ✅ Neon-friendly engine: small pool + pre_ping
engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=0,
)

class Base(DeclarativeBase): pass

class SeasonState(Base):
    __tablename__ = "season_states"
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    season:  Mapped[str] = mapped_column(String, primary_key=True, default=SEASON)
    state:   Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class GwLog(Base):
    __tablename__ = "gw_logs"
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    season:  Mapped[str] = mapped_column(String, primary_key=True, default=SEASON)
    gw:      Mapped[int] = mapped_column(Integer, primary_key=True)
    entry:   Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

def init_db():
    Base.metadata.create_all(engine)

def load_state(user_id: str, season: str = SEASON) -> Optional[dict]:
    with Session(engine) as s:
        row = s.get(SeasonState, {"user_id": user_id, "season": season})
        return row.state if row else None

def save_state(user_id: str, state: dict, season: str = SEASON):
    with Session(engine) as s:
        row = s.get(SeasonState, {"user_id": user_id, "season": season})
        if row:
            row.state = state
        else:
            row = SeasonState(user_id=user_id, season=season, state=state)
        s.merge(row)
        s.commit()

def append_gw_log(user_id: str, gw: int, entry: dict, season: str = SEASON):
    with Session(engine) as s:
        s.merge(GwLog(user_id=user_id, season=season, gw=gw, entry=entry))
        s.commit()

def get_gw_logs(user_id: str, season: str = SEASON) -> list[dict]:
    with Session(engine) as s:
        rows = s.execute(
            select(GwLog).where(GwLog.user_id==user_id, GwLog.season==season).order_by(GwLog.gw.asc())
        ).scalars().all()
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
