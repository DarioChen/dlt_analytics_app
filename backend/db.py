
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime
import os
from sqlalchemy import create_engine, Column, Integer, String, Date, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dlt.sqlite")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()

class Draw(Base):
    __tablename__ = "draws"
    id = Column(Integer, primary_key=True, autoincrement=True)
    issue = Column(String, nullable=False)  # 期号，如 25101
    date = Column(Date, nullable=False)     # 开奖日期
    f1 = Column(Integer); f2 = Column(Integer); f3 = Column(Integer); f4 = Column(Integer); f5 = Column(Integer)  # 前区
    b1 = Column(Integer); b2 = Column(Integer)  # 后区
    sales = Column(String, nullable=True)       # 当期销量（字符串保存以避免千分位/单位差异）
    pool = Column(String, nullable=True)        # 奖池金额（字符串）

    __table_args__ = (UniqueConstraint('issue', name='uq_issue'),)

    def front(self) -> Tuple[int,int,int,int,int]:
        return (self.f1,self.f2,self.f3,self.f4,self.f5)

    def back(self) -> Tuple[int,int]:
        return (self.b1,self.b2)

def init_db():
    Base.metadata.create_all(engine)

@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
