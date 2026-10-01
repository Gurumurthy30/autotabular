from sqlmodel import Session, SQLModel, create_engine

from app.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)


def init_db() -> None:
    """Creates all tables if they do not exist."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI dependency for database session."""
    with Session(engine) as session:
        yield session
