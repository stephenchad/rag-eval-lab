from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Reads from environment, falls back to .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAGLAB_",
        extra="ignore",
    )

    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"

    # Chunking
    chunk_target_tokens: int = 400
    chunk_min_tokens: int = 80
    chunk_max_tokens: int = 800

    # Logging
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()