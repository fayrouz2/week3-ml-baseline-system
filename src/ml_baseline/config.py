
from dataclasses import dataclass
from pathlib import Path

class Paths:
    def __init__(self, root: Path):
        self.root = root

    @property
    def data_processed_dir(self) -> Path:
        return self.root / "data" / "processed"


@dataclass(frozen=True)
class TrainCfg:
    features_path: Path
    target: str
    id_cols: tuple[str, ...] = ("id",)
    time_col: str | None = None
    session_id: int = 42
