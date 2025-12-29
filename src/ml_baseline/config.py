
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_processed_dir: Path



def from_repo_root(root: Path) -> Paths:
    data = root/ "data"
    return Paths(
        root=root,
        data_processed_dir=data/"processed",
    )
