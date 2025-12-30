import typer
from src.ml_baseline.sample_data import make_sample_feature_table
from src.ml_baseline.train import run_train
from .config import TrainCfg, Paths
from pathlib import Path
import pandas as pd



app = typer.Typer()
paths=Paths(Path(__file__).resolve().parents[2])
root = paths.root

@app.command()
def make_sample_data(root: Path = root):
    """
    Generate Sample to data\processed.
    """
    make_sample_feature_table(root = root)
    print("Data Sample Generated. ")

@app.command()
def train(target: str):
    """
    Training Sample.
    """
    tcfg = TrainCfg(
        features_path=paths.data_processed_dir / "features.parquet",
        target=target,
    )

    run_train(tcfg, root=root)
    
    print("Train. ")

if __name__ == "__main__":
    app()
