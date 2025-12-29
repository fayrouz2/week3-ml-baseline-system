import typer
from src.ml_baseline.sample_data import make_sample_feature_table
from pathlib import Path

app = typer.Typer()

@app.command()
def make_sample_data(root: Path = Path(__file__).resolve().parents[2]):
    """
    Generate Sample to data\processed.
    """
    
    make_sample_feature_table(root = root)
    print("Data Sample Generated. ")

if __name__ == "__main__":
    app()
