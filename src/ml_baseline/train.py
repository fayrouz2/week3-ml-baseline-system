import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .config import TrainCfg

from datetime import datetime
from datetime import timezone
from pathlib import Path
import os
import json
import hashlib
import sys
import platform
import subprocess
import joblib

import logging
log = logging.getLogger(__name__)


def fit_classifier(X: pd.DataFrame, y: pd.Series):
    num = X.select_dtypes(include=["number"]).columns
    cat = X.select_dtypes(exclude=["number"]).columns

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat),
        ]
    )

    model = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X, y)   
    return model


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pip_freeze() -> str:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        return f"# pip freeze failed: {e!r}\n" 


def run_train(cfg: TrainCfg, *, root: Path, run_tag: str = "clf") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    run_id = f"{ts}_{run_tag}_session{cfg.session_id}"
    run_dir = root / "models" / "runs" / run_id

    for d in ["metrics", "plots", "tables", "schema", "env", "model"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("Run dir: %s", run_dir)

    df = pd.read_parquet(cfg.features_path)
    assert cfg.target in df.columns, f"Missing target: {cfg.target}"
    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)

    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=cfg.session_id
    )


    model = fit_classifier(X_train, y_train)
    joblib.dump(model, run_dir / "model" / "model.joblib")


    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }



    # Optional: enforce time sorting for time-based evaluation
    if cfg.time_col:
        assert cfg.time_col in df.columns, f"Missing time_col: {cfg.time_col}"
        df = df.sort_values(cfg.time_col).reset_index(drop=True)

    # Build schema contract: required features vs optional IDs
    feature_cols = [c for c in df.columns if c not in {cfg.target, *cfg.id_cols}]
    schema = {
        "target": cfg.target,
        "required_feature_columns": feature_cols,
        "optional_id_columns": [c for c in cfg.id_cols if c in df.columns],
        "feature_dtypes": {c: str(df[c].dtype) for c in feature_cols},
        "datetime_columns": [c for c in feature_cols if "datetime" in str(df[c].dtype).lower()],
        "policy_unknown_categories": "tolerant (OneHotEncoder handle_unknown=ignore)",
        "forbidden_columns": [cfg.target],
    }
    (run_dir / "schema" / "input_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Environment capture
    (run_dir / "env" / "pip_freeze.txt").write_text(_pip_freeze(), encoding="utf-8")
    env_meta = {
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "platform": platform.platform(),
    }
    (run_dir / "env" / "env_meta.json").write_text(json.dumps(env_meta, indent=2), encoding="utf-8")
    
    # Keep PyCaret outputs inside run_dir
    cwd = Path.cwd()
    os.chdir(run_dir)

    registry_dir = root / "models" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    
    metrics_file = run_dir / "metrics" / "baseline_holdout.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))


    latest_file = registry_dir / "latest.txt"
    latest_file.write_text(
        json.dumps({
            "model": str(run_dir / "model" / "model.joblib"),
            "metrics": str(run_dir / "metrics" / "baseline_holdout.json")
        }, indent=2),
        encoding="utf-8"
    )

    log.info("Updated latest model path: %s", latest_file)




