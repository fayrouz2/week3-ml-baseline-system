def write_tabular(df, path):
    df.to_csv(path, index=False)
    print(f"Data successfully written to {path}")

def parquet_supported(df, path):
    try:
        df.to_parquet(path/"features.parquet")
        return True
    except Exception:
        return False

