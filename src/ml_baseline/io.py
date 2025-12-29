def write_tabular(df, path):
    df.to_csv(path, index=False)
    print(f"Data successfully written to {path}")

def parquet_supported(df, path):
    try:
            # Attempt to write to a temporary or the target path
            df.to_parquet(path)
            return True
    except Exception:
            return False

