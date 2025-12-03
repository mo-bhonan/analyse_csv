import os
import pandas as pd

def read_csv(path):
    return pd.read_csv(path)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_match_path(outdir_csv, mtgcsv, msgcsv):
    return f"{outdir_csv}/{mtgcsv.split('.csv')[0]}_{msgcsv.split('.csv')[0]}_matches_nn.csv"

def write_matches_csv(df, path):
    df.to_csv(path, index=False)

def clear_text(path):
    with open(path, "w") as f:
        pass
