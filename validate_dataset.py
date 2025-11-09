from __future__ import annotations
import json, sys
import pandas as pd
import numpy as np
from config import settings

def fail(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)

def main(csv_path: str, schema_path: str):
    schema = json.load(open(schema_path, "r", encoding="utf-8"))

    df = pd.read_csv(csv_path, sep=schema.get("delimiter", ","))

    # Columns
    required = [c["name"] for c in schema["columns"] if c.get("required", True)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"Missing required columns: {missing}")

    # Types & NA
    for col in schema["columns"]:
        name = col["name"]
        if name not in df.columns:
            continue
        typ = col.get("type", "float")
        allow_na = col.get("allow_na", False)
        if typ in ("float","number"):
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif typ == "int":
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
        elif typ == "str":
            df[name] = df[name].astype(str)
        else:
            fail(f"Unsupported type in schema: {typ} for column {name}")

        if not allow_na and df[name].isna().any():
            n = int(df[name].isna().sum())
            fail(f"Column {name} has {n} NA values but allow_na=false")


    # Ranges
    for col in schema["columns"]:
        name = col["name"]
        if name not in df.columns:
            continue
        if "min" in col:
            bad = (df[name] < col["min"]).sum()
            if bad:
                fail(f"Column {name}: {bad} values < min {col['min']}")
        if "max" in col:
            bad = (df[name] > col["max"]).sum()
            if bad:
                fail(f"Column {name}: {bad} values > max {col['max']}")

    # Unique key (optional)
    uniq = schema.get("unique", [])
    if uniq:
        if df.duplicated(subset=uniq).any():
            ndup = int(df.duplicated(subset=uniq).sum())
            fail(f"Unique constraint failed for {uniq}: {ndup} duplicates")


    print("OK: dataset passed validation.")

if __name__ == "__main__":
    cfg = settings.section("validate_dataset")
    main(cfg.get("csv"), cfg.get("schema"))
