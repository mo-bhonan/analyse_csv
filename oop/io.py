from pathlib import Path

def derive_time_from_path(self, path: Path):
    # Safer than split("_")[1]; customize to your naming convention
    # Example: "prefix_20250101_suffix.csv" -> "20250101"
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 1 else None

def resolve_path(self, p):
    p = Path(p)
    return p if p.is_absolute() else (self.indir / p)

