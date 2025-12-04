import pandas as pd
import numpy as np
import gc
from pathlib import Path

class Dataset:
    """
    Usage:
        for dataset in plotter.iter_loaded():
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            region = meta['region']
            # ... analysis ...
    """

    def __init__(self, metadata):
        # Copy metadata to avoid mutating external dicts
        self.metadata = dict(metadata)

        self.file_path_msg = io.resolve_path(self.metadata.get("csv_msg"))
        self.file_path_mtg = io.resolve_path(self.metadata.get("csv_mtg"))

        # Derive times safely
        self.metadata['time_msg'] = self.derive_time_from_path(self.file_path_msg)
        self.metadata['time_mtg'] = self.derive_time_from_path(self.file_path_mtg)

        latlonlist = self.metadata['latlon'].split("_")
        self.lon_range = (float(latlonlist[2]),float(latlonlist[3]))
        self.lat_range = (float(latlonlist[0]),float(latlonlist[1]))

        # Data holders (lazy-loaded)
        self.data_msg = None
        self.data_mtg = None

        # Lat/Lon numpy arrays
        self.lons_msg = None
        self.lons_mtg = None
        self.lats_msg = None
        self.lats_mtg = None

        # Optional derived stats (set during load)
        self.lonmin_msg = None
        self.lonmax_msg = None
        self.latmin_msg = None
        self.latmax_msg = None

        self.lonmin_mtg = None
        self.lonmax_mtg = None
        self.latmin_mtg = None
        self.latmax_mtg = None

        self.n_msg = None
        self.n_mtg = None

    def _load(self):
        self.data_msg = pd.read_csv(self.file_path_msg)
        self.data_mtg = pd.read_csv(self.file_path_mtg)

        # Validate expected columns exist
        for dfname, df in (("msg", self.data_msg), ("mtg", self.data_mtg)):
            for col in ("Lon", "Lat"):
                if col not in df.columns:
                    raise KeyError(f"Expected column '{col}' in {dfname} CSV: {df.columns.tolist()}")

        # Convert Lon/Lat columns to NumPy arrays for fast access
        self.lons_msg = self.data_msg["Lon"].to_numpy()
        self.lats_msg = self.data_msg["Lat"].to_numpy()
        self.lons_mtg = self.data_mtg["Lon"].to_numpy()
        self.lats_mtg = self.data_mtg["Lat"].to_numpy()

        # Compute min/max directly from DataFrames to avoid storing arrays
        self.lonmin_msg = self.lons_msg.min()
        self.lonmax_msg = self.lons_msg.max()
        self.latmin_msg = self.lats_msg.min()
        self.latmax_msg = self.lats_msg.max()

        self.lonmin_mtg = self.lons_mtg.min()
        self.lonmax_mtg = self.lons_mtg.max()
        self.latmin_mtg = self.lats_mtg.min()
        self.latmax_mtg = self.lats_mtg.max()

        self.n_msg = len(self.data_msg)
        self.n_mtg = len(self.data_mtg)

    def _unload(self):
        # Release memory by clearing references
        self.data_msg = None
        self.data_mtg = None

        # Release NumPy arrays
        self.lons_msg = None
        self.lats_msg = None
        self.lons_mtg = None
        self.lats_mtg = None

        # Release derived stats. Not really necessary, but for completeness
        self.lonmin_msg = None
        self.lonmax_msg = None
        self.latmin_msg = None
        self.latmax_msg = None
        self.lonmin_mtg = None
        self.lonmax_mtg = None
        self.latmin_mtg = None
        self.latmax_mtg = None
        self.n_msg = None
        self.n_mtg = None

        # Optionally call gc.collect() under heavy memory pressure
        # gc.collect()

    def _derive_time_from_path(self, path: Path):
        # Safer than split("_")[1]; customize to your naming convention
        # Example: "prefix_20250101_suffix.csv" -> "20250101"
        parts = path.stem.split("_")
        return parts[1] if len(parts) > 1 else None


