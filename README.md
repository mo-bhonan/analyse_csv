# analyse_csv

A toolkit for analysing, filtering, matching, and visualising paired satellite CSV datasets (MSG ↔ MTG). The package focuses on:

- Applying configurable constraints/masks to data columns.
- Classifying differences between MSG/MTG outcomes using nearest neighbour matches (KDTree).
- Plotting geographic distributions, histograms, beta masks, and frequency summaries.
- Producing comparison text files and matched CSV outputs for downstream inspection.

## Project Layout

```
analyse_csv.py            # CLI entry / examples to run plotting/analysis flows
config.py                 # Constants, enums, labels, operator maps, colormap helpers
dataset.py                # Dataset container (paths, metadata, dataframes, derived state)
filters.py                # Constraint parsing + boolean mask building utilities
flags.py                  # Flag and retrieval-code helpers (types/enums)
nn.py                     # KDTree helpers for matching points by lat/lon
plot_utils.py             # Shared plotting helpers and utilities
plotting.py               # Plotter class (maps, histograms, beta masks, frequency plots)

csv_files/                # Input CSVs and master pair lists
plots/                    # Output plots (generated)
```

## Data Expectations

Each per-satellite CSV is expected to contain at least:
- `Lat`, `Lon` (float): geographic coordinates (degrees)
- Retrieval/aux variables used in constraints/plots, e.g. `BTD2_conf`, `VolcanicAsh_BTD3`,
  `PreFilter_VA_Confidence`, `BMCon`, `BMLib`, and thresholds like `c1`, `c3`, `c4`, `BTD3thresh`.

A master CSV of MSG/MTG pairs contains at least:
- `msg_csv`, `mtg_csv` (filenames relative to `--indir`)
- Optional metadata columns (e.g. region hints) depending on your workflow

## Usage (Quick Start)

Run the main script to generate plots and analyses from a master CSV of MSG/MTG pairs:

```
python analyse_csv.py \
  --indir /home/users/benjamin.honan/Work/analyse_csv/csv_files \
  --outdir /home/users/benjamin.honan/Work/analyse_csv/plots \
  --master_csv_file MSG_MTG_pairs_ethiopia_0830_1200.csv \
  --show_plots
```

Command-line arguments (as used by `analyse_csv.py`):
- `--indir`: input directory for CSVs (default points to this repo’s `csv_files/`)
- `--outdir`: output directory for plots/exports (default `plots/`)
- `--master_csv_file`: the pairs table (e.g. `MSG_MTG_pairs_ethiopia_0830_1200.csv`)
- `--recreate_csv`: if set, regenerate intermediate outputs
- `--plotseparate`: if set, create separate MSG and MTG plots where supported
- `--show_plots`: if set, display plots in a window after saving

## Key Concepts

### Constraints / Masks
Use `filters.py` to build boolean masks from constraint dictionaries. Example:

```python
constraints = {
  'variables':  ["BTD2_conf", "BMCon"],
  'operators':  ['>=', '=='],
  'values':     [0.5, 'T']
}
```

- Supports combining multiple conditions (AND by default, OR by prefixing operator with `or_`).
- Supports referencing DataFrame columns by using values like `df_c4`.
- Format helpers create readable titles/filenames for plots.

### Nearest Neighbour Matching (KDTree)
`nn.py` centralises matching logic:
- Build KDTree once per dataset (`coords = list(zip(Lat, Lon))`).
- Vectorised query to find nearest neighbours (`k=1`) from search points.
- Returns indices/distances and optional matched rows.

### Retrieval Codes
`flags.py` / `config.py` define enums and friendly labels identifying why MSG/MTG differ (e.g., threshold failures, masks, both/none retrieved). Use these codes for map colouring or frequency plots.

## Plotting

All plotting is encapsulated in `plotting.py` via the `Plotter` class. Typical usage pattern:

```python
from plotting import Plotter
plotter = Plotter(indir, outdir, master_csv_file)
```

Available plots plots:
- Detection-type map scatter with legend and optional region box/marker
- Frequency bar chart of retrieval codes (with counts annotated, legend keyed by index)
- BTD histograms with threshold lines (C1/C3/C4), visible labels, and optional percentages
- Beta-mask visualisations and BTD3 maps (pcolormesh with zero-centred colormap)

## Outputs

- Matches CSV: nearest-neighbour MSG matches with `retrieval_code` column appended. This CSV file is created so that the computationally expensive NN matching doesn't have to be repeated.
- Plots: saved to `--outdir`, file names include region/time/cut where relevant.

## AI Disclaimer

README file produced with help of MS Copilot.
