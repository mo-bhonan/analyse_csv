import pandas as pd
from collections import Counter
from .io import read_csv, ensure_dir, build_match_path, write_matches_csv
from .filters import apply_constraints, format_constraints_for_title
from .nn import match_rows
from .retrieval import flags_from_pair, pick_code
from .plotting import Plotter
from .config import DICT_CUT_CONSTRAINT, RETRIEVAL_CODE_LABELS, RetrievalCode, select_cmap

def run_nearest_neighbor(indir, outdir, master_csv_file, recreate_csv=False, show_plots=False):
    df_master = read_csv(f"{indir}/{master_csv_file}")
    ensure_dir(outdir)
    outdir_csv = f"{outdir}/matches"
    ensure_dir(outdir_csv)
    for mtgcsv, msgcsv, region, latlon, conf_cut in zip(df_master['mtg_csv'], df_master['msg_csv'], df_master['region'], df_master['latlon'], df_master['conf_cut']):
        f_output_csv = build_match_path(outdir_csv, mtgcsv, msgcsv)
        df_mtg = read_csv(f"{indir}/{mtgcsv}")
        df_msg = read_csv(f"{indir}/{msgcsv}")
        # optional: apply cut by conf_cut if desired
        pairs = []
        # build NN pairs on full domain or prefiltered by region
        rows, dist = match_rows(df_mtg, df_msg, k=1)  # mtg nearest to msg
        # convert to codes
        codes = []
        for mtg_row, msg_row in zip(rows, df_msg.itertuples(index=False)):
            flags = flags_from_pair(mtg_row, msg_row)
            codes.append(pick_code(flags))
            pairs.append((mtg_row, msg_row))
        # write matches
        df_msg_matches = pd.DataFrame([p[1] for p in pairs])
        df_msg_matches['retrieval_code'] = pd.Series(codes)
        write_matches_csv(df_msg_matches, f_output_csv)
        # plotting
        plotter = Plotter(indir, outdir, show_plots=show_plots, region=region, time=msgcsv.split("_")[1], constraint_cut=conf_cut)
        # detection map
        unique_codes = df_msg_matches['retrieval_code'].unique()
        cmap = select_cmap(len(unique_codes))
        labels = []
        for i, code in enumerate(unique_codes):
            subset = df_msg_matches[df_msg_matches['retrieval_code'] == code]
            label = RETRIEVAL_CODE_LABELS.get(code if isinstance(code, RetrievalCode) else RetrievalCode(code.split('.')[-1].lower()), str(code))
            labels.append((label, subset, cmap(i)))
        lon_range = tuple(map(float, latlon.split("_")[2:4]))
        lat_range = tuple(map(float, latlon.split("_")[0:2]))
        plotter.detection_map(df_msg_matches, labels, lon_range, lat_range, "MSG/MTG Detection Type Comparison")
        # histogram
        code_strings = [RETRIEVAL_CODE_LABELS.get(c, str(c)) for c in codes]
        counts = Counter(code_strings)
        labels_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        meta_text = f"Region: {region}\nCut: {conf_cut}\nTime: {plotter.time}"
        plotter.freq_hist(labels_counts, meta_text)
