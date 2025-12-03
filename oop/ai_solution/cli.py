import argparse
from .analysis import run_nearest_neighbor

def parse_args():
    p = argparse.ArgumentParser(description="MSG/MTG analysis")
    p.add_argument('--indir', type=str, default='/home/users/benjamin.honan/Work/analyse_csv/csv_files/')
    p.add_argument('--outdir', type=str, default='/home/users/benjamin.honan/Work/analyse_csv/plots/')
    p.add_argument('--master_csv_file', type=str, default='MSG_MTG_pairs.csv')
    p.add_argument('--recreate_csv', action='store_true')
    p.add_argument('--show_plots', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    run_nearest_neighbor(args.indir, args.outdir, args.master_csv_file, args.recreate_csv, args.show_plots)
