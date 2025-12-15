import argparse
#from .analysis import run_nearest_neighbor
from plotting import Plotter

def parse_args():
    p = argparse.ArgumentParser(description="MSG/MTG analysis")
    p.add_argument('--indir', type=str, default='/home/users/benjamin.honan/Work/analyse_csv/oop/csv_files/')
    p.add_argument('--outdir', type=str, default='/home/users/benjamin.honan/Work/analyse_csv/oop/plots/')
    p.add_argument('--master_csv_file', type=str, default='MSG_MTG_pairs_ethiopia_1100_1200.csv')
    #p.add_argument('--master_csv_file', type=str, default='MSG_MTG_pairs_ethiopia_0910_1200.csv')
    #p.add_argument('--master_csv_file', type=str, default='MSG_MTG_pairs_ethiopia_0830_1200.csv')
    #p.add_argument('--master_csv_file', type=str, default='MSG_MTG_pairs_lakes_kintyre_20251012022.csv')
    p.add_argument('--recreate_csv', action='store_true')
    p.add_argument('--show_plots', action='store_true')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    myplotter = Plotter(args.indir, args.outdir, args.master_csv_file, show_plots=args.show_plots)
    #myplotter.plot_nearestneighbors(args.recreate_csv)
    #myplotter.make_btd_plots()
    myplotter.plot_beta_masks()
