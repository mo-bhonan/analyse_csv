import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle, Patch
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import config
import filters
import gc
import io
import nn
from dataset import Dataset

class Plotter:
    def __init__(self, indir, outdir, master_csv):
        self.indir = Path(indir)
        self.outdir = Path(outdir)
        self.outdir_matches = self.indir / "matches" 
        self.outdir_matches.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(master_csv)
        self.datasets = [Dataset(row.to_dict()) for _, row in df.iterrows()]

    def iter_loaded(self):
        """
        Generator that yields each Dataset after loading.
        Automatically unloads the previous dataset when moving to the next
        (or if the caller raises/breaks).
        """
        for ds in self.datasets:
            ds._load()
            try:
                yield ds
            finally:
                ds._unload()

    def new_geo_axes(self):
        ax = plt.axes(projection=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
        gl.xlines = True
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        ax.coastlines()
        return ax

    def get_matches_and_codes(indir, file_msg, file_mtg, write_output_matches, f_output_csv, output_txt=False, threshold=0.01):

        for dataset in plotter.iter_loaded():
            # df_msg, df_mtg will go out of scope at the end of each iteration, so there won't be a memory leak.
            # They point to the same dataframe as dataset.data_msg and dataset.data_mtg
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            cutstr = config.getcutstr(meta['conf_cut'])

            retrievalcodes = []
            msg_matches = []
            mtg_matches = []

            # Apply cut
            cutstr = config.getcutstr(meta['conf_cut'])
            if cutstr:
                sat, variable, operator, value = (cut_str.split(" ")[i] for i in range(4))
                constraints = {"variables" : [variable], "operator" : [operator], "values": [value]}
                if sat.lower() == "msg":
                    df_msg = filters.apply_constraints(df_msg, constraints)
                else:
                    df_mtg = filters.apply_constraints(df_mtg, constraints)
                    
            search_matches = nn.find_nn(dataset, threshold=0.01)
            for msg_match, mtg_match in search_matches:
                flags_msg = retrieval.flags_from_series(msg_match)
                flags_mtg = retrieval.flags_from_series(mtg_match)
                retrievalcode = retrieval.pick_code(flags_msg, flags_mtg)
                retrievalcodes.append(retrievalcode)
                
            df_msg_matches = pd.DataFrame(msg_matches).reset_index(drop=True)
            if len(retrievalcodes) != len(df_msg_matches):
                raise ValueError("Length of retrievalcodes list not equal to list of nearest-neighbour MSG matches")
            df_msg_matches['retrieval_code'] = pd.Series(retrievalcodes)

            if not df_msg_matches.empty and write_output_matches:
                df_mtg_matches = pd.DataFrame(mtg_matches).reset_index(drop=True)
                df_msg_matches['MTG_BTD2_conf'] = df_mtg_matches['BTD2_conf']
                df_msg_matches['MTG_PreFilter_VA_Confidence'] = df_mtg_matches['PreFilter_VA_Confidence']
                df_msg_matches['MTG_Median_VA_Confidence'] = df_mtg_matches['Median_VA_Confidence']
                df_msg_matches.to_csv(f_output_csv, index=False)
                print(f"Nearest-neighbour MSG matches written to {f_output_csv}")

        return (msg_matches, mtg_matches, retrievalcodes)


    def plot_nearestneighbors(self, recreate_csv, write_output_matches=True, show_plots=False):

        #for mtgcsv, msgcsv, region, latlon, conf_cut in zip(df_master['mtg_csv'], df_master['msg_csv'], df_master['region'], df_master['latlon'], df_master['conf_cut']):
        for dataset in self.datasets:
            df_msg, df_mtg = dataset.load_data()
            meta = dataset.metadata
            f_output_csv = f"{self.outdir_matches}/{dataset.filepath_msg.split('.csv')[0]}_{dataset.filepath_mtg.split('.csv')[0]}_matches_nn.csv"
            cutstr = config.getcutstr(meta['conf_cut'])

            if not os.path.exists(f_output_csv) or recreate_csv:
                #TODO: In future can have a dictionary mapping cuts to regions
                msg_matches, mtg_matches, retrievalcodes = get_matches_and_codes(indir, msgcsv, mtgcsv, write_output_matches, f_output_csv, conf_cut=conf_cut)

                df_msg_matches = pd.DataFrame(msg_matches).reset_index(drop=True)
                if len(retrievalcodes) != len(df_msg_matches):
                    raise ValueError("Length of retrievalcodes list not equal to list of nearest-neighbour MSG matches")
                df_msg_matches['retrieval_code'] = pd.Series(retrievalcodes)
                read_from_file=False
            else:
                df_msg_matches = pd.read_csv(f_output_csv)
                read_from_file=True

            plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())

            codes = df_msg_matches['retrieval_code']
            unique_codes = df_msg_matches['retrieval_code'].unique()
            # Work-around for reading from file because can't read enums from a file...
            _codes_to_ignore = [str(_code) for _code in codes_to_ignore] if read_from_file else codes_to_ignore
            if len(unique_codes) < 9:
                colors = plt.get_cmap('Dark2', len(unique_codes))
            elif len(unique_codes) < 11:
                colors = plt.get_cmap('tab10', len(unique_codes))
            elif len(unique_codes) < 13:
