import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle, Patch
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
from pathlib import Path
import config
import filters
import gc
import nn
from dataset import Dataset

class Plotter:
    def __init__(self, indir, outdir, master_csv):
        self.indir = Path(indir)
        self.outdir = Path(outdir)
        self.matches_dir = self.indir / "matches" 
        self.matches_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(master_csv)
        self.datasets = [Dataset(row.to_dict()) for _, row in df.iterrows()]

    def resolve_path(self, path):
        p = Path(p)
        return p if p.is_absolute() else (self.indir / path)

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

    def setup_figure(self, title=None, xlim=None, ylim=None, legendtitle=None, plotstr=None):
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.legend(title=legendtitle, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        ax.coastlines()
        plt.title(title)

        plt.text(xlim[0] + 0.02*(xlim[1]-xlim[0]), ylim[1] - (ylim[1]-ylim[0])*0.02, plotstr, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False)
        gl.xlines = True   
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 14} 
        gl.ylabel_style = {'size': 14}    
        return ax

    #def plot_nearestneighbors(self, recreate_csv, write_output_matches=True, show_plots=False):
    def get_matches_and_codes(indir, file_msg, file_mtg, write_output_matches, f_output_csv, output_txt=False, threshold=0.01):

        for dataset in plotter.iter_loaded():
            # df_msg, df_mtg will go out of scope at the end of each iteration, so there won't be a memory leak.
            # They point to the same dataframe as dataset.data_msg and dataset.data_mtg
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            cutstr = config.getcutstr(meta['conf_cut'])
            # if not reading the nn csv from file
            if not os.path.exists(f_output_csv) or recreate_csv:
                f_output_csv = f"{self.matches_dir}/{dataset.filepath_msg.split('.csv')[0]}_{dataset.filepath_mtg.split('.csv')[0]}_matches_nn.csv"

                retrievalcodes = []
                msg_matches = []
                mtg_matches = []

                # Apply cut
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

            # if reading csv from file
            else:
                df_msg_matches = pd.read_csv(f_output_csv)

            codes = df_msg_matches['retrieval_code']
            unique_codes = df_msg_matches['retrieval_code'].unique()
            if len(unique_codes) < 9:
                colors = plt.get_cmap('Dark2', len(unique_codes))
            elif len(unique_codes) < 11:
                colors = plt.get_cmap('tab10', len(unique_codes))
            elif len(unique_codes) < 13:
                colors = plt.get_cmap('Paired', len(unique_codes))
            else:
                colors = plt.get_cmap('tab20', len(unique_codes))

            self.setup_figure(title="MSG/MTG Detection Type Comparison", 
                              xlim=dataset.lon_range,
                              ylim=dataset.lat_range,
                              legendtitle='Detection Type',
                              plotstr=f"Region: {meta['region']}\n"+f"{cutstr}\n"+f"Time: {meta['time_msg']}"
                              )

            for i, code in enumerate(unique_codes):
                code_subset = df_msg_matches[df_msg_matches['retrieval_code'] == code]
                if code not in config.codes_to_ignore:
                    plt.scatter(
                        code_subset['Lon'], code_subset['Lat'],
                        s=2,
                        label=retrieval_code_labels.get(code),
                        color=colors(i),
                        alpha=0.8
                    )

            # Save the plot before showing
            outname = f"{region.replace(" ","_")}_{timestr}_detection_type_map.png"
            plot_path = outdir + "/" + outname
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            if savesvgpdf:
                plt.savefig(plot_path.replace("png","svg"), dpi=300, bbox_inches='tight')
                plt.savefig(plot_path.replace("png","pdf"), dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
            if show_plots:
                plt.show()
            plt.close()

            # Convert codes to strings if they are enums
            code_strings = [retrieval_code_labels.get(RetrievalCode(code.split('.')[-1].lower()), str(code)) if read_from_file else retrieval_code_labels.get(code, str(code)) for code in codes.values]

            # Count occurrences of each code
            code_counts = Counter(code_strings)

            # Sort by frequency (descending)
            sorted_items = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0] for item in sorted_items]
            counts = [item[1] for item in sorted_items]

            # Assign integer indices to each label
            indices = list(range(1, len(labels)+1))

            plt.figure()
            bars = plt.bar(indices, counts)
            plt.xticks(indices)  # x-axis ticks are integers

            #TODO: Reconstructing dfs for this not very nice. Ideally would combine all common things amongst plotting functions into a plotting object instance.
            msgpath = indir+'/'+msgcsv
            df_msg = pd.read_csv(msgpath)
            lons_msg = np.array(df_msg["Lon"])
            lats_msg = np.array(df_msg["Lat"])
            lon_min = np.min(lons_msg, initial=100.)
            lon_max = np.max(lons_msg, initial=-100.)
            lat_min = np.min(lats_msg, initial=100.)
            lat_max = np.max(lats_msg, initial=-100.)
            latstr = '('+str(round(lat_min,1))+','+str(round(lat_max,1))+')'
            lonstr = '('+str(round(lon_min,1))+','+str(round(lon_max,1))+')'
            latlonstr=f"Lat/Lon: {latstr}/{lonstr}"

            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()

            plt.ylabel("Count")
            plt.xlabel("Detection Type")
            plt.title("Detection Type Frequency")
            plt.tight_layout()
            ylim = plt.gca().get_ylim()
            plt.ylim(ylim[0], ylim[1] * 1.30)

            # Annotate each bar with its count value
            for bar, count in zip(bars, counts):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(count),
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            # Create legend mapping integers to labels, matching bar colors
            legend_handles = [
                Patch(facecolor=bar.get_facecolor(), label=f"{i}: {label}")
                for i, label, bar in zip(indices, labels, bars)
            ]
            plt.legend(handles=legend_handles, title="Detection Type", fontsize='small', loc='upper right')

            # Get the bounding box of the legend in display coordinates
            bbox = plt.gca().get_legend().get_window_extent()
            # Transform to axes coordinates
            ax = plt.gca()
            inv = ax.transAxes.inverted()
            bbox_ax = inv.transform(bbox)
            # Place the text just below the legend
            x_text = bbox_ax[0][0]  # left of legend
            y_text = bbox_ax[0][1] - 0.05  # slightly below legend
            plt.text(
            x_text, y_text,
            f"{latlonstr}\nRegion: {region}\n{cut_str}\nTime: {timestr}",
            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            transform=ax.transAxes
            )

            outname = f"{region.replace(" ","_")}_{timestr}_detection_type_histogram.png"
            plot_path = outdir + "/" + outname
            print(f"Plot saved to: {plot_path}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            if savesvgpdf:
                plt.savefig(plot_path.replace("png","svg"), bbox_inches='tight')
                plt.savefig(plot_path.replace("png","pdf"), bbox_inches='tight')
            if show_plots:
                plt.show()
            plt.close()


