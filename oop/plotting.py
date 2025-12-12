import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle, Patch
import pandas as pd
from pathlib import Path
import config
import filters
import retrieval
import gc
import nn
import plot_utils
from dataset import Dataset
from collections import Counter
import io

class Plotter:
    def __init__(self, indir, outdir, master_csv, show_plots=False):
        self.indir = Path(indir)
        self.outdir = Path(outdir)
        self.show_plots = show_plots

        df = pd.read_csv(self.indir / Path(master_csv))
        self.datasets = [Dataset(self.indir, row.to_dict()) for _, row in df.iterrows()]

    def plot_nearestneighbors(self, recreate_csv, plot_histograms=True):
        for dataset in plot_utils.iter_loaded(self.datasets):
            # df_msg, df_mtg will go out of scope at the end of each iteration, so there won't be a memory leak.
            # They point to the same dataframe as dataset.data_msg and dataset.data_mtg
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            cutstr = config.getcutstr(meta['conf_cut'])
            # if not reading the nn csv from file
            matches_dir = self.indir / Path("matches")
            matches_dir.mkdir(parents=True, exist_ok=True)
            path_output_csv = matches_dir / Path(f"{dataset.filepath_msg.stem}_{dataset.filepath_mtg.stem}_matches_nn.csv")
            if not path_output_csv.exists() or recreate_csv:

                retrievalcodes = []
                msg_matches = []
                mtg_matches = []

                # Apply cut
                if cutstr:
                    sat, variable, operator, value = (cutstr.split(" ")[i] for i in range(4))
                    constraints = {"variables" : [variable], "operators" : [operator], "values": [float(value)]}
                    if sat.lower() == "msg":
                        df_msg = filters.apply_constraints(df_msg, constraints)
                    else:
                        df_mtg = filters.apply_constraints(df_mtg, constraints)
                        
                search_matches = nn.find_nn(dataset, threshold=0.01)
                for msg_match, mtg_match in search_matches:
                    msg_matches.append(msg_match)
                    mtg_matches.append(mtg_match)
                    flags_msg = retrieval.flags_from_series(msg_match)
                    flags_mtg = retrieval.flags_from_series(mtg_match)
                    retrievalcode = retrieval.pick_code(flags_msg, flags_mtg)
                    retrievalcodes.append(retrievalcode)
                    
                df_msg_matches = pd.DataFrame(msg_matches).reset_index(drop=True)
                if len(retrievalcodes) != len(df_msg_matches):
                    raise ValueError("Length of retrievalcodes list not equal to list of nearest-neighbour MSG matches")
                df_msg_matches['retrieval_code'] = pd.Series(retrievalcodes)

                if not df_msg_matches.empty:
                    df_mtg_matches = pd.DataFrame(mtg_matches).reset_index(drop=True)
                    df_msg_matches['MTG_BTD2_conf'] = df_mtg_matches['BTD2_conf']
                    df_msg_matches['MTG_PreFilter_VA_Confidence'] = df_mtg_matches['PreFilter_VA_Confidence']
                    df_msg_matches['MTG_Median_VA_Confidence'] = df_mtg_matches['Median_VA_Confidence']
                    df_msg_matches.to_csv(path_output_csv, index=False)
                    print(f"Nearest-neighbour MSG matches written to {path_output_csv}")

            # if reading csv from file
            else:
                df_msg_matches = pd.read_csv(path_output_csv)

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

            plot_utils.setup_figure(title="MSG/MTG Detection Type Comparison", 
                              xlim=dataset.lon_range,
                              ylim=dataset.lat_range,
                              plotstr=f"Region: {meta['region']}\n"+f"{cutstr}\n"+f"Time: {meta['time_msg']}"
                              )

            for i, code in enumerate(unique_codes):
                code_subset = df_msg_matches[df_msg_matches['retrieval_code'] == code]
                if code not in config.codes_to_ignore:
                    plt.scatter(
                        code_subset['Lon'], code_subset['Lat'],
                        s=2,
                        label=config.RETRIEVAL_CODE_LABELS[code],
                        color=colors(i),
                        alpha=0.8
                    )

            plt.legend(title='Detection Type', fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
            # Save the plot before showing
            outname = f"{meta['region'].replace(" ","_")}_{meta['time_msg']}_detection_type_map"
            plot_utils.save_plots(self.outdir, outname)
            if self.show_plots:
                plt.show()
            plt.close()

            if plot_histograms:
                # Count occurrences of each code
                code_strings = [config.RETRIEVAL_CODE_LABELS[code] for code in codes.values if code not in config.codes_to_ignore]
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
                f"{dataset.latlonstr}\nRegion: {meta['region']}\n{cutstr}\nTime: {meta['time_msg']}",
                ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                transform=ax.transAxes
                )

                outname = f"{meta['region'].replace(" ","_")}_{meta['time_msg']}_detection_type_histogram"
                plot_utils.save_plots(self.outdir, outname)
                if self.show_plots:
                    plt.show()
                plt.close()

    def make_btd_plots(self):
        for dataset in plot_utils.iter_loaded(self.datasets):
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            cutstr = config.getcutstr(meta['conf_cut'])

            # Restrict MTG dataframe to the MSG lon/lat
            df_mtg = df_mtg[
                (df_mtg['Lat'] > dataset.latmin_msg) & (df_mtg['Lat'] < dataset.latmax_msg) &
                (df_mtg['Lon'] > dataset.lonmin_msg) & (df_mtg['Lon'] < dataset.lonmax_msg)
            ]

            mtg_btd2 = df_mtg["BTD2_conf"].values
            msg_btd2 = df_msg["BTD2_conf"].values
            mtg_btd3 = df_mtg["VolcanicAsh_BTD3"].values
            msg_btd3 = df_msg["VolcanicAsh_BTD3"].values

            # Plot BTD2 histogram
            fig, ax = plot_utils.plot_btd_hist(
                        [mtg_btd2, msg_btd2],
                        xlabel="BTD2",
                        ylabel="Probability Density",
                        title="BTD2 values",
                        xmin = 0,
                        plotc4=meta["plotc4"],
                        plotc3=meta["plotc3"],
                        plotc1=meta["plotc1"],
                        outname=f"BTD2_{meta['region'].replace(" ","_")}_{meta['time_msg']}.png",
                        latlonstr=dataset.latlonstr,
                        regionstr=f"Plot Region: {meta['region']}",
                        timestr=f"Time: {meta['time_msg']}",
                        outdir=self.outdir,
                        conf_cut=meta["conf_cut"],
                    )

            if self.show_plots:
                plt.show()
            plt.close()

            # Plot BTD3 histogram
            fig, ax = plot_utils.plot_btd_hist(
                        [mtg_btd3, msg_btd3],
                        xlabel="BTD3",
                        ylabel="Probability Density",
                        title="BTD3 values",
                        xmin=0,
                        plotBTD3thresh=True,
                        outname=f"BTD3_{meta['region'].replace(" ","_")}_{meta['time_msg']}.png",
                        latlonstr=dataset.latlonstr,
                        regionstr=f"Plot Region: {meta['region']}",
                        timestr=f"Time: {meta['time_msg']}",
                        outdir=self.outdir,
                        conf_cut=meta["conf_cut"],
                    )

            if self.show_plots:
                plt.show()
            plt.close()

