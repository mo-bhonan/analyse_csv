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
from dataset import Dataset, iter_loaded
from collections import Counter
import io
import copy

class Plotter:
    def __init__(self, indir, outdir, master_csv, show_plots=False):
        self.indir = Path(indir)
        self.outdir = Path(outdir)
        self.show_plots = show_plots

        df = pd.read_csv(self.indir / Path(master_csv))
        self.datasets = [Dataset(self.indir, row.to_dict()) for _, row in df.iterrows()]

    def plot_nearestneighbors(self, recreate_csv, plot_histograms=True):
        for dataset in iter_loaded(self.datasets):
            # df_msg, df_mtg will go out of scope at the end of each iteration, so there won't be a memory leak.
            # They point to the same dataframe as dataset.data_msg and dataset.data_mtg
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            cutstr = config.getcutstr(meta.get('conf_cut'))
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
                    sat, variable, operator, values = config.splitcutstr(cutstr)
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
        for dataset in iter_loaded(self.datasets):
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            cutstr = config.getcutstr(meta.get('conf_cut'))

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

    def plot_beta_masks(self, plotproxies=True, plotlatlon=True):

        for dataset in iter_loaded(self.datasets):
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            cutstr = config.getcutstr(meta.get('conf_cut'))
            _, var, op, val = config.splitcutstr(cutstr)
            # Doesn't make sense to plot the beta masks for some cases such as conf-7 not over NH arid regions, as the beta masks are not used for detection
            if var == "Median_VA_Confidence" and op == "==" and val == "7":
                continue

            df_mtg_unfiltered = df_mtg[(df_mtg['aa'] == -0.4) & (df_mtg['bb'] == -0.4) & (df_mtg['cc'] == 2.5)]
            df_mtg_low_lat = df_mtg[(df_mtg['aa'] == -0.9) & (df_mtg['bb'] == 0.0) & (df_mtg['cc'] == 2.3)]
            df_mtg_high_zenith = df_mtg[(df_mtg['aa'] == -1.0) & (df_mtg['bb'] == 0.0) & (df_mtg['cc'] == 2.3)]
            df_mtg_sh_arid = df_mtg[(df_mtg['aa'] == -1.0) & (df_mtg['bb'] == 0.0) & (df_mtg['cc'] == 1.6)]
            df_mtg_nh_arid = df_mtg[(df_mtg['aa'] == -1.0) & (df_mtg['bb'] == 0.0) & (df_mtg['cc'] == 1.3)]

            df_msg_unfiltered = df_msg[(df_msg['aa'] == -0.4) & (df_msg['bb'] == -0.4) & (df_msg['cc'] == 2.5)]
            df_msg_low_lat = df_msg[(df_msg['aa'] == -0.9) & (df_msg['bb'] == 0.0) & (df_msg['cc'] == 2.3)]
            df_msg_high_zenith = df_msg[(df_msg['aa'] == -1.0) & (df_msg['bb'] == 0.0) & (df_msg['cc'] == 2.3)]
            df_msg_sh_arid = df_msg[(df_msg['aa'] == -1.0) & (df_msg['bb'] == 0.0) & (df_msg['cc'] == 1.6)]
            df_msg_nh_arid = df_msg[(df_msg['aa'] == -1.0) & (df_msg['bb'] == 0.0) & (df_msg['cc'] == 1.3)]

            dfs = [(df_mtg_unfiltered, df_msg_unfiltered, "Unfiltered"), (df_mtg_low_lat, df_msg_low_lat, "Low_Lat"),
                   (df_mtg_high_zenith, df_msg_high_zenith, "High_Zenith"), (df_mtg_sh_arid, df_msg_sh_arid, "SH_Arid"),
                   (df_mtg_nh_arid, df_msg_nh_arid, "NH_Arid")]

            # If we want to plot the SO2 and Ash proxies 
            if plotproxies:
                threshold = 0.1
                df_msg_det = df_msg[(df_msg['BTD2_conf'] <= -1.0) | (df_msg['VolcanicAsh_BTD3'] <= 1.5)]
                df_mtg_det = df_mtg[(df_mtg['BTD2_conf'] <= -1.0) | (df_mtg['VolcanicAsh_BTD3'] <= 1.5)]

                # Create a shallow-copy of dataset and set new references to modified dataframes
                dataset_det = copy.copy(dataset)
                dataset_det.load(df_msg=df_msg_det, df_mtg=df_mtg_det)
                if plotlatlon:
                    plot_utils.plot_latlon_points_dataset(dataset_det, custombounds=((40.,44.),(13.,15.)))
                    outname = f"latlon_det_all_msg{meta['time_msg']}_mtg{meta['time_mtg']}"
                    plot_utils.save_plots(self.outdir, outname)
                    if self.show_plots:
                        plt.show()
                        plt.close()

                # Set lt to False to perform the search for nearest neighbours ABOVE the threshold distance
                # and get the MSG matches corresponding to points which have a far away MTG point (proxy for SO2 cloud)
                search_matches = nn.find_nn(dataset_det, threshold, lt=False)
                msg_matches = [match[1] for match in search_matches]
                df_msg_matches = pd.DataFrame(msg_matches).reset_index(drop=True)

                # Filter MSG points to lat/lon region around the SO2 cloud (set in config file)
                df_msg_matches = df_msg_matches[((df_msg_matches['Lat'] > config.SO2_cloud_latmin) & (df_msg_matches['Lat'] < config.SO2_cloud_latmax)) &
                                                ((df_msg_matches['Lon'] > config.SO2_cloud_lonmin) & (df_msg_matches['Lon'] < config.SO2_cloud_lonmax))]

                if not df_msg_matches.empty:

                    # Now use the MSG matches to search for the nearest neighbour MTG pixels without any detection thresholds
                    # Filter the lat/lon region to speed up the search
                    df_mtg_filt = df_mtg[((df_mtg['Lat'] > 13.) & (df_mtg['Lat'] < 14.)) & ((df_mtg['Lon'] > 40.) & (df_mtg['Lon'] < 45.))]
                    dataset_so2 = copy.copy(dataset)
                    dataset_so2.load(df_msg=df_msg_matches, df_mtg=df_mtg_filt)
                    # Do a strict nearest-neighbour search around the msg SO2 cloud points, and return the nearest MTG neighbours
                    mtg_matches = nn.find_nn(dataset_so2)

                    df_mtg_matches = pd.DataFrame(mtg_matches).reset_index(drop=True)
                    dfs.append((df_mtg_matches, df_msg_matches, "SO2_Proxy"))
                    if plotlatlon:
                        dataset_so2.load(df_msg=df_msg_matches, df_mtg=df_mtg_matches)
                        plot_utils.plot_latlon_points_dataset(dataset_so2, boundsfrommeta=False)
                        outname = f"latlon_det_so2_msg{meta['time_msg']}_mtg{meta['time_mtg']}"
                        plot_utils.save_plots(self.outdir, outname)
                        if self.show_plots:
                            plt.show()
                        plt.close()

                    londiff = abs(dataset_det.lonmax_mtg - dataset_det.lonmin_mtg)
                    latdiff = abs(dataset_det.latmax_mtg - dataset_det.latmin_mtg)

                    df_mtg_filt = df_mtg_det[((df_mtg_det['Lat'] > config.ashbounds[0][0]) & (df_mtg_det['Lat'] < config.ashbounds[0][1])) &
                                            ((df_mtg_det['Lon'] > config.ashbounds[1][0]) & (df_mtg_det['Lon'] < config.ashbounds[1][1]))]
                    # Filter MSG to a window around the MTG ash cloud to make computation faster
                    df_msg_filt = df_msg[((df_msg['Lat'] > config.ashbounds[0][0] - 0.1*latdiff) & (df_msg['Lat'] < config.ashbounds[0][1] + 0.1*latdiff))&
                                        ((df_msg['Lon'] > config.ashbounds[1][0] - 0.1*londiff) & (df_msg['Lon'] < config.ashbounds[1][1] + 0.1*latdiff))]
                    dataset_ash = copy.copy(dataset)
                    dataset_ash.load(df_msg = df_msg_filt, df_mtg=df_mtg_filt)
                    
                    msg_matches = nn.find_nn(dataset_ash)
                    df_msg_matches = pd.DataFrame(msg_matches).reset_index(drop=True)
                    dfs.append((df_mtg_det, df_msg_matches, "Ash_Proxy"))
                    if plotlatlon:
                        dataset_ash.load(df_msg=df_msg_matches, df_mtg=df_mtg_filt)
                        plot_utils.plot_latlon_points_dataset(dataset_ash, boundsfrommeta=False)
                        outname = f"latlon_det_ash_msg{meta['time_msg']}_mtg{meta['time_mtg']}"
                        plot_utils.save_plots(self.outdir, outname)
                        if self.show_plots:
                            plt.show()
                        plt.close()
                else:
                    print(f"WARNING: Skipping make proxy beta mask plots for MSG {meta['time_msg']}, MTG {meta['time_mtg']}.")

            for df_mtg, df_msg, name in dfs:
                if len(df_mtg) == 0 or len(df_msg) == 0:
                    continue

                aa, bb, c = config.getbmthresholds(name)
                # Define x range
                x = np.linspace(0, 2.5, 100)

                # Define polynomial function
                y_conservative = aa * x**2 + bb * x + c - 0.4
                y_liberal = aa * x**2 + bb * x + c

                mtg_beta_870_108, msg_beta_870_108 = df_mtg['Beta_870_108'], df_msg['Beta_870_108']
                mtg_beta_120_108, msg_beta_120_108 = df_mtg['Beta_120_108'], df_msg['Beta_120_108']
                for mode in ["msg", "mtg"]:

                    # Create plot
                    plt.figure(figsize=(8, 6))
                    _name = name.replace("_"," ")
                    if _name in ["SO2 Proxy", "Ash Proxy"]:
                        _name = "Low Lat"
                    plt.plot(x, y_conservative, 'r--', label=f'Conservative Beta Mask: {_name}')
                    plt.plot(x, y_liberal, 'b--', label=f'Liberal Beta Mask: {_name}')
                    if mode == "msg":
                        plt.xlabel(r'$\beta$(8.7,10.8)')
                        plt.ylabel(r'$\beta$(12.0,10.8)')
                    else:
                        plt.xlabel(r'$\beta$(8.7,10.5)')
                        plt.ylabel(r'$\beta$(12.3,10.5)')
                    plt.grid(True)
                    plt.xlim(0, 2.5)
                    plt.ylim(0, 2.5)

                    # Create a 2D histogram for MTG and MSG beta values
                    x_bins = np.linspace(0, 2.5, 100)
                    y_bins = np.linspace(0, 2.5, 100)
                    if mode == 'msg':
                        xvals = np.array(msg_beta_870_108)
                        yvals = np.array(msg_beta_120_108)
                        H, xedges, yedges = np.histogram2d(xvals, yvals, bins=[x_bins, y_bins])
                    else:
                        xvals = np.array(mtg_beta_870_108)
                        yvals = np.array(mtg_beta_120_108)
                        H, xedges, yedges = np.histogram2d(xvals, yvals, bins=[x_bins, y_bins])

                    # Plot density
                    X, Y = np.meshgrid(xedges, yedges)
                    pcm = plt.pcolormesh(X, Y, H.T, cmap='gist_heat_r', shading='auto')

                    # Calculate percentage below conservative and liberal lines
                    # For each point, check if y < y_conservative(x) or y < y_liberal(x)
                    def poly_conservative(xv):
                        return aa * xv**2 + bb * xv + c - 0.4
                    def poly_liberal(xv):
                        return aa * xv**2 + bb * xv + c

                    below_conservative = np.sum(yvals < poly_conservative(xvals))
                    below_liberal = np.sum(yvals < poly_liberal(xvals))
                    total_points = len(xvals)
                    perc_below_conservative = (below_conservative / total_points) * 100 if total_points > 0 else 0
                    perc_below_liberal = (below_liberal / total_points) * 100 if total_points > 0 else 0
                    passed_str = f"{perc_below_conservative:.1f}% below con., {perc_below_liberal:.1f}% below lib."

                    # Add colorbar for combined scale
                    plt.colorbar(pcm, label='Count', orientation='vertical')

                    plotstr = f"{meta['region']}\n"+f"{dataset.latlonstr_msg}\n"+f"{passed_str}\n"+f"Time MSG: {meta['time_msg']}\n"+f"Time MTG: {meta['time_mtg']}"
                    xlim = plt.gca().get_xlim()
                    ylim = plt.gca().get_ylim()
                    plt.text(xlim[0] + 0.025*(xlim[1]-xlim[0]), ylim[0]+0.15*(ylim[1]-ylim[0]), plotstr, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                    plt.legend()
                    if name == "SO2_Proxy":
                        plt.title(mode.upper() + r' $\beta$ space for pixels flagged with unique MSG detections')
                    elif name == "Ash_Proxy":
                        if mode == "mtg":
                            plt.title(mode.upper() + r' $\beta$ space for pixels passing detection')
                        else:
                            plt.title(r'$\beta$ space for MSG pixels matched to MTG pixels passing detection')
                    else:
                        plt.title(mode.upper() + r' $\beta$ space')
                    # Save plot
                    outname = f"beta_space_{mode}_{meta['region'].replace(" ","_")}_msg{meta['time_msg']}_mtg{meta['time_mtg']}_{name}.png"
                    plot_utils.save_plots(self.outdir, outname)
                    if self.show_plots:
                        plt.show()
                    plt.close()

    def plot_constraints(self):

        for dataset in iter_loaded(self.datasets):
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            cutstr = config.getcutstr(meta.get('conf_cut'))
            constraintslist = config.DICT_CUT_CONSTRAINT.get(meta.get('conf_cut'),[])
            for constraints in constraintslist:

                # Apply the constraints and calculate the percentage passed 
                df_msg_constr, perc_passed_msg = filters.apply_constraints(df_msg, constraints, output_perc=True)
                df_mtg_constr, perc_passed_mtg = filters.apply_constraints(df_mtg, constraints, output_perc=True)
                dataset_constr = copy.copy(dataset)
                dataset_constr.load(df_msg = df_msg_constr, df_mtg=df_mtg_constr)

                passed_str_msg = f"MSG passed: {round(perc_passed_msg,1)}%"
                passed_str_mtg = f"MTG passed: {round(perc_passed_mtg,1)}%"
                passed_str = f"{passed_str_msg}\n{passed_str_mtg}"
                constr_str_title = filters.format_constraints_for_title(constraints)
                constr_str_file = filters.format_constraints_for_filename(constraints)
                plotstr = f"Plot Region: {meta['region']}\n"+f"Percentage Passed: {passed_str}\n"+f"MSG Time: {meta['time_msg']}\n"+f"MTG Time: {meta['time_mtg']}"
                title = f'Pixels passing {constr_str_title}'

                plot_utils.plot_latlon_points_dataset(dataset_constr, title=title, plotstr=plotstr)
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                plt.text(xlim[0] + 0.02*(xlim[1]-xlim[0]), ylim[1] - 0.025*(ylim[1]-ylim[0]), plotstr, color='black', va='top', ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
                outname = f"grid_{constr_str_file}_{meta['region'].replace(" ","_")}_msg{meta['time_msg']}_mtg{meta['time_mtg']}.png"
                plot_utils.save_plots(self.outdir, outname)
                if self.show_plots:
                    plt.show()
                plt.close()

    def plot_btd3(self):

        for dataset in iter_loaded(self.datasets):
            df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
            meta = dataset.metadata
            df_msg = df_msg[(df_msg['BTD2_conf'] <= -1.0) | (df_msg['VolcanicAsh_BTD3'] <= 1.5)]
            df_msg = df_msg[((df_msg['Lat'] > config.ashbounds[0][0]) & (df_msg['Lat'] < config.ashbounds[0][1])) &
                                    ((df_msg['Lon'] > config.ashbounds[1][0]) & (df_msg['Lon'] < config.ashbounds[1][1]))]
            df_mtg = df_mtg[(df_mtg['BTD2_conf'] <= -1.0) | (df_mtg['VolcanicAsh_BTD3'] <= 1.5)]
            df_mtg = df_mtg[((df_mtg['Lat'] > config.ashbounds[0][0]) & (df_mtg['Lat'] < config.ashbounds[0][1])) &
                                    ((df_mtg['Lon'] > config.ashbounds[1][0]) & (df_mtg['Lon'] < config.ashbounds[1][1]))]
            dataset.load(df_msg=df_msg, df_mtg=df_mtg)

            ax_msg, ax_mtg = plot_utils.plot_latlon_points_dataset(dataset, plotBTD3=True)
            outname_msg = f"MSG_BTD3_map_{meta['time_msg']}"
            plot_utils.save_plots(self.outdir, outname_msg)

            outname_mtg = f"MTG_BTD3_map_{meta['time_mtg']}"
            plot_utils.save_plots(self.outdir, outname_mtg)
            if self.show_plots:
                plt.show()
            plt.close()

