import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.spatial import KDTree
from enum import Enum

#file_msg = "MSG_202510120200_namib_vars.csv"
#file_mtg = "MTG_202510120200_vars.csv"

#lat_range = (52,53)
#lon_range = (-12,-13)
lat_range = (49,57)
lon_range = (-16,-10)
#lat_range = (45,60)
#lon_range = (-20,-5)
#lat_range = (-30,-20)
#lon_range = (10,20)

#lat_range = (-75,75)
#lon_range = (-75,75)

class RetrievalCode(Enum):
    CONF7_LIBMASK = "conf7_libmask" # Detected conf 7 retrievals in MTG and none in MSG, and reason MSG fails is the liberal beta mask
    CONF7_C1 = "conf7_c1" # Detected conf 7 retrievals in MTG and none in MSG, and reason MSG fails is the BTD2 C1 threshold
    CONF7_C1_LIBMASK = "conf7_c1_libmask" # Detected conf 7 retrievals in MTG and none in MSG, and reason MSG fails is the BTD2 C1 threshold and the beta mask
    CONF7_OTHER = "conf7_other" # Detected conf 7 retrievals in MTG and none in MSG, and reason MSG fails is something else
    CONF4_CONMASK = "conf4_conmask" # Detected conf 4 retrievals in MTG and none in MSG, and reason MSG fails is the conservative beta mask
    CONF4_C4 = "conf4_c4" # Detected conf 4 retrievals in MTG and none in MSG, and reason MSG fails is the BTD2 C4 threshold
    CONF4_C4_CONMASK = "conf4_c4_conmask" # Detected conf 4 retrievals in MTG and none in MSG, and reason MSG fails is the BTD2 C4 threshold and the beta mask
    CONF4_OTHER = "conf4_other" # Detected conf 4 retrievals in MTG and none in MSG, and reason MSG fails is something else
    CONF3_BTDCUTOFF_BTD3_CONMASK = "conf3_btdcutoff_btd3_conmask"
    CONF3_BTDCUTOFF_BTD3 = "conf3_btdcutoff_btd3"
    CONF3_BTDCUTOFF_CONMASK = "conf3_btdcutoff_conmask"
    CONF3_C3_BTD3_CONMASK = "conf3_c3_btd3_conmask"
    CONF3_C3_BTD3 = "conf3_c3_btd3"
    CONF3_C3_CONMASK = "conf3_c3_conmask"
    CONF3_BTD3_CONMASK = "conf3_btd3_conmask"
    CONF3_BTD3 = "conf3_btd3"
    CONF3_CONMASK = "conf3_conmask"
    CONF3_BTDCUTOFF = "conf3_btdcutoff"
    CONF3_C3 = "conf3_c3"
    CONF3_OTHER = "conf3_other"
    NORET = "noret" # No retrievals in either MSG and MTG
    BOTH = "both" # Retrieved ash in both MSG and MTG
    OTHER = "other" # Anything else

codes_to_ignore=[RetrievalCode.NORET, RetrievalCode.BOTH]

retrieval_code_labels = {
    RetrievalCode.CONF7_LIBMASK: "MTG Conf 7, MSG fails: Liberal Mask",
    RetrievalCode.CONF7_C1: "MTG Conf 7, MSG fails: C1 Threshold",
    RetrievalCode.CONF7_C1_LIBMASK: "MTG Conf 7, MSG fails: C1 & Lib Mask",
    RetrievalCode.CONF7_OTHER: "MTG Conf 7, MSG fails: Other",
    RetrievalCode.CONF4_CONMASK: "MTG Conf 4, MSG fails: Conservative Mask",
    RetrievalCode.CONF4_C4: "MTG Conf 4, MSG fails: C4 Threshold",
    RetrievalCode.CONF4_C4_CONMASK: "MTG Conf 4, MSG fails: C4 & Con Mask",
    RetrievalCode.CONF4_OTHER: "MTG Conf 4, MSG fails: Other",
    RetrievalCode.CONF3_BTDCUTOFF_BTD3_CONMASK: "MTG Conf 3, MSG fails: BTDcutoff, BTD3, Con Mask",
    RetrievalCode.CONF3_BTDCUTOFF_BTD3: "MTG Conf 3, MSG fails: BTDcutoff, BTD3",
    RetrievalCode.CONF3_BTDCUTOFF_CONMASK: "MTG Conf 3, MSG fails: BTDcutoff, Con Mask",
    RetrievalCode.CONF3_C3_BTD3_CONMASK: "MTG Conf 3, MSG fails: C3, BTD3, Con Mask",
    RetrievalCode.CONF3_C3_BTD3: "MTG Conf 3, MSG fails: C3, BTD3",
    RetrievalCode.CONF3_C3_CONMASK: "MTG Conf 3, MSG fails: C3, Con Mask",
    RetrievalCode.CONF3_BTD3_CONMASK: "MTG Conf 3, MSG fails: BTD3, Con Mask",
    RetrievalCode.CONF3_BTD3: "MTG Conf 3, MSG fails: BTD3",
    RetrievalCode.CONF3_CONMASK: "MTG Conf 3, MSG fails: Con Mask",
    RetrievalCode.CONF3_BTDCUTOFF: "MTG Conf 3, MSG fails: BTDcutoff",
    RetrievalCode.CONF3_C3: "MTG Conf 3, MSG fails: C3",
    RetrievalCode.CONF3_OTHER: "MTG Conf 3, MSG fails: Other",
    RetrievalCode.NORET: "No Retrieval",
    RetrievalCode.BOTH: "Both Retrieved",
    RetrievalCode.OTHER: "Other"
}

def analyse_csv(indir):

    file_path_msg = indir + file_msg
    file_path_mtg = indir + file_mtg
    
    # Load the CSV file into a DataFrame
    df_msg = pd.read_csv(file_path_msg)
    df_mtg = pd.read_csv(file_path_mtg)

    coords_msg = tuple(zip(np.array(df_msg['Lat']), np.array(df_msg['Lon'])))
    coords_mtg = tuple(zip(np.array(df_mtg['Lat']), np.array(df_mtg['Lon'])))

    loc_coord_msg = [i for i, c in enumerate(coords_msg) if c == coord_msg]
    loc_coord_mtg = [i for i, c in enumerate(coords_mtg) if c == coord_mtg]
    idx_msg = loc_coord_msg[0]
    idx_mtg = loc_coord_mtg[0]

    output_txt = file_path + "{}_{}_comparison.txt".format(file_msg.rsplit("_vars.csv")[0], file_mtg.rsplit("_vars.csv")[0])
    with open(output_txt, "w") as f:
        f.write(f"MSG coordinate: {coord_msg}\n")
        f.write(f"MTG coordinate: {coord_mtg}\n")
        f.write(f"{'Variable':<25}{'MSG':>20}{'MTG':>20}\n")
        for col in df_msg.columns:
            val_msg = df_msg.iloc[idx_msg][col]
            val_mtg = df_mtg.iloc[idx_mtg][col] if col in df_mtg.columns else "N/A"
            f.write(f"{col:<25}{str(val_msg):>20}{str(val_mtg):>20}\n")

    print(f"Comparison written to {output_txt}")

def plot_btd_hist(btds, xlabel, ylabel, title, xmin=0, xmax=0, nbins=50, plotc4=False, plotc3=False, plotc1=False, plotBTD3thresh=False, showhist=False, savehist=True, plot_dir='/home/users/benjamin.honan/Work/analyse_csv/plots/', outname="btdhist.png"):

    if len(btds) > 1 and len(btds[0]) > 1:
        plot_msg_mtg = True 
    else:
        plot_msg_mtg = False

    xmin = np.concatenate(btds).min() if xmin == 0 else xmin
    xmax = np.concatenate(btds).max() if xmax == 0 else xmax

    colors = 'skyblue'
    labels = ''
    if plot_msg_mtg:
        colors = ('orange','skyblue')
        labels = ('MTG','MSG')

    plt.hist(btds, bins=nbins, range=(xmin, xmax), density=True, color=colors, label=labels, edgecolor='black')
    plt.legend(title="Satellite Type")

    '''
        plt.hist(btds[0], bins=nbins, range=(xmin, xmax), density=True, color='orange', label='MTG', edgecolor='black')
        plt.hist(btds[1], bins=nbins, range=(xmin, xmax), density=True, color='skyblue', label='MSG', edgecolor='black')
        plt.legend(title="Satellite Type")
    else:
        plt.hist(btds, bins=nbins, range=(xmin, xmax), color='skyblue', edgecolor='black')
    '''
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Plot vertical lines for c1, c3, c4 thresholds (assume -1 for now) with visible labels
    ylim = plt.gca().get_ylim()
    if plotc1:
        plt.axvline(-2.0, color='red', linestyle='--')
        plt.text(-2.0, ylim[1]*0.95, 'C1 MSG', color='red', rotation=90, va='top', ha='right', backgroundcolor='white')
        plt.axvline(-2.06, color='blue', linestyle='--')
        plt.text(-2.06, ylim[1]*0.95, 'C1 MTG', color='blue', rotation=90, va='top', ha='right', backgroundcolor='white')
    if plotc3:
        plt.axvline(-1, color='green', linestyle='--')
        plt.text(-1, ylim[1]*0.85, 'C3', color='green', rotation=90, va='top', ha='right', backgroundcolor='white')
    if plotc4:
        plt.axvline(-0.5, color='purple', linestyle='--')
        plt.text(-0.5, ylim[1]*0.75, 'C4 MSG', color='purple', rotation=90, va='top', ha='right', backgroundcolor='white')
        plt.axvline(-0.29, color='green', linestyle='--')
        plt.text(-0.29, ylim[1]*0.75, 'C4 MTG', color='green', rotation=90, va='top', ha='right', backgroundcolor='white')

    if showhist:
        plt.show()
    if savehist:
        plt.savefig(plot_dir+'/'+outname)

def get_matches_and_codes(indir, msg_mtg_pairs, write_output_matches, f_output_csv, output_txt=False, threshold=0.01):

    retrievalcodes = []
    msg_matches = []
    mtg_matches = []

    if output_txt:
        f_output_txt = f_output_csv.replace('csv','txt')
        with open(f_output_txt, "w") as f:
            pass  # Just to clear the file at the start

    for file_msg, file_mtg in msg_mtg_pairs:

        file_path_msg = indir + '/' + file_msg
        file_path_mtg = indir + '/' + file_mtg
    
        # Load the CSV file into a DataFrame
        df_msg = pd.read_csv(file_path_msg)
        df_mtg = pd.read_csv(file_path_mtg)

        coords_msg = tuple(zip(np.array(df_msg['Lat']), np.array(df_msg['Lon'])))
        coords_mtg = tuple(zip(np.array(df_mtg['Lat']), np.array(df_mtg['Lon'])))

        n_msg, n_mtg = len(df_msg), len(df_mtg)
        build_msg_tree = True if n_msg < n_mtg else False
        sat_tree, sat_search = ("MSG","MTG") if build_msg_tree else ("MTG","MSG")
        # Define the df used to build the tree as the smallest one to optimise speed (building the tree is O(n log n))
        df_tree, df_search = (df_msg, df_mtg) if build_msg_tree else (df_mtg, df_msg)
        coords_tree = tuple(zip(np.array(df_tree['Lat']), np.array(df_tree['Lon'])))
        coords_search = tuple(zip(np.array(df_search['Lat']), np.array(df_search['Lon'])))
        search_matches = []

        for coord in coords_search:
            tree = KDTree(coords_tree)
            distance, index = tree.query(coord)
            if distance < threshold:
                idx_tree = index
                idx_search = coords_search.index(coord)
                var_tree = df_tree.iloc[idx_tree]
                var_search = df_search.iloc[idx_search]
                # Always append MTG first
                if build_msg_tree:
                    search_matches.append((var_search,var_tree))
                else:
                    search_matches.append((var_tree,var_search))
                if output_txt:
                    with open(f_output_txt, "a") as f:
                        f.write(f"{'Satellite':<25}{sat_tree:>20}{sat_search:>20}\n")
                        for col in df_tree.columns:
                            val_tree = var_tree[col]
                            val_search = var_search[col] if col in df_search.columns else "N/A"
                            f.write(f"{col:<25}{str(val_tree):>20}{str(val_search):>20}\n")
                        f.write("\n")

        for pair in search_matches:
            mtg_match, msg_match = pair[0], pair[1]
            msg_matches.append(msg_match)
            mtg_matches.append(mtg_match)
            mtg_conf, msg_conf = mtg_match["PreFilter_VA_Confidence"], msg_match["PreFilter_VA_Confidence"]
            mtg_btd2, msg_btd2 = mtg_match["BTD2_conf"], msg_match["BTD2_conf"]
            mtg_btd3, msg_btd3 = mtg_match["VolcanicAsh_BTD3"], msg_match["VolcanicAsh_BTD3"]
            mtg_conmask, msg_conmask = mtg_match["BMCon"], msg_match["BMCon"]
            mtg_mbask, msg_libmask = mtg_match["BMLib"], msg_match["BMLib"]
            if mtg_conf == 4 and msg_conf == 0:
                failc4 = msg_btd2 > msg_match[" c4"]
                failconmask = msg_conmask == 'F'
                if failc4 and failconmask:
                    retrievalcode = RetrievalCode("conf4_c4_conmask")
                elif failc4:
                    retrievalcode = RetrievalCode("conf4_c4")
                elif failconmask:
                    retrievalcode = RetrievalCode("conf4_conmask")
                else:
                    retrievalcode = RetrievalCode("conf4_other")
            elif mtg_conf == 3 and msg_conf == 0:
                failc3 = msg_btd2 <= msg_match[" c3"]
                failbtd3 = msg_btd3 > msg_match[" BTD3thresh"]
                failbtdcutoff = msg_btd2 > -0.1
                failconmask = msg_conmask == 'F'
                if failbtdcutoff and failbtd3 and failconmask:
                    retrievalcode = RetrievalCode("conf3_btdcutoff_btd3_conmask")
                elif failbtdcutoff and failbtd3:
                    retrievalcode = RetrievalCode("conf3_btdcutoff_btd3")
                elif failbtdcutoff and failconmask:
                    retrievalcode = RetrievalCode("conf3_btdcutoff_conmask")
                if failc3 and failbtd3 and failconmask:
                    retrievalcode = RetrievalCode("conf3_c3_btd3_conmask")
                elif failc3 and failbtd3:
                    retrievalcode = RetrievalCode("conf3_c3_btd3")
                elif failc3 and failconmask:
                    retrievalcode = RetrievalCode("conf3_c3_conmask")
                elif failbtd3 and failconmask:
                    retrievalcode = RetrievalCode("conf3_btd3_conmask")
                elif failbtd3:
                    retrievalcode = RetrievalCode("conf3_btd3")
                elif failconmask:
                    retrievalcode = RetrievalCode("conf3_conmask")
                elif failbtdcutoff:
                    retrievalcode = RetrievalCode("conf3_btdcutoff")
                elif failc3:
                    retrievalcode = RetrievalCode("conf3_c3")
                else:
                    retrievalcode = RetrievalCode("conf3_other")
            elif mtg_conf == 7 and msg_conf == 0:
                failc1 = msg_btd2 > msg_match["c1"]
                failconmask = msg_libmask == 'F'
                if failc1 and failconmask:
                    retrievalcode = RetrievalCode("conf7_c1_libmask")
                elif failc1:
                    retrievalcode = RetrievalCode("conf7_c1")
                elif failconmask:
                    retrievalcode = RetrievalCode("conf7_libmask")
                else:
                    retrievalcode = RetrievalCode("conf7_other")
            elif mtg_conf == 0 and msg_conf == 0:
                retrievalcode = RetrievalCode("noret")
            elif mtg_conf > 0 and msg_conf > 0:
                retrievalcode = RetrievalCode("both")
            else:
                retrievalcode = RetrievalCode("other")
            retrievalcodes.append(retrievalcode)
        
    df_msg_matches = pd.DataFrame(msg_matches).reset_index(drop=True)
    if len(retrievalcodes) != len(df_msg_matches):
        raise ValueError("Length of retrievalcodes list not equal to list of nearest-neighbour MSG matches")
    df_msg_matches['retrieval_code'] = pd.Series(retrievalcodes)

    if write_output_matches:
        df_msg_matches.to_csv(f_output_csv, index=False)
        print(f"Nearest-neighbour MSG matches written to {f_output_csv}")

    return (msg_matches, mtg_matches, retrievalcodes)

def make_btd_plots(indir, hists_to_plot):

    # Define UK region
    # latlon_uk = (49., 62., -24., 4.)  # lat_min, lat_max, lon_min, lon_max

    hist_path_map = {"UK":('MTG_202510120200_vars.csv','MSG_202510120200_vars.csv')}
    for histstr in hists_to_plot:
        file_path_mtg = indir + '/' + hist_path_map[histstr][0]
        file_path_msg = indir + '/' + hist_path_map[histstr][1]

        # Load the CSV file into a DataFrame
        df_mtg = pd.read_csv(file_path_mtg)
        df_msg = pd.read_csv(file_path_msg)

        lat_min, lat_max = df_msg['Lat'].min(), df_msg['Lat'].max()
        lon_min, lon_max = df_msg['Lon'].min(), df_msg['Lon'].max()

        # Restrict MTG dataframe to the MSG lon/lat
        df_mtg = df_mtg[
            (df_mtg['Lat'] > lat_min) & (df_mtg['Lat'] < lat_max) &
            (df_mtg['Lon'] > lon_min) & (df_mtg['Lon'] < lon_max)
        ]

        # Restrict MSG dataframe to confidence >= 1
        df_msg = df_msg[(df_msg['PreFilter_VA_Confidence'] >= 1)]

        mtg_btd2 = df_mtg["BTD2_conf"].values
        msg_btd2 = df_msg["BTD2_conf"].values
        mtg_btd3 = df_mtg["VolcanicAsh_BTD3"].values
        msg_btd3 = df_msg["VolcanicAsh_BTD3"].values

    # Plot BTD2 histogram for UK
    plt.figure()
    plot_btd_hist(
        [mtg_btd2, msg_btd2],
        xlabel="BTD2",
        ylabel="Probability Density",
        title="UK BTD2 values, VA_Confidence >= 1",
        xmin = -1.,
        plotc4=True,
        outname="UK_BTD2_hist.png",
        showhist=True
    )
    plt.close()

    # Plot BTD3 histogram for UK
    #plt.figure()
    #plot_btd_hist(
    #    [mtg_btd3, msg_btd3],
    #    xlabel="BTD3",
    #    ylabel="Instances",
    #    title="UK: BTD3 values for MSG/MTG matches",
    #    plotc4=True,
    #    outname="UK_BTD3_hist.png"
    #)
    #plt.close()

def analyse_csv_nearestneighbors(indir, outdir, master_csv_file, write_output_matches=True):

    df_master = pd.read_csv(indir+'/'+master_csv_file)
    msg_mtg_pairs = list(zip(df_master['msg_csv'], df_master['mtg_csv']))

    f_output_csv = outdir + "/{}_msg_matches_nn.csv".format(master_csv_file.rsplit(".csv")[0])
    if not os.path.exists(f_output_csv):
        msg_matches, mtg_matches, retrievalcodes = get_matches_and_codes(indir, msg_mtg_pairs, write_output_matches, f_output_csv)
        read_from_file=False
    else:
        df_msg_matches = pd.read_csv(f_output_csv)
        read_from_file=True

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())

    codes = df_msg_matches['retrieval_code'].unique()
    # Work-around for reading from file because can't read enums from a file...
    _codes_to_ignore = [str(_code) for _code in codes_to_ignore] if read_from_file else codes_to_ignore
    colors = plt.cm.get_cmap('tab10', len(codes))

    for i, code in enumerate(codes):
        code_subset = df_msg_matches[df_msg_matches['retrieval_code'] == code]
        if code not in _codes_to_ignore:
            plt.scatter(
                code_subset['Lon'], code_subset['Lat'],
                s=2,
                label=retrieval_code_labels.get(RetrievalCode(code.split('.')[-1].lower()), str(code)) if read_from_file else retrieval_code_labels.get(code, str(code)),
                color=colors(i),
                alpha=0.8
            )

    plt.legend(title='Retrieval Type', fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(lon_range[0], lon_range[1])
    plt.ylim(lat_range[0], lat_range[1])
    ax.coastlines()
    plt.title("MSG/MTG Retrieval Type Comparison")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    # Save the plot before showing
    plot_path = outdir + "/{}_retrieval_codes.png".format(master_csv_file.rsplit(".csv")[0])
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()

def plot_latlon_points(indir, master_csv_file):

    df_master = pd.read_csv(indir+'/'+master_csv_file)
    csvfiles = df_master.values.flatten().tolist()
    csvpaths = [indir+'/'+file for file in csvfiles]

    colors = plt.cm.get_cmap('tab10', len(csvpaths))
    ax = plt.axes(projection=ccrs.PlateCarree())
    for icsvpath, csvpath in enumerate(csvpaths):
        df = pd.read_csv(csvpath)
        lons = np.array(df["Lon"])
        lats = np.array(df["Lat"])

        if 'MSG' in csvfiles[icsvpath]:
            ax.scatter(lons, lats, s=1, label=csvfiles[icsvpath], color=colors(icsvpath))
    plt.xlim(lon_range[0], lon_range[1])
    plt.ylim(lat_range[0], lat_range[1])
    plt.legend(title='CSV File')
    ax.coastlines()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Create MSG/MTG retrieval code plots")
    parser.add_argument('--indir', type=str, default='/home/users/benjamin.honan/Work/analyse_csv/csv_files/')
    parser.add_argument('--outdir', type=str, default='/home/users/benjamin.honan/Work/analyse_csv/plots/')
    parser.add_argument('--master_csv_file', type=str, default='MSG_MTG_pairs.csv')
    parser.add_argument('--plot_points', action="store_true")
    parser.add_argument('--plot_btd', action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.plot_points:
        plot_latlon_points(args.indir, args.master_csv_file)
    elif args.plot_btd:
        make_btd_plots(args.indir, ["UK"])
    else:
        analyse_csv_nearestneighbors(args.indir, args.outdir, args.master_csv_file)

