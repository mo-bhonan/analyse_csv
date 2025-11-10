import pandas as pd
import os
import argparse
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from scipy.spatial import KDTree
from enum import Enum

#file_msg = "MSG_202510120200_namib_vars.csv"
#file_mtg = "MTG_202510120200_vars.csv"

#lat_range = (52,53)
#lon_range = (-14,-13)
lat_range = (49,58)
lon_range = (-16,-8)
#lat_range = (45,60)
#lon_range = (-20,-5)
#lat_range = (-30,-20)
#lon_range = (10,20)

#lat_range = (-75,75)
#lon_range = (-75,75)
op_map = {
    '==': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le
}

op_map_str = {
    '==': 'eq',
    '!=': 'ne',
    '>': 'gt',
    '<': 'lt',
    '>=': 'ge',
    '<=': 'le'
}

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
    CONF1_C3_LIBMASK = "conf1_c3_libmask"
    CONF1_C4_LIBMASK = "conf1_c4_libmask"
    CONF1_C4 = "conf1_c4"
    CONF1_C3 = "conf1_c3"
    CONF1_LIBMASK = "conf1_libmask"
    CONF1_OTHER = "conf1_other"
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
    RetrievalCode.CONF1_C3_LIBMASK : "MTG Conf 1, MSG fails: C3, Liberal Mask",
    RetrievalCode.CONF1_C4_LIBMASK : "MTG Conf 1, MSG fails: C4, Liberal Mask",
    RetrievalCode.CONF1_C4 : "MTG Conf 1, MSG fails: C4 Threshold",
    RetrievalCode.CONF1_C3 : "MTG Conf 1, MSG fails: C3 Threshold",
    RetrievalCode.CONF1_LIBMASK : "MTG Conf 1, MSG fails: Liberal Mask",
    RetrievalCode.CONF1_OTHER : "MTG Conf 1, MSG fails: Other",
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

def plot_btd_hist(btds, xlabel, ylabel, title, xmin=0, xmax=0, nbins=50, plotc4=False, plotc3=False, plotc1=False, plotBTD3thresh=False, showhist=False, savehist=True, plot_dir='/home/users/benjamin.honan/Work/analyse_csv/plots/', outname="btdhist.png", latlonstr="", regionstr="", timestr=""):

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
        #mtg first
        mtg_perc_below_c4 = msg_perc_below_c4 = mtg_perc_below_btd3 = msg_perc_below_btd3 = 0.
        if plotc4:
            mtg_perc_below_c4 = (sum(1 for btd in btds[0] if btd < -0.29)/len(btds[0])) * 100
            msg_perc_below_c4 = (sum(1 for btd in btds[1] if btd < -0.5)/len(btds[1])) * 100
        if plotBTD3thresh:
            mtg_perc_above_btd3 = (sum(1 for btd in btds[0] if btd > 1.5)/len(btds[0])) * 100
            msg_perc_above_btd3 = (sum(1 for btd in btds[1] if btd > 1.5)/len(btds[1])) * 100

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
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plt.ylim(ylim[0], ylim[1] * 1.45)  # Increase y-limit by 45% to make space for text
    ylim = plt.gca().get_ylim()
    if regionstr:
        plt.text(
            xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*0.98,
            regionstr,
            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    if latlonstr:
        plt.text(
            xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*0.92,
            latlonstr,
            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
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
        plt.text(-0.5, ylim[1]*0.65, 'C4 MSG', color='purple', rotation=90, va='top', ha='right', backgroundcolor='white')
        plt.axvline(-0.29, color='green', linestyle='--')
        plt.text(-0.29, ylim[1]*0.65, 'C4 MTG', color='green', rotation=90, va='top', ha='right', backgroundcolor='white')
        textstrmtg = f"MTG % below C4: {mtg_perc_below_c4:.1f}%"
        textstrmsg = f"MSG % below C4: {msg_perc_below_c4:.1f}%"
        plt.text(
            xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*0.86,
            textstrmtg,
            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        plt.text(
            xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*0.80,
            textstrmsg,
            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    if plotBTD3thresh:
        plt.axvline(1.5, color='purple', linestyle='--')
        plt.text(1.5, ylim[1]*0.65, 'BTD3 Thresh', color='purple', rotation=90, va='top', ha='right', backgroundcolor='white')
        textstrmtg = f"MTG % above BTD3 Thresh: {mtg_perc_above_btd3:.1f}%"
        textstrmsg = f"MSG % above BTD3 Thresh: {msg_perc_above_btd3:.1f}%"
        plt.text(
            xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*0.86,
            textstrmtg,
            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        plt.text(
            xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*0.80,
            textstrmsg,
            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    if timestr:
        plt.text(
            xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*0.74,
            timestr,
            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    if savehist:
        print(f"Saving figure to:{plot_dir+'/'+outname}")
        plt.savefig(plot_dir+'/'+outname)
    if showhist:
        plt.show()

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
            #mtg_conf, msg_conf = mtg_match["Median_VA_Confidence"], msg_match["PreFilter_VA_Confidence"]
            mtg_btd2, msg_btd2 = mtg_match["BTD2_conf"], msg_match["BTD2_conf"]
            mtg_btd3, msg_btd3 = mtg_match["VolcanicAsh_BTD3"], msg_match["VolcanicAsh_BTD3"]
            mtg_conmask, msg_conmask = mtg_match["BMCon"], msg_match["BMCon"]
            mtg_mbask, msg_libmask = mtg_match["BMLib"], msg_match["BMLib"]
            if mtg_conf == 4 and msg_conf == 0:
                failc4 = msg_btd2 > msg_match["c4"]
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
                failc3 = msg_btd2 <= msg_match["c3"]
                failbtd3 = msg_btd3 > msg_match["BTD3thresh"]
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
            elif mtg_conf == 1 and msg_conf == 0:
                failc4 = msg_btd2 > msg_match["c4"]
                failc3 = msg_btd2 <= msg_match["c3"]
                faillibmask = msg_libmask == 'F'
                if failc3 and faillibmask:
                    retrievalcode = RetrievalCode("conf1_c3_libmask")
                elif failc4 and faillibmask:
                    retrievalcode = RetrievalCode("conf1_c4_libmask")
                elif failc4:
                    retrievalcode = RetrievalCode("conf1_c4")
                elif failc3:
                    retrievalcode = RetrievalCode("conf1_c3")
                elif faillibmask:
                    retrievalcode = RetrievalCode("conf1_libmask")
                else:
                    retrievalcode = RetrievalCode("conf1_other")
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
            elif mtg_conf == 0: # and msg_conf == 0:
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
        df_mtg_matches = pd.DataFrame(mtg_matches).reset_index(drop=True)
        df_msg_matches['MTG_BTD2_conf'] = df_mtg_matches['BTD2_conf']
        df_msg_matches['MTG_PreFilter_VA_Confidence'] = df_mtg_matches['PreFilter_VA_Confidence']
        df_msg_matches['MTG_Median_VA_Confidence'] = df_mtg_matches['Median_VA_Confidence']
        df_msg_matches.to_csv(f_output_csv, index=False)
        print(f"Nearest-neighbour MSG matches written to {f_output_csv}")

    return (msg_matches, mtg_matches, retrievalcodes)

#def make_btd_plots(indir, hists_to_plot):
#def plot_latlon_points(indir, outdir, master_csv_file, constraints=None, plotonly=""):
'''
    df_master = pd.read_csv(indir+'/'+master_csv_file)
    #csvfiles = df_master.values.flatten().tolist()
    #csvpaths = [indir+'/'+file for file in csvfiles]

    #for icsvpath, csvpath in enumerate(csvpaths):
    for mtgcsv, msgcsv, region in zip(df_master['mtg_csv'], df_master['msg_csv'], df_master['region']):
'''
def make_btd_plots(indir, outdir, master_csv_file):

    #hist_path_map = {"UK":('MTG_202510120200_all_points_vars.csv','MSG_202510120200_all_points_vars.csv')}
    #hist_path_map = {"UK":('../plots/MSG_MTG_pairs_UK_allpoints_msg_matches_nn.csv','../plots/MSG_MTG_pairs_UK_allpoints_msg_matches_nn.csv')}
    df_master = pd.read_csv(indir+'/'+master_csv_file)

    for mtgcsv, msgcsv, region in zip(df_master['mtg_csv'], df_master['msg_csv'], df_master['region']):
        mtgpath, msgpath = indir+'/'+mtgcsv, indir+'/'+msgcsv
        df_mtg = pd.read_csv(mtgpath)
        df_msg = pd.read_csv(msgpath)

        lat_min, lat_max = df_msg['Lat'].min(), df_msg['Lat'].max()
        lon_min, lon_max = df_msg['Lon'].min(), df_msg['Lon'].max()

        latstr = '('+str(round(lat_min,1))+','+str(round(lat_max,1))+')'
        lonstr = '('+str(round(lon_min,1))+','+str(round(lon_max,1))+')'
        timestr = msgcsv.split("_")[1]

        # Restrict MTG dataframe to the MSG lon/lat
        df_mtg = df_mtg[
            (df_mtg['Lat'] > lat_min) & (df_mtg['Lat'] < lat_max) &
            (df_mtg['Lon'] > lon_min) & (df_mtg['Lon'] < lon_max)
        ]

        # Restrict dataframes to confidence >= 1
        #df_msg = df_msg[(df_msg['prefilter_va_confidence'] >= 1)]
        #df_mtg = df_mtg[(df_mtg['prefilter_va_confidence'] >= 1)]
        #df_msg = df_msg[(df_msg['MTG_PreFilter_VA_Confidence'] == 4) & (df_msg['PreFilter_VA_Confidence'] == 0)]
        #df_mtg = df_mtg[(df_mtg['MTG_PreFilter_VA_Confidence'] == 4) & (df_msg['PreFilter_VA_Confidence'] == 0)]

        mtg_btd2 = df_mtg["BTD2_conf"].values
        msg_btd2 = df_msg["BTD2_conf"].values
        mtg_btd3 = df_mtg["VolcanicAsh_BTD3"].values
        msg_btd3 = df_msg["VolcanicAsh_BTD3"].values

        #mtg_btd2 = df_mtg["MTG_BTD2_conf"].values
        #msg_btd2 = df_msg["BTD2_conf"].values
        #mtg_btd3 = df_mtg["VolcanicAsh_BTD3"].values
        #msg_btd3 = df_msg["VolcanicAsh_BTD3"].values

    # Plot BTD2 histogram
    plt.figure()
    plot_btd_hist(
        [mtg_btd2, msg_btd2],
        xlabel="BTD2",
        ylabel="Probability Density",
        title="BTD2 values",
        xmin = -1.,
        plotc4=True,
        outname=f"BTD2_{region}_{timestr}.png",
        latlonstr=f"Lat/Lon: {latstr}/{lonstr}",
        regionstr=f"Plot Region: {region}",
        timestr=f"Time: {timestr}",
        showhist=True
    )
    plt.close()

    # Plot BTD3 histogram
    plt.figure()
    plot_btd_hist(
        [mtg_btd3, msg_btd3],
        xlabel="BTD3",
        ylabel="Probability Density",
        title="UK: BTD3 values for MSG/MTG matches",
        xmin=1.0,
        plotBTD3thresh=True,
        outname=f"BTD3_{region}_{timestr}.png",
        latlonstr=f"Lat/Lon: {latstr}/{lonstr}",
        regionstr=f"Plot Region: {region}",
        timestr=f"Time: {timestr}",
        showhist=True
    )
    plt.close()

def analyse_csv_nearestneighbors(indir, outdir, master_csv_file, recreate_csv, write_output_matches=True):

    df_master = pd.read_csv(indir+'/'+master_csv_file)
    msg_mtg_pairs = list(zip(df_master['msg_csv'], df_master['mtg_csv']))

    f_output_csv = outdir + "/{}_msg_matches_nn.csv".format(master_csv_file.rsplit(".csv")[0])
    if not os.path.exists(f_output_csv) or recreate_csv:
        msg_matches, mtg_matches, retrievalcodes = get_matches_and_codes(indir, msg_mtg_pairs, write_output_matches, f_output_csv)

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

    codes = df_msg_matches['retrieval_code'].unique()
    # Work-around for reading from file because can't read enums from a file...
    _codes_to_ignore = [str(_code) for _code in codes_to_ignore] if read_from_file else codes_to_ignore
    colors = plt.get_cmap('tab10', len(codes))

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

def plot_beta_masks(indir, outdir, master_csv_file, *args):
    aa = -0.4
    bb = -0.4
    c = 2.5

    plotmodes = ['mtg','msg','both']
    plotmode = args[0].lower() if args[0].lower() in plotmodes else ''

    # Define x range
    x = np.linspace(0, 2.5, 100)

    # Define polynomial function
    y_conservative = aa * x**2 + bb * x + c - 0.4
    y_liberal = aa * x**2 + bb * x + c

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_conservative, 'r--', label='y = aa*x^2 + bb*x + c - 0.4')
    plt.plot(x, y_liberal, 'b--', label='y = aa*x^2 + bb*x + c')
    plt.xlabel(r'$\beta$(8.7,10.8)')
    plt.ylabel(r'$\beta$(12.0,10.8)')
    plt.grid(True)
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)

    df_master = pd.read_csv(indir+'/'+master_csv_file)

    # Current just plot the first MSG, MTG pair
    df_mtg = pd.read_csv(indir + '/' + df_master['mtg_csv'][0])
    df_msg = pd.read_csv(indir + '/' + df_master['msg_csv'][0])

    mtg_beta_870_108, msg_beta_870_108 = df_mtg['Beta_870_108'], df_msg['Beta_870_108']
    mtg_beta_120_108, msg_beta_120_108 = df_mtg['Beta_120_108'], df_msg['Beta_120_108']

    if plotmode in ['mtg','both']:
        plt.scatter(
            mtg_beta_870_108, mtg_beta_120_108, 
            s=3,
            label='MTG',
            color='green',
            alpha=1.0
        )

    if plotmode in ['msg','both']:
        plt.scatter(
            msg_beta_870_108, msg_beta_120_108, 
            s=3,
            label='MSG',
            color='blue',
            alpha=1.0
        )

    plt.legend()
    plt.title(r'$\beta$ space '+plotmode.upper())
    # Save plot
    plotpath = outdir+'/'+f"beta_masks{'_'+plotmode if plotmode else ''}.png"
    plt.savefig(plotpath)
    plt.show()
    print(f"Plot saved as {plotpath}")

def plot_latlon_points(indir, outdir, master_csv_file, constraints=None, plotonly=""):

    # constraints should be of the format {'variables':[],'operators':[],'values':[]}, e.g.
    # {'variables':["BTD2_conf", "BMCon"], 'operators':['>=', '=='], 'values':['df_c4','T']}
    # if 'df_x' is given in the values, df['x'] is treated as the value

    plotmsg = True
    plotmtg = True
    if plotonly.lower() == "msg":
        plotmtg = False
    elif plotonly.lower() == "mtg":
        plotmsg = False

    df_master = pd.read_csv(indir+'/'+master_csv_file)
    #csvfiles = df_master.values.flatten().tolist()
    #csvpaths = [indir+'/'+file for file in csvfiles]

    #for icsvpath, csvpath in enumerate(csvpaths):
    for mtgcsv, msgcsv, region in zip(df_master['mtg_csv'], df_master['msg_csv'], df_master['region']):
        ax = plt.axes(projection=ccrs.PlateCarree())
        lon_min = lat_min = 100.
        lon_max = lat_max = -100.
        passed_str = ""
        legend_elements = []
        mtgpath, msgpath = indir+'/'+mtgcsv, indir+'/'+msgcsv
        df_mtg = pd.read_csv(mtgpath)
        df_msg = pd.read_csv(msgpath)
        dfs = []
        if plotmtg:
            dfs.append((df_mtg, 'MTG'))
        if plotmsg:
            dfs.append((df_msg, 'MSG'))

        # Apply the constraints and calculate the percentage passed 
        if constraints:
            if not isinstance(constraints, dict):
                raise TypeError(f"Expected constraints parameter to be a dictionary. Got {type(constraints)}.")
            for idf, dflist in enumerate(dfs):
                df = dflist[0]
                sat = dflist[1]
                mask = np.ones(len(df), dtype=bool)
                for var, op_str, val in zip(constraints['variables'], constraints['operators'], constraints['values']):
                    op_func = op_map[op_str]
                    # if 'df_x' is given in the values, df['x'] is treated as the value
                    if isinstance(val, str) and val.startswith('df_'):
                        val = df[val[3:]]
                    mask &= op_func(df[var], val)
                # Modify original dataframe
                #dflist[idf] = df[mask]
                if sat == 'MTG':
                    df_mtg = df_mtg[mask]
                else:
                    df_msg = df_msg[mask]
                perc_passed = (np.array(mask).sum()/len(mask))*100.
                if idf != len(dfs)-1:
                    passed_str += f"{round(perc_passed,1)}% {sat}, "
                else:
                    passed_str += f"{round(perc_passed,1)}% {sat}"

        if plotmsg:
            lons_msg = np.array(df_msg["Lon"])
            lats_msg = np.array(df_msg["Lat"])
            # Assume lon/lat min max are same for MSG and MTG. TODO: Could add error catching to make sure this is true
            #lon_min = min(lon_min, lons_msg.min())
            #lon_max = max(lon_max, lons_msg.max())
            #lat_min = min(lat_min, lats_msg.min())
            #lat_max = max(lat_max, lats_msg.max())

            lon_min = min(lon_min, np.min(lons_msg, initial=100.))
            lon_max = max(lon_max, np.max(lons_msg, initial=-100.))
            lat_min = min(lat_min, np.min(lats_msg, initial=100.))
            lat_max = max(lat_max, np.max(lats_msg, initial=-100.))

        if plotmtg:
            lons_mtg = np.array(df_mtg["Lon"])
            lats_mtg = np.array(df_mtg["Lat"])
            # Assume lon/lat min max are same for MSG and MTG. TODO: Could add error catching to make sure this is true
            #lon_min = min(lon_min, lons_mtg.min())
            #lon_max = max(lon_max, lons_mtg.max())
            #lat_min = min(lat_min, lats_mtg.min())
            #lat_max = max(lat_max, lats_mtg.max())

            lon_min = min(lon_min, np.min(lons_mtg, initial=100.))
            lon_max = max(lon_max, np.max(lons_mtg, initial=-100.))
            lat_min = min(lat_min, np.min(lats_mtg, initial=100.))
            lat_max = max(lat_max, np.max(lats_mtg, initial=-100.))

        # Assume time string is same for MSG and MTG. TODO: Could add error catching to make sure this is true
        timestr = msgcsv.split("_")[1]
        if plotmtg:
            ax.scatter(lons_mtg, lats_mtg, s=1, alpha=0.5, color='red')
            legend_elements.append(Patch(facecolor='red', edgecolor='red', label='MTG'))
        if plotmsg:
            ax.scatter(lons_msg, lats_msg, s=1, alpha=0.5, color='blue')
            legend_elements.append(Patch(facecolor='blue', edgecolor='blue', label='MSG'))

        plt.xlim(lon_range[0], lon_range[1])
        plt.ylim(lat_range[0], lat_range[1])

        plt.legend(handles=legend_elements, title='Satellite', loc='lower center', ncol=2)
        values = [str(val)[3:] if 'df_' in str(val) else str(val) for val in constraints['values']]
        constraintstrs = [f"{var} {op} {val}" for var, op, val in zip(constraints['variables'], constraints['operators'], values)]
        constraintstr = "; ".join(constraintstrs)
        satstr = ""
        if plotmtg and plotmsg:
            satstr = "_MTG_MSG"
        elif plotmtg:
            satstr = "_MTG"
        elif plotmsg:
            satstr = "_MSG"

        plt.title(f'Pixels passing {constraintstr}')
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        latstr = '('+str(round(lat_min,1))+','+str(round(lat_max,1))+')'
        lonstr = '('+str(round(lon_min,1))+','+str(round(lon_max,1))+')'
        plt.text(xlim[0] + 0.02*(xlim[1]-xlim[0]), ylim[1]*0.9975, f"Plot Region: {region}", color='black', va='top', ha='left')
        plt.text(xlim[0] + 0.02*(xlim[1]-xlim[0]), ylim[1]*0.9910, f"Lat/Lon: {latstr}/{lonstr}",color='black', va='top', ha='left')
        plt.text(xlim[0] + 0.02*(xlim[1]-xlim[0]), ylim[1]*0.9845, f"Percentage Passed: {passed_str}",color='black', va='top', ha='left')
        plt.text(xlim[0] + 0.02*(xlim[1]-xlim[0]), ylim[1]*0.9780, f"Time: {timestr}",color='black', va='top', ha='left')
        #outname = f"grid_Median_VA_conf_4_{master_csv_file.split('.csv')[0]}_{str(lon_range[0])}_{str(lon_range[1])}_{str(lat_range[0])}_{str(lat_range[1])}.png"
        constraintstrs_output = [f"{var}_{op_map_str[op]}_{val}" for var, op, val in zip(constraints['variables'], constraints['operators'], values)]
        constraintstr_output = "_".join(constraintstrs_output)
        outname = f"grid_{constraintstr_output}_{region}_{timestr}{satstr}.png"
        print(f"Saving fig to {outdir+'/'+outname}")
        ax.coastlines()
        plt.savefig(outdir+'/'+outname)
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Create MSG/MTG retrieval code plots")
    parser.add_argument('--indir', type=str, default='/home/users/benjamin.honan/Work/analyse_csv/csv_files/')
    parser.add_argument('--outdir', type=str, default='/home/users/benjamin.honan/Work/analyse_csv/plots/')
    parser.add_argument('--args', type=str, default='')
    parser.add_argument('--master_csv_file', type=str, default='MSG_MTG_pairs.csv')
    parser.add_argument('--plot_points', action="store_true")
    parser.add_argument('--plot_btd', action="store_true")
    parser.add_argument('--plot_beta_masks', action="store_true")
    parser.add_argument('--recreate_csv', action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.plot_points:
        #plot_latlon_points(args.indir, args.outdir, args.master_csv_file, constraints = {'variables':["BTD2_conf"], 'operators':['<='], 'values':['df_c4']})
        #plot_latlon_points(args.indir, args.outdir, args.master_csv_file, constraints = {'variables':["BMCon"], 'operators':['=='], 'values':['T']})
        #plot_latlon_points(args.indir, args.outdir, args.master_csv_file, constraints = {'variables':["BTD2_conf", "BMCon"], 'operators':['<=', '=='], 'values':['df_c4','T']})
        plot_latlon_points(args.indir, args.outdir, args.master_csv_file, constraints = {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[4]})
        plot_latlon_points(args.indir, args.outdir, args.master_csv_file, constraints = {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[4]})
        #plot_latlon_points(args.indir, args.outdir, args.master_csv_file, constraints = {'variables':["BTD2_conf", "BMCon"], 'operators':['<=', '=='], 'values':['df_c4','T']}, plotonly='msg')
        #plot_latlon_points(args.indir, args.outdir, args.master_csv_file, constraints = {'variables':["BTD2_conf", "BMCon"], 'operators':['<=', '=='], 'values':['df_c4','T']}, plotonly='mtg')
    elif args.plot_btd:
        #make_btd_plots(args.indir, ["UK"])
        make_btd_plots(args.indir, args.outdir, args.master_csv_file)
    elif args.plot_beta_masks:
        plot_beta_masks(args.indir, args.outdir, args.master_csv_file, args.args)
    else:
        analyse_csv_nearestneighbors(args.indir, args.outdir, args.master_csv_file, args.recreate_csv)

