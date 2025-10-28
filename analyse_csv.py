import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.spatial import KDTree
from enum import Enum

parent_csv_path = "/home/users/benjamin.honan/Work/analyse_csv/csv_files/"
file_msg = "MSG_202510120200_vars.csv"
file_mtg = "MTG_202510120200_vars.csv"
#coord_msg = (52.13277,-12.73369)
#coord_mtg = (52.13658,-12.73348)

#lat_range = (52,53)
#lon_range = (-12,-13)
lat_range = (49,57)
lon_range = (-16,-10)

class RetrievalCode(Enum):
    CONF4_CONMASK = "conf4_conmask" # Detected conf 4 retrievals in MTG and none in MSG, and reason MSG fails is the conservative beta mask
    CONF4_C4 = "conf4_c4" # Detected conf 4 retrievals in MTG and none in MSG, and reason MSG fails is the BTD2 C4 threshold
    CONF4_C4_CONMASK = "conf4_c4_conmask" # Detected conf 4 retrievals in MTG and none in MSG, and reason MSG fails is the BTD2 C4 threshold and the beta mask
    CONF4_OTHER = "conf4_other" # Detected conf 4 retrievals in MTG and none in MSG, and reason MSG fails is something else
    NORET = "noret" # No retrievals in either MSG and MTG
    BOTH = "both" # Retrieved ash in both MSG and MTG
    OTHER = "other" # Anything else

def analyse_csv(parent_csv_path):

    file_path_msg = parent_csv_path + file_msg
    file_path_mtg = parent_csv_path + file_mtg
    
    # Load the CSV file into a DataFrame
    df_msg = pd.read_csv(file_path_msg)
    df_mtg = pd.read_csv(file_path_mtg)

    coords_msg = tuple(zip(np.array(df_msg['Lat']), np.array(df_msg['Lon'])))
    coords_mtg = tuple(zip(np.array(df_mtg['Lat']), np.array(df_mtg['Lon'])))

    loc_coord_msg = [i for i, c in enumerate(coords_msg) if c == coord_msg]
    loc_coord_mtg = [i for i, c in enumerate(coords_mtg) if c == coord_mtg]
    idx_msg = loc_coord_msg[0]
    idx_mtg = loc_coord_mtg[0]

    output_file = file_path + "{}_{}_comparison.txt".format(file_msg.rsplit("_vars.csv")[0], file_mtg.rsplit("_vars.csv")[0])
    with open(output_file, "w") as f:
        f.write(f"MSG coordinate: {coord_msg}\n")
        f.write(f"MTG coordinate: {coord_mtg}\n")
        f.write(f"{'Variable':<25}{'MSG':>20}{'MTG':>20}\n")
        for col in df_msg.columns:
            val_msg = df_msg.iloc[idx_msg][col]
            val_mtg = df_mtg.iloc[idx_mtg][col] if col in df_mtg.columns else "N/A"
            f.write(f"{col:<25}{str(val_msg):>20}{str(val_mtg):>20}\n")

    print(f"Comparison written to {output_file}")

def analyse_csv_nearestneighbors(parent_csv_path):

    threshold = 0.01  # degrees

    file_path_msg = parent_csv_path + file_msg
    file_path_mtg = parent_csv_path + file_mtg
    
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

    output_file = parent_csv_path + "{}_{}_comparison_nn.txt".format(file_msg.rsplit("_vars.csv")[0], file_mtg.rsplit("_vars.csv")[0])
    # Open the file in write mode before the loop to overwrite it each time
    with open(output_file, "w") as f:
        pass  # Just to clear the file at the start

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
            with open(output_file, "a") as f:
                f.write(f"{'Satellite':<25}{sat_tree:>20}{sat_search:>20}\n")
                for col in df_tree.columns:
                    val_tree = var_tree[col]
                    val_search = var_search[col] if col in df_search.columns else "N/A"
                    f.write(f"{col:<25}{str(val_tree):>20}{str(val_search):>20}\n")
                f.write("\n")

    retrievalcodes = []
    msg_matches = []
    for pair in search_matches:
        mtg_match, msg_match = pair[0], pair[1]
        msg_matches.append(msg_match)
        mtg_conf, msg_conf = mtg_match["PreFilter_VA_Confidence"], msg_match["PreFilter_VA_Confidence"]
        mtg_btd2, msg_btd2 = mtg_match["BTD2_conf"], msg_match["BTD2_conf"]
        mtg_conmask, msg_conmask = mtg_match["BMCon"], msg_match["BMCon"]
        if mtg_conf == 4 and msg_conf == 0:
            failc4 = msg_btd2 > msg_match[" c4"]
            failmask = msg_conmask == 'F'
            if failc4 and failmask:
                retrievalcode = RetrievalCode("conf4_c4_conmask")
            elif failc4:
                retrievalcode = RetrievalCode("conf4_c4")
            elif failmask:
                retrievalcode = RetrievalCode("conf4_conmask")
            else:
                retrievalcode = RetrievalCode("conf4_other")
        elif mtg_conf == 0 and msg_conf == 0:
            retrievalcode = RetrievalCode("noret")
        elif mtg_conf > 0 and msg_conf > 0:
            retrievalcode = RetrievalCode("both")
        else:
            retrievalcode = RetrievalCode("other")
        retrievalcodes.append(retrievalcode)
    
    df_msg_matches = pd.DataFrame(msg_matches).reset_index(drop=True)
    df_msg_matches['retrieval_code'] = pd.Series(retrievalcodes)





    '''

    MTG_CONF4_CONMASK = "mtg_conf4_conmask" # Detected conf 4 retrievals in MTG and none in MSG, and reason is the conservative beta mask
    MTG_CONF4_BTD2_C4 = "mtg_conf4_btd2_c4" # Detected conf 4 retrievals in MTG and none in MSG, and reason is the BTD2 C4 threshold
    MTG_CONF4_BTD2_C4_CONMASK = "mtg_conf4_btd2_c4_conmask" # Detected conf 4 retrievals in MTG and none in MSG, and reason is the BTD2 C4 threshold and the beta mask
    MTG_CONF4_OTHER = "mtg_conf4_other" # Detected conf 4 retrievals in MTG and none in MSG, and reason is something else
    MTG_NORET = "noret" # No retrievals in either MSG and MTG
    BOTH = "both" # Retrieved ash in both MSG and MTG
    OTHER = "other" # Anything else

    Lat                        51.46804
Lon                       -12.99969
BTD2_conf                  -0.25195
Median_VA_Confidence              1
PreFilter_VA_Confidence           0
BMLib                             T
BMCon                             T
VolcanicAsh_BTD3            2.89111
c1                            -2.06
c2                            -1.47
 c3                           -0.88
 c4                           -0.29
 BTD3thresh                     1.5
 aa                            -0.4
 bb                            -0.4
 cc                             1.9
    '''

def plot_latlon_points(file_path_1, file_path_2):

    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)
    lats1 = np.array(df1["Lat"])
    lons1 = np.array(df1["Lon"])
    lats2 = np.array(df2["Lat"])
    lons2 = np.array(df2["Lon"])

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.scatter(lons1, lats1, s=1)
    ax.scatter(lons2, lats2, s=1, color="green")
    plt.xlim(lon_range[0], lon_range[1])
    plt.ylim(lat_range[0], lat_range[1])
    ax.coastlines()
    plt.show()



if __name__ == "__main__":
    analyse_csv_nearestneighbors(parent_csv_path)
    #msg_fullpath = parent_csv_path + file_msg
    #mtg_fullpath = parent_csv_path + file_mtg
    #plot_latlon_points(msg_fullpath, mtg_fullpath, )

