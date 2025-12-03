import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle, Patch
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import config
import filters

class Dataset:

    '''
    Usage:

    for dataset in self.datasets:
        df_msg, df_mtg = dataset.load_data()
        meta = dataset.metadata
        region = meta['region']
        ...

    '''

    def __init__(self, file_path_msg, file_path_mtg, metadata):
        self.file_path_msg = file_path_msg
        self.file_path_mtg = file_path_mtg
        self.metadata = metadata
        self.metadata.update({'time_msg':self.file_path_msg.split("_")[1]})
        self.metadata.update({'time_mtg':self.file_path_msg.split("_")[1]})
        self._data_msg = None  # Lazy load
        self._data_mtg = None 
        self.lons_msg = None 
        self.lats_msg = None
        self.lons_mtg = None
        self.lats_mtg = None
        self.latmin_msg = -90.
        self.latmax_msg = 90.
        self.latmin_mtg = -90.
        self.latmax_mtg = 90.

    def load_data(self):
        if self._data_msg is None:
            self._data_msg = pd.read_csv(self.file_path_msg)
        if self._data_mtg is None:
            self._data_mtg = pd.read_csv(self.file_path_mtg)
        if self.lons_msg == self.lats_msg == self.lons_mtg == self.lats_mtg == None:
            self.lons_msg = np.array(self._data_msg["Lon"])
            self.lats_msg = np.array(self._data_msg["Lat"])
            self.lons_mtg = np.array(self._data_mtg["Lon"])
            self.lats_mtg = np.array(self._data_mtg["Lat"])
            self.lonmin_msg = self.lons_msg.min()
            self.lonmax_msg = self.lons_msg.max()
            self.lonmin_mtg = self.lons_mtg.min()
            self.lonmax_mtg = self.lons_mtg.max()
        return (self._data_msg, self._data_mtg)

class Plotter:
    def __init__(self, indir, outdir, master_csv):
        self.indir = indir
        self.outdir = outdir
        self.outdir_matches = self.indir+"/matches/"
        df = pd.read_csv(master_csv)
        self.datasets = [
            Dataset(row['csv_msg'], row['csv_mtg'], row.to_dict()) for _, _, row in df.iterrows()
        ]

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

        for dataset in self.datasets:
            df_msg, df_mtg = dataset.load_data()
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

            if len(coords_tree) != 0:
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
                mtg_libmask, msg_libmask = mtg_match["BMLib"], msg_match["BMLib"]
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
                    if failc1:
                        retrievalcode = RetrievalCode("conf7_c1")
                    else:
                        retrievalcode = RetrievalCode("conf7_other")
                elif mtg_conf == 4 and msg_conf == 1:
                    failc4 = msg_btd2 > msg_match["c4"]
                    failconmask = msg_conmask == 'F'
                    if failc4 and failconmask:
                        retrievalcode = RetrievalCode("conf4_c4_conmask_msgconf1")
                    elif failc4:
                        retrievalcode = RetrievalCode("conf4_c4_msgconf1")
                    elif failconmask:
                        retrievalcode = RetrievalCode("conf4_conmask_msgconf1")
                    else:
                        retrievalcode = RetrievalCode("conf4_other_msgconf1")
                elif mtg_conf == 3 and msg_conf == 1:
                    failc3 = msg_btd2 <= msg_match["c3"]
                    failbtd3 = msg_btd3 > msg_match["BTD3thresh"]
                    failbtdcutoff = msg_btd2 > -0.1
                    failconmask = msg_conmask == 'F'
                    if failbtdcutoff and failbtd3 and failconmask:
                        retrievalcode = RetrievalCode("conf3_btdcutoff_btd3_conmask_msgconf1")
                    elif failbtdcutoff and failbtd3:
                        retrievalcode = RetrievalCode("conf3_btdcutoff_btd3_msgconf1")
                    elif failbtdcutoff and failconmask:
                        retrievalcode = RetrievalCode("conf3_btdcutoff_conmask_msgconf1")
                    if failc3 and failbtd3 and failconmask:
                        retrievalcode = RetrievalCode("conf3_c3_btd3_conmask_msgconf1")
                    elif failc3 and failbtd3:
                        retrievalcode = RetrievalCode("conf3_c3_btd3_msgconf1")
                    elif failc3 and failconmask:
                        retrievalcode = RetrievalCode("conf3_c3_conmask_msgconf1")
                    elif failbtd3 and failconmask:
                        retrievalcode = RetrievalCode("conf3_btd3_conmask_msgconf1")
                    elif failbtd3:
                        retrievalcode = RetrievalCode("conf3_btd3_msgconf1")
                    elif failconmask:
                        retrievalcode = RetrievalCode("conf3_conmask_msgconf1")
                    elif failbtdcutoff:
                        retrievalcode = RetrievalCode("conf3_btdcutoff_msgconf1")
                    elif failc3:
                        retrievalcode = RetrievalCode("conf3_c3_msgconf1")
                    else:
                        retrievalcode = RetrievalCode("conf3_other_msgconf1")
                elif mtg_conf == 1 and msg_conf == 1:
                    retrievalcode = RetrievalCode("conf1_msgconf1")
                elif mtg_conf == 7 and msg_conf == 1:
                    failc1 = msg_btd2 > msg_match["c1"]
                    if failc1:
                        retrievalcode = RetrievalCode("conf7_c1_msgconf1")
                    else:
                        retrievalcode = RetrievalCode("conf7_other_msgconf1")
                elif mtg_conf == 4 and msg_conf == 4:
                    retrievalcode = RetrievalCode("conf4_msgconf4")
                elif mtg_conf == 4 and msg_conf == 2:
                    failc4 = msg_btd2 > msg_match["c4"]
                    failconmask = msg_conmask == 'F'
                    if failc4 and failconmask:
                        retrievalcode = RetrievalCode("conf4_c4_conmask_msgconf2")
                    elif failc4:
                        retrievalcode = RetrievalCode("conf4_c4_msgconf2")
                    elif failconmask:
                        retrievalcode = RetrievalCode("conf4_conmask_msgconf2")
                    else:
                        retrievalcode = RetrievalCode("conf4_other_msgconf2")
                elif msg_conf == 4 and mtg_conf == 1:
                    failc4 = mtg_btd2 > mtg_match["c4"]
                    failconmask = mtg_conmask == 'F'
                    if failc4 and failconmask:
                        retrievalcode = RetrievalCode("conf1_c4_conmask_msgconf4")
                    elif failc4:
                        retrievalcode = RetrievalCode("conf1_c4_msgconf4")
                    elif failconmask:
                        retrievalcode = RetrievalCode("conf1_conmask_msgconf4")
                    else:
                        retrievalcode = RetrievalCode("conf1_other_msgconf4")
                elif mtg_conf == 7 and msg_conf == 7:
                    retrievalcode = RetrievalCode("conf7_msgconf7")
                elif mtg_conf == 6 and msg_conf == 3:
                    retrievalcode = RetrievalCode("conf6_msgconf3")
                elif mtg_conf == 3 and msg_conf == 4:
                    retrievalcode = RetrievalCode("conf3_msgconf4")
                elif mtg_conf == 3 and msg_conf == 3:
                    retrievalcode = RetrievalCode("conf3_msgconf3")
                elif mtg_conf == 0:
                    retrievalcode = RetrievalCode("noret")
                else:
                    retrievalcode = RetrievalCode("other")
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
