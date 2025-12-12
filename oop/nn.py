from scipy.spatial import KDTree

def find_nn(dataset, threshold = 0., lt=True):

    build_msg_tree = True if dataset.n_msg < dataset.n_mtg else False
    # If searching using just nearest neighbours without a threshold, have to actually build the tree using the larger dataset
    if not threshold:
        build_msg_tree = not build_msg_tree
    # Define the df used to build the tree as the smallest one to optimise speed (building the tree is O(n log n))
    if build_msg_tree:
        df_tree, df_search = (dataset.data_msg, dataset.data_mtg) 
        coords_tree = tuple(zip(dataset.lats_msg, dataset.lons_msg))
        coords_search = tuple(zip(dataset.lats_mtg, dataset.lons_mtg))
    else:
        df_tree, df_search = (dataset.data_mtg, dataset.data_msg) 
        coords_tree = tuple(zip(dataset.lats_mtg, dataset.lons_mtg))
        coords_search = tuple(zip(dataset.lats_msg, dataset.lons_msg))
        
    search_matches = []
    if threshold:
        if len(coords_tree) != 0:
            for coord in coords_search:
                tree = KDTree(coords_tree)
                distance, index = tree.query(coord)
                if lt:
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
                # if want to find nearest neighbours at least a certain distance away from an element
                else:
                    if distance >= threshold:
                        idx_tree = index
                        idx_search = coords_search.index(coord)
                        var_tree = df_tree.iloc[idx_tree]
                        var_search = df_search.iloc[idx_search]
                        # Always append MTG first
                        if build_msg_tree:
                            search_matches.append((var_search,var_tree))
                        else:
                            search_matches.append((var_tree,var_search))
    else: #Find nearest neighbours if threshold not given
        tree = KDTree(coords_tree)
        nearest_dist, nearest_ind = tree.query(coords_search, k=1)
        for iind, index in enumerate(nearest_ind):
            search_matches.append(df_tree.iloc[index])

    return search_matches
