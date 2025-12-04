
def find_nn(dataset1, dataset2, threshold = 0.):
    n_df1, n_df2 = len(df1), len(df2)
    build_df1_tree = True if n_df1 < n_df2 else False

    # Define the df used to build the tree as the smallest one to optimise speed (building the tree is O(n log n))
    df_tree, df_search = (df_msg, df_mtg) if build_msg_tree else (df_mtg, df_msg)
    search_matches = []

    if threshold:
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
    else: #Find nearest neighbours if threshold not given
        tree = KDTree(coords_tree)
        nearest_dist, nearest_ind = tree.query(coords_search, k=1)
        for iind, index in enumerate(nearest_ind):
            mtg_matches.append(df_tree.iloc[index])
