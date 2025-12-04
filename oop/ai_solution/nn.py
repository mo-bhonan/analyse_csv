import numpy as np
from scipy.spatial import KDTree

def nearest_neighbors(df_tree, df_search, k=1):
    coords_tree = coords_from_df(df_tree)
    coords_search = coords_from_df(df_search)
    tree = KDTree(coords_tree)
    dist, ind = tree.query(coords_search, k=k)
    if k == 1:
        return dist, ind
    return dist, ind[:, 0]

def match_rows(df_tree, df_search, k=1):
    dist, ind = nearest_neighbors(df_tree, df_search, k=k)
    rows = [df_tree.iloc[i] for i in (ind if k == 1 else ind[:,0])]
    return rows, dist
