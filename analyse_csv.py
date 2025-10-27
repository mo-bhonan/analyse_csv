import pandas as pd
import numpy as np

file_path = "/home/users/benjamin.honan/Work/analyse_csv/csv_files/"
file_msg = "MSG_202510120200_vars.csv"
file_mtg = "MTG_202510120200_vars.csv"
coord_msg = (52.13277,-12.73369)
coord_mtg = (52.13658,-12.73348)

def analyse_csv(file_path):

    file_path_msg = file_path + file_msg
    file_path_mtg = file_path + file_mtg
    
    # Load the CSV file into a DataFrame
    df_msg = pd.read_csv(file_path_msg)
    df_mtg = pd.read_csv(file_path_mtg)

    coords_msg = tuple(zip(np.array(df_msg['Lat']), np.array(df_msg['Lon'])))
    coords_mtg = tuple(zip(np.array(df_mtg['Lat']), np.array(df_mtg['Lon'])))

    loc_coord_msg = [i for i, c in enumerate(coords_msg) if c == coord_msg]
    if len(loc_coord_msg) == 0:
        raise ValueError(f"No match found for coord_msg {coord_msg} in MSG coordinates.")
    elif len(loc_coord_msg) > 1:
        raise ValueError(f"Multiple matches found for coord_msg {coord_msg} in MSG coordinates: {loc_coord}")
    # If exactly one match, proceed as normal
    idx_msg = loc_coord_msg[0]

    loc_coord_mtg = [i for i, c in enumerate(coords_mtg) if c == coord_mtg]
    if len(loc_coord_mtg) == 0:
        raise ValueError(f"No match found for coord_mtg {coord_mtg} in MSG coordinates.")
    elif len(loc_coord_mtg) > 1:
        raise ValueError(f"Multiple matches found for coord_mtg {coord_mtg} in MSG coordinates: {loc_coord}")
    # If exactly one match, proceed as normal
    idx_mtg = loc_coord_mtg[0]

    output_file = file_path + "MSG_MTG_comparison.txt"
    with open(output_file, "w") as f:
        f.write(f"MSG coordinate: {coord_msg}\n")
        f.write(f"MTG coordinate: {coord_mtg}\n")
        f.write(f"{'Variable':<25}{'MSG':>20}{'MTG':>20}\n")
        for col in df_msg.columns:
            val_msg = df_msg.iloc[idx_msg][col]
            val_mtg = df_mtg.iloc[idx_mtg][col] if col in df_mtg.columns else "N/A"
            f.write(f"{col:<25}{str(val_msg):>20}{str(val_mtg):>20}\n")

    print(f"Comparison written to {output_file}")

if __name__ == "__main__":
    analyse_csv(file_path)
