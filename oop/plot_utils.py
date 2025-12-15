from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Rectangle, Patch
import matplotlib.colors as mcolors

def setup_figure(title=None, xlim=None, ylim=None, plotstr=None):
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    ax.coastlines()
    plt.title(title)
    if plotstr:
        plt.text(xlim[0] + 0.02*(xlim[1]-xlim[0]), ylim[1] - (ylim[1]-ylim[0])*0.02, plotstr, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False)
    gl.xlines = True   
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14} 
    gl.ylabel_style = {'size': 14}    
    return ax

def save_plots(outdir, outname, savesvgpdf=False):
    outname = Path(outname).stem
    extensions = ['.png']
    if savesvgpdf:
        extensions += ['.pdf', '.svg']
    
    for ext in extensions:
        plotpath = outdir / Path(outname+ext)
        plt.savefig(plotpath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plotpath}")

def plot_btd_hist(btds, xlabel, ylabel, title, xmin=0, xmax=0, nbins=50, plotc4=False, plotc3=False, plotc1=False, plotBTD3thresh=False, savehist=True, outdir='/home/users/benjamin.honan/Work/analyse_csv/plots/', outname="btdhist.png", latlonstr="", regionstr="", timestr="", conf_cut=""):

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
        elif plotc3:
            mtg_perc_below_c3 = (sum(1 for btd in btds[0] if btd < -0.88)/len(btds[0])) * 100
            msg_perc_below_c3 = (sum(1 for btd in btds[1] if btd < -1.0)/len(btds[1])) * 100
            mtg_perc_above_c3 = 100. - mtg_perc_below_c3
            msg_perc_above_c3 = 100. - msg_perc_below_c3
            mtg_perc_below_btd_cutoff = (sum(1 for btd in btds[0] if btd < -0.1)/len(btds[0])) * 100
            msg_perc_below_btd_cutoff = (sum(1 for btd in btds[1] if btd < -0.1)/len(btds[1])) * 100
        elif plotc1:
            mtg_perc_below_c1 = (sum(1 for btd in btds[0] if btd < -2.06)/len(btds[0])) * 100
            msg_perc_below_c1 = (sum(1 for btd in btds[1] if btd < -2.00)/len(btds[1])) * 100
        if plotBTD3thresh:
            mtg_perc_above_btd3 = (sum(1 for btd in btds[0] if btd > 1.5)/len(btds[0])) * 100
            msg_perc_above_btd3 = (sum(1 for btd in btds[1] if btd > 1.5)/len(btds[1])) * 100

    #plt.figure()
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.hist(btds, bins=nbins, range=(xmin, xmax), density=True, color=colors, label=labels, edgecolor='black')
    plt.legend(title="Satellite Type")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Plot vertical lines for c1, c3, c4 thresholds (assume -1 for now) with visible labels
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plt.ylim(ylim[0], ylim[1] * 1.45)  # Increase y-limit by 45% to make space for text
    ylim = plt.gca().get_ylim()
    timestry=0.74
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
        plt.axvline(-2.0, color='purple', linestyle='--')
        plt.text(-1.80, ylim[1]*0.65, 'C1 MSG', color='purple', rotation=90, va='top', ha='right', backgroundcolor='white')
        plt.axvline(-2.06, color='green', linestyle='--')
        plt.text(-2.12, ylim[1]*0.65, 'C1 MTG', color='green', rotation=90, va='top', ha='right', backgroundcolor='white')
        textstrmtg = f"MTG % below C1: {mtg_perc_below_c1:.1f}%"
        textstrmsg = f"MSG % below C1: {msg_perc_below_c1:.1f}%"
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
    elif plotc3:
        plt.axvline(-1.0, color='purple', linestyle='--')
        plt.text(-0.96, ylim[1]*0.65, 'C3 MSG', color='purple', rotation=90, va='top', ha='right', backgroundcolor='white')
        plt.axvline(-0.88, color='green', linestyle='--')
        plt.text(-0.84, ylim[1]*0.65, 'C3 MTG', color='green', rotation=90, va='top', ha='right', backgroundcolor='white')
        plt.axvline(-0.1, color='red', linestyle='--')
        plt.text(-0.06, ylim[1]*0.85, 'BTD Cutoff', color='red', rotation=90, va='top', ha='right', backgroundcolor='white')
        if "3" in conf_cut:
            textstrmtg = f"MTG % above C3: {mtg_perc_above_c3:.1f}%"
            textstrmsg = f"MSG % above C3: {msg_perc_above_c3:.1f}%"
            textstrmtg2 = f"MTG % below BTD Cutoff: {mtg_perc_below_btd_cutoff:.1f}%"
            textstrmsg2 = f"MSG % below BTD Cutoff: {msg_perc_below_btd_cutoff:.1f}%"
        else:
            textstrmtg = f"MTG % below C3: {mtg_perc_below_c3:.1f}%"
            textstrmsg = f"MSG % below C3: {msg_perc_below_c3:.1f}%"
            textstrmtg2 = ""
            textstrmsg2 = ""
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
        if textstrmtg2:
            timestry=0.62
            plt.text(
                xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*0.74,
                textstrmtg2,
                ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            plt.text(
                xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*0.68,
                textstrmsg2,
                ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
    elif plotc4:
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
        plt.text(1.5, ylim[1]*0.8, 'BTD3 Thresh', color='purple', rotation=90, va='top', ha='right', backgroundcolor='white')
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
            xlim[1] - 0.55*(xlim[1]-xlim[0]), ylim[1]*timestry,
            timestr,
            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    if savehist:
        save_plots(outdir, outname)

    return(fig, ax)

def plot_latlon_points_dataset(dataset, plotX=True, plotbox=False, boundsfrommeta=True, plotBTD3=False, custombounds=None):

    df_msg, df_mtg = dataset.data_msg, dataset.data_mtg
    if plotBTD3:
        BTD3_msg = np.array(df_msg_matches["VolcanicAsh_BTD3"])
        BTD3_mtg = np.array(df_mtg_matches["VolcanicAsh_BTD3"])
         
        # Create 2D histogram
        x_bins = np.linspace(lons_msg.min(), lons_msg.max(), 50)
        y_bins = np.linspace(lats_msg.min(), lats_msg.max(), 20)
            
        # Create histogram
        H_msg, xedges_msg, yedges_msg = np.histogram2d(lons_msg, lats_msg, bins=[x_bins, y_bins], weights=BTD3_msg)
        H_mtg, xedges_mtg, yedges_mtg = np.histogram2d(lons_mtg, lats_mtg, bins=[x_bins, y_bins], weights=BTD3_mtg)

        # Mask zero bins
        H_msg = np.ma.masked_where(H_msg == 0., H_msg)
        H_mtg = np.ma.masked_where(H_mtg == 0., H_mtg)
            
        # Create meshgrid for pcolormesh
        X_msg, Y_msg = np.meshgrid(xedges_msg, yedges_msg)
        X_mtg, Y_mtg = np.meshgrid(xedges_mtg, yedges_mtg)

        londiff_msg = abs(dataset.lonmax_msg - dataset.lonmin_msg)
        latdiff_msg = abs(dataset.latmax_msg - dataset.latmin_msg)
        xlim = (lonmin_msg - londiff_msg*0.5, lonmax_msg + londiff_msg*0.5)
        ylim = (latmin_msg - latdiff_msg*0.5, latmax_msg + latdiff_msg*0.5)

        ax_msg = setup_figure(title="MSG BTD3 Values", xlim=xlim, ylim=ylim)

        colors_lowbtd = mcolors.LinearSegmentedColormap.from_list('lowbtd', ['red','yellow'])(np.linspace(0, 1, 256))
        colors_highbtd = plt.cm.Blues(np.linspace(0, 1, 256))

        colors_btd = np.vstack((colors_lowbtd, colors_highbtd))
        btd_map = mcolors.LinearSegmentedColormap.from_list('btd_map', colors_btd)
        btd_map.set_bad(alpha=0)
        divnorm = mcolors.TwoSlopeNorm(vmin=-5., vcenter=1.5, vmax=5.)

        # Determine the data range to center the colormap at zero
        vmin = H_msg.T.min()
        vmax = H_msg.T.max()
        # Make the color scale symmetric around zero
        vlim = max(abs(vmin), abs(vmax))

        pcm = ax.pcolormesh(
            X_msg, Y_msg, H_msg.T,
            cmap=btd_map,
            shading='auto',
            transform=ccrs.PlateCarree(),
            norm=divnorm
        )

        plt.colorbar(pcm, label='BTD3 Value')
        plt.savefig(outdir+'/'+f"MSG_BTD3_map_{timestr_msg}.png")
        plt.show()
        plt.close()

        # Plot MTG
        londiff_mtg = abs(dataset.lonmax_mtg - dataset.lonmin_mtg)
        latdiff_mtg = abs(dataset.latmax_mtg - dataset.latmin_mtg)
        xlim = (lonmin_mtg - londiff_mtg*0.5, lonmax_msg + londiff_mtg*0.5)
        ylim = (latmin_mtg - latdiff_mtg*0.5, latmax_msg + latdiff_mtg*0.5)

        ax_mtg = setup_figure(title="MTG BTD3 Values", xlim=xlim, ylim=ylim)

        # Determine the data range to center the colormap at zero
        vmin = H_mtg.T.min()
        vmax = H_mtg.T.max()
        # Make the color scale symmetric around zero
        vlim = max(abs(vmin), abs(vmax))

        pcm = ax.pcolormesh(
            X_mtg, Y_mtg, H_mtg.T,
            cmap=btd_map,
            shading='auto',
            transform=ccrs.PlateCarree(),
            norm=divnorm
        )

        plt.colorbar(pcm, label='BTD3 Value')

        return (ax_msg, ax_mtg)

    else:
        if custombounds:
            try:
                xlim=custombounds[0]
                ylim=custombounds[1]
            except:
                return ValueError(f"Custom bounds incorrect format. Expected list or tuple with two two-element elements. Got {str(custombounds)}.")
        else:
            if boundsfrommeta:
                xlim = (dataset.lon_range[0], dataset.lon_range[1])
                ylim = (dataset.lat_range[0], dataset.lat_range[1])
            else:
                londiff = abs(dataset.lonmax_mtg - dataset.lonmin_mtg)
                latdiff = abs(dataset.latmax_mtg - dataset.latmin_mtg)
                multiplier = 1.5
                if dataset.n_msg + dataset.n_mtg < 100:
                    multiplier = 5.0
                xlim = (dataset.lonmin_mtg - londiff*multiplier, dataset.lonmax_mtg + londiff*multiplier)
                ylim = (dataset.latmin_mtg - latdiff*multiplier, dataset.latmax_mtg + latdiff*multiplier)

        ax = setup_figure(xlim=xlim, ylim=ylim)
        ax.scatter(dataset.lons_msg, dataset.lats_msg, s=1, alpha=0.5, color='blue')
        ax.scatter(dataset.lons_mtg, dataset.lats_mtg, s=1, alpha=0.5, color='red')

        legend_elements = []
        legend_elements.append(Patch(facecolor='blue', edgecolor='blue', label='MSG'))
        legend_elements.append(Patch(facecolor='red', edgecolor='red', label='MTG'))
        plt.legend(handles=legend_elements, title='Satellite', ncol=2)

        if plotbox:
            #lat_min, lat_max = 11.96,14.29 #lats_msg.min(), lats_msg.max()
            #lon_min, lon_max = 40.75,45.62 #lons_msg.min(), lons_msg.max()
            lat_min, lat_max = 10,25 #lats_msg.min(), lats_msg.max()
            lon_min, lon_max = 35,55 #lons_msg.min(), lons_msg.max()
            # Add rectangle to show the boundaries
            rect = Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linewidth=2,
                edgecolor='black',
                facecolor='none',
                linestyle='-',
                transform=ccrs.PlateCarree()
            )
            ax.add_patch(rect)
        
        if plotX:
            # Add X marker at volcano
            marker_lat = 13.51  
            marker_lon = 40.72  

            ax.scatter(
                marker_lon, marker_lat,
                marker='x',
                s=20,  # Size of the X
                c='black',  # Color
                linewidths=1,  # Thickness of the X
                transform=ccrs.PlateCarree(),
                zorder=10  # Ensure it's plotted on top
            )

        return ax

