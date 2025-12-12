from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def iter_loaded(datasets):
	"""
	Generator that yields each Dataset after loading.
	Automatically unloads the previous dataset when moving to the next
	(or if the caller raises/breaks).
	"""
	for ds in datasets:
		ds._load()
		try:
			yield ds
		finally:
			ds._unload()

def setup_figure(title=None, xlim=None, ylim=None, plotstr=None):
	plt.figure()
	ax = plt.axes(projection=ccrs.PlateCarree())
	plt.xlim(xlim[0], xlim[1])
	plt.ylim(ylim[0], ylim[1])
	ax.coastlines()
	plt.title(title)

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

