import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle, Patch
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

class Plotter:
    def __init__(self, indir, outdir, show_plots=False, region="", time="", constraint_cut=""):
        self.indir = indir
        self.outdir = outdir
        self.show = show_plots
        self.region = region
        self.time = time
        self.cut = constraint_cut

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

    def add_box(self, ax, lon_min, lon_max, lat_min, lat_max):
        ax.add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                               linewidth=2, edgecolor='black', facecolor='none',
                               linestyle='-', transform=ccrs.PlateCarree()))

    def add_marker(self, ax, lat, lon, color='black', size=20):
        ax.scatter(lon, lat, marker='x', s=size, c=color, linewidths=1,
                   transform=ccrs.PlateCarree(), zorder=10)

    def save(self, fig, name):
        path = f"{self.outdir}/{name}"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {path}")
        if self.show: plt.show()
        plt.close(fig)

    def detection_map(self, df, labels, lon_range, lat_range, title):
        fig = plt.figure()
        ax = self.new_geo_axes()
        for label, subset, color in labels:  # [(label, df_subset, color), ...]
            ax.scatter(subset['Lon'], subset['Lat'], s=2, alpha=0.8, label=label, color=color)
        plt.legend(title='Detection Type', fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(lon_range); plt.ylim(lat_range)
        plt.title(title)
        self.save(fig, f"{self.region}_{self.time}_detection_type_map.png")

    def freq_hist(self, labels_counts, meta_text):
        fig = plt.figure()
        indices = list(range(1, len(labels_counts)+1))
        labels = [lc[0] for lc in labels_counts]
        counts = [lc[1] for lc in labels_counts]
        bars = plt.bar(indices, counts)
        plt.xticks(indices)
        plt.ylabel("Count"); plt.xlabel("Detection Type"); plt.title("Detection Type Frequency")
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), str(count), ha='center', va='bottom', fontsize=9)
        legend_handles = [Patch(facecolor=bar.get_facecolor(), label=f"{i}: {label}") for i, label, bar in zip(indices, labels, bars)]
        plt.legend(handles=legend_handles, title="Detection Type", fontsize='small', loc='upper right')
        plt.text(0.02, 0.95, meta_text, transform=plt.gca().transAxes,
                 ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        self.save(fig, f"{self.region}_{self.time}_detection_type_histogram.png")