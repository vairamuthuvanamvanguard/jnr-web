import threading

import numpy as np
import pandas as pd
import geopandas as gpd
import dask.array as da
from fast_histogram import histogram1d

import xarray as xr
import rasterio as rio
import rioxarray as rxr
import xrspatial as xs
from rasterio.features import rasterize
from shapely.geometry import mapping

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class UDefAllocation():  
    def __init__(self, outpath="./", ncat=30):
        self.outpath = outpath
        self.ncat = ncat 

    def chunk(self, data):
        return data.chunk(chunks={"x": self.chunksize, "y": self.chunksize}) 

    def load_data(self, fcc=None, jurisdiction=None, next_admin_boundary=None, chunksize=100):  
        self.chunksize = chunksize
        self.fcc  = rxr.open_rasterio(fcc, masked=True, chunks={"x": chunksize, "y": chunksize})
        self.fcc = self.fcc.squeeze().astype("uint8")
        self.crs  = self.fcc.rio.crs  
        self.resolution = self.fcc.rio.resolution()[0]

        #print('Ensure that FCC and jurisdiction shapefile have same CRS.')
        print('Check chunksize. It is one of the crucial parameters influencing the computational performance.')

        self.border = gpd.read_file(jurisdiction)
        self.next_admin_boundary = gpd.read_file(next_admin_boundary)
        
        if self.border.crs != self.crs:
                self.border.to_crs(self.crs, inplace=True)
                
        if self.next_admin_boundary.crs != self.crs:
                self.next_admin_boundary.to_crs(self.crs, inplace=True)

        #print('Rasterization done.')
        gt = self.fcc.rio.transform().to_gdal()
        ar_pix = -1 * gt[1] * gt[5] / 10000
        self.ar_pix = ar_pix
        return  
        
    def create_l2_raster(self, value_column='row number', l2_raster_name="l2_raster.tif", driver="GTiff", compress="LZW", tiled=True, lock=None):
        self.l2_raster_path = self.outpath + l2_raster_name
        
        # Initialize an empty array to store the rasterized values
        rasterized_values = None

        # Iterate over each feature in the GeoDataFrame
        for idx, row in self.next_admin_boundary.iterrows():
                # Extract the geometry of the feature
                geometry = row.geometry

                # Extract the value for this feature from the specified column
                value = row[value_column]
            
                # Rasterize the geometry into a mask
                mask = rasterize(
                    [(mapping(geometry), value)],
                    out_shape=self.fcc.shape[-2:],  # Shape of the template raster
                    transform=self.fcc.rio.transform(),
                    fill=0,
                    dtype='uint8'  # Data type of the rasterized values
                )

                # Merge the rasterized values
                if rasterized_values is None:
                    rasterized_values = mask
                    
                else:
                    rasterized_values[mask != 0] = mask[mask != 0]

        # Save the rasterized values to the output raster
        output_dataarray = self.fcc.copy(data=rasterized_values).astype("uint8")
        output_dataarray.rio.to_raster(self.l2_raster_path, driver=driver, compress=compress, tiled=tiled, lock=lock)

    def write(self, raster, filename="raster.tif", driver="GTiff", compress="LZW", tiled=True, lock=None):
        self.fcc.copy(data=raster.astype('float32')).rio.to_raster(self.outpath+filename, driver=driver, compress=compress, tiled=tiled, lock=lock)  
        
    def get_edge_dist(self, targets=[0]):
        if self.crs=='EPSG:4326':
            distance_metric = "GREAT_CIRCLE"
        else:
            distance_metric = "EUCLIDEAN"
        
        dist = xs.proximity(self.fcc, target_values=targets, distance_metric=distance_metric)
        dist = xr.where(dist>0, dist, np.nan)
        return self.chunk(dist).astype("uint32")
    
    def get_darr(self, deforest_targets=[1, 2], non_forest_targets=[0]):
        depix = self.fcc.isin(deforest_targets)
        dist = self.get_edge_dist(targets=non_forest_targets) 
        dedist = dist * depix
        return dedist.data.ravel()
    
    def get_nrt(self, NRT_thresh=99.5, fig_name='distance_stats.png', stats_file='stats.xlsx', show=False):                                          
        deforest_targets = [1, 2]
        non_forest_targets = [0] 

        darr = self.get_darr(deforest_targets=deforest_targets, non_forest_targets=non_forest_targets) 
        max_dist = np.nanmax(darr)         
        bin_edges = np.arange(self.resolution, max_dist, self.resolution)
        hist = histogram1d(darr, bins=bin_edges.shape[0], range=[self.resolution, max_dist])
        npix = np.sum(hist) 

        stats = pd.DataFrame(hist, index=bin_edges)   
        stats['count'] = hist
        stats['perc'] = 100 * hist / npix   
        stats['cum'] = np.cumsum(stats.perc) 
        stats['area'] = self.ar_pix * hist

        inrt = np.argmax(stats.cum >= NRT_thresh)    
        nrt = int((bin_edges[inrt] + bin_edges[inrt + 1]) / 2) 

        summary = {'NRT': nrt,
               'tot_defores_area': npix * self.ar_pix,
               'perc_thres': stats.cum.iloc[inrt]}  
               
        stats.to_excel(self.outpath + stats_file, index_label='distance') 
 
        self.nrt = nrt
        self.summary = summary   
        
        self.plot_distance_stats(stats, summary, show=show, fig_name=fig_name)  
        return summary
    
    def get_vulnerability_map(self, non_forest_targets=[0]):
        llmax = self.resolution
        llmin = self.summary['NRT']
        
        ratio = (llmax / llmin) ** (1 / (self.ncat-1))  
        bins = np.array([llmin * ratio ** i for i in range(self.ncat)])

        dist = self.get_edge_dist(targets=non_forest_targets) 
        excl = ~self.fcc.isin(non_forest_targets)
        deforest_category = self.chunk(xr.apply_ufunc(da.digitize, dist, bins[:-1], dask='allowed') + 1) * excl
        l2_raster = self.chunk(rxr.open_rasterio(self.l2_raster_path, masked=True)).squeeze()
        vulnerability_map = deforest_category * 1000 + l2_raster.data
        return vulnerability_map.astype("uint16")  
    
    def get_unique_values(self, vulnerability_map, columns=['cat', 'ntotal']):
        df = pd.DataFrame(np.asarray((np.unique(vulnerability_map, return_counts=True))).T, columns=columns)
        return df

    def get_relative_frequency(self, vulnerability_map, targets=[1, 2]):
        bins = self.fcc.isin(targets)
        df1 = self.get_unique_values(vulnerability_map,        columns=['cat', 'ntotal'])
        df2 = self.get_unique_values(vulnerability_map * bins, columns=['cat', 'ndefor'])

        new = pd.merge(df1, df2, how='outer', on='cat').fillna(0).astype('int')
        new['nfor'] = new.ntotal - new.ndefor
        new['rate'] = new.ndefor / new.ntotal
        new = new[new.cat!=0]
        return new
    
    def impute_missing_region(self, df, relf):    
        df['risk_class'] = df.cat // 1000

        miss = np.setdiff1d(relf, df.cat)
        miss = pd.DataFrame(miss, columns=['cat'])
        miss['risk_class'] = miss.cat // 1000

        df1 = df[df.risk_class.isin(miss.risk_class)].copy()
        df1['total'] = df1.rate * (self.ar_pix * (df1.nfor + df1.ndefor))

        df2 = df1.groupby('risk_class').sum()
        df2['rate'] = df2.total / (self.ar_pix * (df2.nfor + df2.ndefor))
        df2 = df2.drop(['cat', 'total'], axis=1)

        df3 = pd.merge(miss, df2, on='risk_class', how='outer', suffixes=[None, '1'])
        new = pd.concat([df, df3]).sort_values(by='cat').reset_index(drop=True)[['cat', 'rate']]
        return new
    
    def get_testing_adj_ratio(self, fit_density, fcc, targets=[1, 2]):
        mod_deforest = fit_density.sum()
        exp_deforest = fcc.isin(targets).sum() * self.ar_pix
        ratio = exp_deforest / mod_deforest
        adj_ratio = ratio.values
        return adj_ratio
    
    def get_app_adj_ratio(self, fit_density, activity_data):
        mod_deforest = fit_density.sum()
        exp_deforest = activity_data
        ratio = exp_deforest / mod_deforest
        adj_ratio = ratio.values
        return adj_ratio
    
    def get_adjusted_pred_density_map(self, pred_density, ar):
        adj_dens = pred_density * ar
        adj_dens = xr.where(adj_dens>self.ar_pix, self.ar_pix, adj_dens)
        return adj_dens
    
    def get_density_map(self, df, vm):
        idx_2d = self.chunk(self.fcc.copy(data=np.searchsorted(df['cat'], vm)))      
        dens = self.chunk(self.fcc.copy(data=df['rate'].values[idx_2d] * self.ar_pix))     
        return dens
    
    def fit(self, stage='testing'): 
        non_forest_targets = [0]   

        if stage=='testing':
            deforest_targets = [1]
            
        if stage=='application':
            deforest_targets = [1, 2]
                 
        vm = self.get_vulnerability_map(non_forest_targets=non_forest_targets)
        
        self.rel_freq_table = self.get_relative_frequency(vm, targets=deforest_targets)
        self.rel_freq_table.to_excel(self.outpath + 'rel_freq_%s_fitting.xlsx'%stage, index_label='region')

        fit_density = self.get_density_map(self.rel_freq_table, vm)
        return self.rel_freq_table, fit_density
    
    def predict(self, stage='testing', activity_data=None, max_iter=100, num_years=10):
        if stage=='testing':
            deforest_targets = [2]
            non_forest_targets = [0, 1]
        
        if stage=='application':
            non_forest_targets = [0, 1, 2]

        vm = self.get_vulnerability_map(non_forest_targets=non_forest_targets)    
        vm_classes = np.unique(vm)
        
        pred_rel_freq = self.impute_missing_region(self.rel_freq_table, vm_classes)
        pred_rel_freq.to_excel(self.outpath + 'rel_freq_%s_prediction.xlsx'%stage, index_label='region')

        adj_dens = self.get_density_map(pred_rel_freq, vm)  

        if stage=='testing':  
            ar = self.get_testing_adj_ratio(adj_dens, self.fcc, targets=deforest_targets)    
        else:
            ar = self.get_app_adj_ratio(adj_dens, activity_data)

        for count in np.arange(max_iter):
            if ar>1.00001:
                adj_dens = self.get_adjusted_pred_density_map(adj_dens, ar)
                if stage=='testing':
                    ar = self.get_testing_adj_ratio(adj_dens, self.fcc, targets=deforest_targets)
                else:
                    ar = self.get_app_adj_ratio(adj_dens, activity_data)
        self.adj_ratio = ar
        if stage=='testing':
            return pred_rel_freq, adj_dens
        else:
            return pred_rel_freq, adj_dens / num_years
    
    def plot_distance_stats(self, stats, summary, fig_name='distance_stats.png', dpi=300, show=False):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))

        ax.plot(stats.index, stats.cum, 'b-', lw=2)
        ax.vlines(summary['NRT'], stats.perc.min(), summary['perc_thres'], color='k', ls=':')
        ax.hlines(summary['perc_thres'], stats.index.min(), summary['NRT'], color='k', ls=':')

        ax.set_xlabel("Distance from forest edge (m)")
        ax.set_ylabel("Cumulative Percentage of total deforestation (%)")

        x1_text = summary['NRT'] - 0.01 * np.max(stats.index)
        y2_text = summary['perc_thres'] - 0.01 * (100 - stats.cum.min())

        ax.text(x1_text, stats.cum.min(), 'NRT: %s m'%(summary['NRT']), ha="right", va="bottom")
        ax.text(0, y2_text, '%0.4f '%(summary['perc_thres']) + '%', ha="left", va="top")

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.minorticks_on()
        plt.savefig(self.outpath + fig_name, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot(self, im, outfile, cmap, colorbar=True, linewidth=1, figsize=(5,4), dpi=300):
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        im.plot.imshow(cmap=cmap, ax=ax, add_colorbar=colorbar)
        self.border.plot(color='None', edgecolor='k', linewidth=linewidth, ax=ax, zorder=4)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')

        fig.tight_layout()
        fig.savefig(self.outpath + outfile, dpi="figure", bbox_inches="tight")        
        return fig
    
    def colormap(self, ctype):
        if ctype == 1:
            col = [(255, 165, 0, 255), (227, 26, 28, 255), (34, 139, 34, 255)]
            colors = [(1, 1, 1, 0)]    # transparent white for 0
            cmax   = 255.0             # float for division
            for i in range(3):
                col_class = tuple(np.array(col[i]) / cmax)
                colors.append(col_class)
            color_map = ListedColormap(colors)
            return color_map     
            
        if ctype == 2:
            # Colormap
            colors = []
            cmax = 255.0  # float for division
            vmax = 30.0   # float for division

            # green
            colors.append((0, (34 / cmax, 139 / cmax, 34 / cmax, 1)))
            # orange
            colors.append((1 / vmax, (1, 165 / cmax, 0, 1)))
            # red
            colors.append((15 / vmax, (227 / cmax, 26 / cmax, 28 / cmax, 1)))
            # black
            colors.append((1, (0, 0, 0, 1)))

            color_map = LinearSegmentedColormap.from_list(name="riskmap", colors=colors, N=31, gamma=1.0)
            # Set transparent color for high out-of-range values.
            color_map.set_over(color=(1, 1, 1, 0))
            return color_map     
            
        else:
            raise NotImplementedError('Colormap for this type is not implemented yet.')