# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:07:42 2022

@author: gonz509
"""

import os, glob
import pandas as pd
import numpy as np
import time
# must be in proper directory before importing
import micasense.metadata as metadata
import micasense.imageset as imageset
from mapboxgl.utils import df_to_geojson
import subprocess
from itertools import groupby
import argparse
from sklearn import cluster
#from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import Metashape
import numpy.ma as ma
#from sklearn.linear_model import LinearRegression
from osgeo import gdal

#%% Input parameters


###########################################
###  I N P U T     P A R A M E T E R S  ###
###########################################


parser = argparse.ArgumentParser()
# required args
parser.add_argument('image_directory', help="directory containing tiles")
parser.add_argument('output_directory', help="output directory")

# optional args
parser.add_argument('--bbox', default=0, nargs="+", type=float, help="bounding box with coordinates given as: ULX ULY LRX LRY")
parser.add_argument('--altmin', default=100, type=int, help="Minmal altitude for image to be considered (in metadata units)")
parser.add_argument('--no_lwir', action='store_false', help="do not include long wave IR (thermal) band in output")
parser.add_argument('--no_sbw', action='store_false', help="do not sort bands by wavelength")
parser.add_argument('--no_vc', action='store_false', help="do not perform vignetting correction")
# although the arguments with the 'no' prefix indicate NOT to do something,
# a True value means it WILL do it and a false value means it WILL NOT.
# the default values are True, meaning if you don't add the flag it WILL
# perform those things
parser.set_defaults(no_lwir=True)
parser.set_defaults(no_sbw=True)
parser.set_defaults(no_vc=True)
args = parser.parse_args()

# set variables from parser
image_dir = args.image_directory  # directory with tiles
out_dir = args.output_directory
altmin = args.altmin  # the minimum altitude for images to be included in processing
# a bounding box with the coordinates ordered as: ULX, ULY, LRX, LRY
bbox = args.bbox
lwir = args.no_lwir  # whether or not to include thermal band in processing
sbw = args.no_sbw  # whether or not to include thermal band in processing
vignette_correct = args.no_vc  # whether to apply vignetting correction


#################################################################
#%% Within script defined input parameters
# output folder
# out_dir = r"\\PNL\Projects\UAV_Imagery\Ilan\reflectance\20211115\unstacked\test_set2"
# altmin = 100  # the minimum altitude for images to be included in processing
# # a bounding box with the coordinates ordered as: ULX, ULY, LRX, LRY
# bbox = None
# lwir = True  # whether or not to include thermal band in processing
# vignette_correct = False  # whether to apply vignetting correction
# image_dir = r"\\pnl\Projects\UAV_Imagery\aafcam\20211115\002"


#%% Functions


def create_nested_list_of_images(image_dir, glob_pattern="*.tif"):
    """Create list of lists grouping all bands for a caputure together"""
    
    # get paths of all TIFF files
    glob_pattern = "*.tif"
    glob_path = os.path.join(image_dir, glob_pattern)
    paths = glob.glob(glob_path)
    
    # this creates sublists of all the bands for each capture
    paths.sort()
    im_groups = [list(i) for j, i in groupby(paths, lambda a: a[:-6])]
    
    return im_groups


def reflectance(im_groups, out_dir, bbox=None, altmin=None, lwir=True,
                sort_by_wavelength=True, vignette_correct=True):
    """
    

    Parameters
    ----------
    im_groups : list
        Nested list of full paths of tiles pertaining to each capture.
    out_dir : str
        Path of the output folder.
    bbox : sequence, optional
        Bounding box tuple with the coordinates ordered as: ULX, ULY, LRX, LRY.
        The default is None.
    altmin : int, optional
        Minimum altitude in metadata units for image to be considered.
        The default is None (i.e. no minimum).
    lwir : bool, optional
        Whether to include long wave IR (thermal) in output.
        The default is True.
    sort_by_wavelength : bool, optional
        Whether to sort the bands in order of thier wavelengths.
        The default is True.
    vignette_correct : bool, optional
        Whether to perform the vigentting correction. The default is True.

    Returns
    -------
    Path of the ImageSet CSV containing necessary fields for filtering

    """
    
    
    # check whether images are within bounds, if specified
    if bbox:
        sub_group = []
        for image in im_groups:
      
            # get image metadata
            meta = metadata.Metadata(image[0])#, exiftoolPath=exiftoolPath)
             
            # unpack bounding box coordinates 
            lx = bbox[0]
            uy = bbox[1]
            rx = bbox[2]
            ly = bbox[3]
            
            # extract geographic coordinates and altitude
            lat, lon, alt = meta.position()
            
            
            # conditions check if image IN bounds
            # check lat bounds
            if (uy < lat) or (ly > lat):
                #print("Out of bounds")
                if image == im_groups[-1]:
                    # print("No images in range in this group")
                    break
                else:
                    continue
            # check lon bounds
            if (lx > lon) or (rx < lon):
                #print("Out of bounds")
                if image == im_groups[-1]:
                    # print("No images in range in this group")
                    break
                else:
                    continue
            # check altitude minimum
            if alt < altmin:
                # print("Too low")
                if image == im_groups[-1]:
                    # print("No images in range in this group")
                    break
                else:
                    continue
     
            # print("In range")
            sub_group.append(image)
        
        print(f"{len(sub_group)} images in range identified")
        
        # check whether any matching image exists
        if len(sub_group) == 0:
            raise Exception("Empty list provided")    
        # unnest the list of groups to pass to ImageSet method 
        imset_paths = [item for sublist in sub_group for item in sublist]
        # from_grouplist method was created by ILAN
        imgset = imageset.ImageSet.from_grouplist(imset_paths)
    else:
        # unnest the list of groups to pass to ImageSet method 
        imset_paths = [item for sublist in im_groups for item in sublist]
        imgset = imageset.ImageSet.from_grouplist(imset_paths)

    
    

    # Save out csv and geojson data so we can open the image capture locations in our GIS
    data, columns = imgset.as_nested_lists()
    df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)
    geojson_data = df_to_geojson(df,columns[3:],lat='latitude',lon='longitude')
    group_str = os.path.dirname(im_groups[0][0])[-3:]
    geojson_name = f"{group_str}_imageSet.geojson"
    csv_name = f"{group_str}_imageSet.csv"
    csv_out = os.path.join(out_dir, csv_name)
    df.to_csv(csv_out, index=False)
    with open(os.path.join(out_dir,geojson_name),'w') as f:
        f.write(str(geojson_data))
    
    # use warp matrices for each image in the group, since all images are
    # from same flight
    # TODO potential issue if altitude/temperature changes
    # TODO might have to code in an altitude check to recalculate
    # warp matrices when altitude changes notably within same collect
    tic = time.perf_counter()
    alt_list = []  # list for storing altitude of each capture
    for i, cap in enumerate(imgset.captures):
        
        # create output file name
        im_path = cap.images[0].path  # extract current image path
        # add add prefix and suffix
        prefix = os.path.dirname(im_path)[-3:] + '_'
        savebase = prefix + os.path.basename(im_path)[:-5] + 'refl'
        savename = os.path.join(out_dir, savebase)        
        
        
        # cap.dls_irradiance_raw() gives spectral irradiance
        # only need this if you want the images in memory
        # save method below calculates internallly
        #refl_imgs = cap.undistorted_reflectance(cap.dls_irradiance_raw())
        
        # calculates and directly saves reflectance images for each band
        # and also performs distortion corrections
        # TODO currently coded to use spectral irradiance instead of horizontal
        # TODO need to modifiy source code if want to use spectral
        # TODO adding option would be good idea
        cap.save_bands_as_refl_float(savename, sort_by_wavelength=sort_by_wavelength, vignette_correct=vignette_correct)
        
        
        # copy the metadata of original file to new one
        # get exiftool path
        if os.name == 'nt':
            exiftool_cmd = os.path.normpath(os.environ.get('exiftoolpath'))
        else:
            exiftool_cmd = 'exiftool'
        # sort bands by wavelengths if specified
        if sort_by_wavelength:
            eo_list = list(np.argsort(np.array(cap.center_wavelengths())[cap.eo_indices()]))
        else:
            eo_list = cap.eo_indices()      
        band_list = eo_list + cap.lw_indices()
        for new, idx in enumerate(band_list):
            im = cap.images[idx]  # get image object
            src_path = im.path  # get path to image
            # create new output name
            dst = savebase + f'_{new + 1}.tif'  # new + 1 b/c of 0 indexing
            dst_path = os.path.join(out_dir, dst)
            
            # make command line call
            # copies original metadata to new images
            cmd = f'{exiftool_cmd} -tagsFromFile {src_path} -all:all -xmp {dst_path}'
            # print(cmd)
            if os.name == 'nt':
                subprocess.check_call(cmd)
            else:
                subprocess.check_call(cmd, shell=True)
            
            # remove duplicate orignal exiftool creates
            os.remove(f"{dst_path}_original")
 
    toc = time.perf_counter()
    print(f"Finished making reflectance images. Execution time: {(toc-tic)/60} minutes")
    
    return csv_out


def altitude_filter(imageSet_csv_path):
    """Filter images by altitude using Kmeans clustering and save to CSV"""
    
    
    # file containing altitudes and paths to read in
    csv = imageSet_csv_path
    
    # define some variables for outputting csv at the end
    prefix = os.path.basename(csv).split('.')[-2]
    out_dir = os.path.dirname(csv)
    out_dir_rel = os.path.basename(out_dir)

    # read file
    df = pd.read_csv(csv, converters={'paths': pd.eval})
    
    # subset relevant columns and sort by altitude
    df = df[['altitude', 'paths']]
    df = df.sort_values(by="altitude")
    
    # count the number of 0 altitude tiles
    if 0 in df.altitude.value_counts():
        zero_count = df.altitude.value_counts()[0]
        print(f"There are {zero_count} out of {len(df)} tiles with an altitude of 0.\n\
              {len(df) - zero_count} tiles remain.")
    
    # subset further to exclude tiles with altitude == 0
    sub = df[df.altitude > 0].reset_index()
    
    # the while loop below uses K-Means clustering to group the altitudes into
    # 3 group: low, medium, high
    # however, in some cases there were only two groups and potentially there
    # could be only one. Thus, the code checks the size of the groups and if one
    # or more is too small (defined here as less than 10% of the images) then it
    # is not retained and the clustering is done with only two groups (or left out
    # althogether if there is only one flying altitude)
    # TODO check that you're not running the clustering with 1 cluster, that would be pointless
    ticker = 0
    while ticker >= 0: 
        n_clusters = 3 - ticker
        cluster_data = sub.altitude.values.reshape(-1,1)
        kmeans_cluster = cluster.KMeans(n_clusters=n_clusters)
        kmeans_cluster.fit(cluster_data)
        cluster_labels = kmeans_cluster.predict(cluster_data)
        cluster_centers = kmeans_cluster.cluster_centers_
        #cluster_labels = kmeans_cluster.labels_
        
        # sort the cluster center indexes their corresponding altitude values
        # create dictionary mapping idexes to altitude category
        idxs = cluster_centers.argsort(axis=0).reshape(-1)
        if ticker == 0:
            categories = {idxs[0]: "Low", idxs[1]: "Medium", idxs[2]: "High"}
        elif ticker == 1:
            categories = {idxs[0]: "Medium", idxs[1]: "High"}
        elif ticker == 2:
            categories = {idxs[0]: "Single_Altitude"}
                
        # calculate what percent of images each altitude category constitutes
        # identify if any category is too small and rerun clustering with
        # one less cluster if so
        to_remove = []  # list of 
        for i in range(n_clusters):
            
            cat = categories[i]  # get the category
            count = len(cluster_labels[cluster_labels==i])
            percent = count / len(cluster_labels) * 100
            
            print(f"{cat} altitude group has {count} tiles. \nThis is {percent}% of the tiles.\n")
            
            if percent < 10:
                to_remove.append((cat, count, percent))
        
        if len(to_remove) > 0:
            cat, count, percent = to_remove[0]
            print(f"Removing {cat} group since it constitutes less than 10% of tiles.\n\
                  The images will consequently be split into 2 groups.\n")
            ticker += 1  # forces while loop to continue with one less cluster
        else:
            ticker = -1  # terminates while loop 
    
    # add columns to df that gives cluster label and corresponding altitude category
    data = pd.concat([sub, pd.DataFrame(cluster_labels, columns=["cluster_label"])], axis=1)
    data["alt_class"] = data[["cluster_label"]].applymap(categories.get)
    # add columns for cluster centers and medians, and distance of each image
    # to its respective centrality measure
    data["clust_center"] = cluster_centers[data["cluster_label"]]
    data["dist_to_center"] = abs(data.altitude - data.clust_center)
    for level in data.alt_class.unique():    
        data.loc[data["alt_class"] == level, "clust_median"] = data[data["alt_class"] == level].altitude.median()
    data["dist_to_median"] = abs(data.altitude - data.clust_median)
    data["PlaceHolder"] = "nihil"  # for seaborn plotting
    # remove images that are too far from stable flying altitude (>20m)
    data_clean = data[data.dist_to_median <= 20]
    
    # subset to only columns for writting out
    cols_out = ["altitude", "paths", "alt_class"]
    df_out = data_clean[cols_out]
    df_out = df_out.explode("paths")  # unpack list of lists
    df_out_name = os.path.join(out_dir, f"{prefix}_altitude_classes.csv")
    
    # create and save a plot showing the altitude classes
    fig, ax = plt.subplots(figsize=(8,6), dpi=200)
    sns.set_theme(style="whitegrid")
    sns.stripplot(ax=ax, x="PlaceHolder", y='altitude', hue='alt_class', data=data_clean, jitter=.4)
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Altitude (m)")
    ax.legend(title="Altitude Level")
    ax.legend(bbox_to_anchor=(1.0, 1))#, loc="upper left")
    # save figure
    figname = os.path.join(out_dir, f"{out_dir_rel}_altitude_classes_figure.png")
    plt.savefig(figname, dpi=250, format='png', bbox_inches='tight')
    
    df_out.to_csv(df_out_name, index=False)
    
    
def agisoft_make_ortho(out_dir):

    norm_path_in = os.path.normpath(out_dir)
    norm_path_out = os.path.join(norm_path_in, "orthos")
    dir_names = norm_path_in.split(os.sep)
    date = dir_names[-2]
    chunk_str = dir_names[-1]
    out_base = os.path.join(norm_path_out, f"{date}_{chunk_str}")
    
    # make output folder
    os.mkdir(norm_path_out)
    
    alt_csv = os.path.join(norm_path_in, f"{chunk_str}_imageSet_altitude_classes.csv")
    df = pd.read_csv(alt_csv)
    
    # Make absolute path to images
    make_abs_path = lambda x: os.path.join(norm_path_in, os.path.basename(x))
    df["abs_path"] = df["paths"].apply(make_abs_path)
    
    
    # get altitude classes
    alt_classes = df.alt_class.unique()
    
    # extract list of files for each altitude class into dictionary
    image_dict = {alt: df.abs_path[df.alt_class == alt].tolist() for alt in alt_classes}
    
    
    #project_name = 'project_005_redo.psx'
    # photos = find_files(image_folder, [".tif", ".tiff"])
    
    image_compression = Metashape.ImageCompression()
    image_compression.tiff_big = True
    
    dem_list = []
    ortho_list = []
    for alt, photos in image_dict.items():
        
        out_project = f"{out_base}_{alt}.psx"  # project output path
    
        # create document object
        doc = Metashape.Document()
        doc.save(out_project)
        
        chunk = doc.addChunk()
        
        chunk.addPhotos(photos)
        doc.save()
        
        print(str(len(chunk.cameras)) + " images loaded")
        
        chunk.matchPhotos(keypoint_limit = 40000, tiepoint_limit = 20000, generic_preselection = True, reference_preselection = True)
        #doc.save()
        
        chunk.alignCameras()
        #doc.save()
    
        chunk.buildDepthMaps(downscale = 2, filter_mode = Metashape.AggressiveFiltering)
        #doc.save()
    
        chunk.buildModel(source_data = Metashape.DepthMapsData)
        #doc.save()
    
        chunk.buildUV(page_count = 2, texture_size = 4096)
        #doc.save()
    
        chunk.buildTexture(texture_size = 4096, ghosting_filter = True)
        #doc.save()
        
        has_transform = chunk.transform.scale and chunk.transform.rotation and chunk.transform.translation
    
        if has_transform:
            chunk.buildDenseCloud()
            #doc.save()
    
            chunk.buildDem(source_data=Metashape.DenseCloudData)
            #doc.save()
    
            chunk.buildOrthomosaic(surface_data=Metashape.ElevationData, fill_holes=False)
            doc.save()
        else:
            raise Exception("Transfrom is missing. Try reloading project.")
        
        if chunk.orthomosaic:
            out_ortho = f"{out_base}_{alt}_ortho.tif"
            chunk.exportRaster(out_ortho, source_data = Metashape.OrthomosaicData,
                               save_alpha=False, image_compression=image_compression)
            print(f"{out_ortho} successfully written to disk.")
            ortho_list.append(out_ortho)
        else:
            raise Exception(f"Orthomosaic not present in chunk object for chunk {chunk_str}_{alt}. Try reloading project.")
            
        if chunk.elevation:
            out_DEM = f"{out_base}_{alt}_DEM.tif"
            chunk.exportRaster(out_DEM, source_data = Metashape.ElevationData,
                               save_alpha=False, title="DEM", image_compression=image_compression,
                               nodata_value = 65535)
            print(f"{out_ortho} successfully written to disk.")
            dem_list.append(out_DEM)
        else:
            raise Exception(f"Orthomosaic not present in chunk object for chunk {chunk_str}_{alt}. Try reloading project.")

    
    
    # run this to close project?
    Metashape.Document()
    
    return norm_path_out, ortho_list, dem_list
        
    
def scale_tiff_ortho(tiff_path, out_dir_post_process):
    
    path_out_base = os.path.basename(f"{tiff_path.split('.')[-2]}_scaled.tif")
    path_out = os.path.join(out_dir_post_process, path_out_base)
    ds = gdal.Open(tiff_path)
    
    dtype = gdal.GDT_UInt16
    XSize = ds.GetRasterBand(1).XSize
    YSize = ds.GetRasterBand(1).YSize
    
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(path_out, XSize, YSize, 6, dtype, options=['BIGTIFF=YES'])
    ds_out.SetProjection(ds.GetProjection())
    ds_out.SetGeoTransform(ds.GetGeoTransform())
    
    
    for i in range(1,7):
        inband = ds.GetRasterBand(i)
        outband = ds_out.GetRasterBand(i)
        dta = inband.ReadAsArray()
        dta_ma = ma.masked_values(dta, 1.)
    
        
        if i <= 5:
            # set scale for optical bands to 1000 for compression
            outband.SetScale(.0001)
            outband.SetOffset(0)
            # change values that were 1.0 in original to no data value
            # apply gain and offset and scale
            dta_ma_scaled = dta_ma * 10000
            
    
        elif i == 6:
            # set scale for thermal to 100 for compression
            outband.SetScale(.01)
            outband.SetOffset(0)
            # change values that were 1.0 in original to no data value
            # apply scale (no empirical line fit)
            dta_ma_scaled = dta_ma * 100
            
            
        dta_scaled = dta_ma_scaled.filled(fill_value=65535)
        # write
        outband.WriteArray(dta_scaled)
        # set no data value
        outband.SetNoDataValue(65535)
        outband.FlushCache()
        # compute statistics (Flase means it uses all values)
        ds_out.GetRasterBand(i).ComputeStatistics(False)
        
    
    # build overviews
    ds_out.BuildOverviews('average', [2, 4, 8, 16, 32, 64])
    
    # required to release variable and finish writing
    ds_out = None
    
    # delete uncompressed ortho
    #os.remove(tiff_path)

def scale_tiff_DEM(tiff_path, out_dir_post_process):
    
    path_out_base = os.path.basename(f"{tiff_path.split('.')[-2]}_scaled.tif")
    path_out = os.path.join(out_dir_post_process, path_out_base)
    ds = gdal.Open(tiff_path)
    
    dtype = gdal.GDT_UInt16
    XSize = ds.GetRasterBand(1).XSize
    YSize = ds.GetRasterBand(1).YSize
    
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(path_out, XSize, YSize, 1, dtype)
    ds_out.SetProjection(ds.GetProjection())
    ds_out.SetGeoTransform(ds.GetGeoTransform())
    
    
    inband = ds.GetRasterBand(1)
    outband = ds_out.GetRasterBand(1)
    dta = inband.ReadAsArray()
    dta_ma = ma.masked_values(dta, 65535)
                

    # set scale for DEM to 10 for compression
    outband.SetScale(.1)
    outband.SetOffset(0)
    # specify unit of DEM
    outband.SetUnitType('meter')
    # change values that were 1.0 in original to no data value
    # apply scale (no empirical line fit)
    dta_ma_scaled = dta_ma * 10
        
        
    dta_scaled = dta_ma_scaled.filled(fill_value=65535)
    # write
    outband.WriteArray(dta_scaled)
    # set no data value
    outband.SetNoDataValue(65535)
    outband.FlushCache()
    # compute statistics (Flase means it uses all values)
    ds_out.GetRasterBand(1).ComputeStatistics(False)
        
    
    # build overviews
    ds_out.BuildOverviews('average', [2, 4, 8, 16, 32, 64])
    
    # required to release variable and finish writing
    ds_out = None

    # delete uncompressed DEM
    #os.remove(tiff_path)

#%% Main

def main():
    
    im_groups = create_nested_list_of_images(image_dir)
    
    imageSet_csv_path = reflectance(im_groups,
                                    out_dir=out_dir,
                                    bbox=bbox,
                                    altmin=altmin,
                                    lwir=lwir,
                                    sort_by_wavelength=sbw,
                                    vignette_correct=vignette_correct)
    
    altitude_filter(imageSet_csv_path)
    
    print("Beginning Agisoft orthomosaic production.")
    ortho_dir, ortho_list, dem_list = agisoft_make_ortho(out_dir)
    print("Finished Agisoft orthomosaic production.")

    print("Beginning post processing.")
    out_dir_post_process = os.path.join(ortho_dir, "post_processed")
    os.mkdir(out_dir_post_process)
    
    for ortho in ortho_list:
        scale_tiff_ortho(ortho, out_dir_post_process)
    
    for dem in dem_list:
        scale_tiff_DEM(dem, out_dir_post_process)
    print("Finished post processing. Open images to find reference tarp coordinates\
           and input those into the last part of the workflow for the empirical line fit.")
    

                                    
if __name__ == '__main__':
    main()    