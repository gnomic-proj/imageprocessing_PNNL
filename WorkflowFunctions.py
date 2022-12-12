# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:07:42 2022

@author: gonz509
"""

import os, glob
import pandas as pd
import numpy as np
import time
import subprocess
from itertools import groupby
import argparse
from sklearn import cluster
from matplotlib import pyplot as plt
import seaborn as sns
import numpy.ma as ma
from osgeo import gdal
from ast import literal_eval

from mapboxgl.utils import df_to_geojson
import Metashape
# must be in proper directory before importing
import micasense.metadata as metadata
import micasense.imageset as imageset



#%% Functions


def split(a, n):
    """Fuction to batch the reflectance calls"""
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def create_nested_list_of_images(image_dir, glob_pattern="*.tif"):
    """Create list of lists grouping all bands for a caputure together"""
    
    # get paths of all TIFF files
    glob_pattern = "*.tif"
    glob_path = os.path.join(image_dir, glob_pattern)
    paths = glob.glob(glob_path)
    
    # this creates sublists of all the bands for each capture
    paths.sort()
    im_groups = [list(i) for j, i in groupby(paths, lambda a: a[:-6])]
    
    print(f"There are {len(im_groups)} image sets")
    
    return im_groups


def filter_images(im_groups, bbox=None, altmin=None):
    """
    Filter images spatially and/or by altitude.\
        
    Parameters
    ----------
    im_groups : list
        Nested list of full paths of tiles pertaining to each capture.
    bbox : sequence, optional
        Bounding box tuple with the coordinates ordered as: ULX, ULY, LRX, LRY.
        The default is None.
    altmin : int, optional
        Minimum altitude in metadata units for image to be considered.
        The default is None (i.e. no minimum).

    Returns
    -------
    ImageSet object
    
    """
    
    sub_group = []
    for image in im_groups:
        
        if altmin or bbox:
            # get image metadata
            meta = metadata.Metadata(image[0])#, exiftoolPath=exiftoolPath)
             
            
            # extract geographic coordinates and altitude
            lat, lon, alt = meta.position()
            if (alt is None) or (lat is None) or (lon is None):
                print(f'One or more of the GPS coordinate values is null (lat={lat}, lon={lon}, alt={alt}). Proceeding to next image.')
                continue
            
            if altmin:
                # check altitude minimum
                if alt < altmin:
                    # print("Too low")
                    if image == im_groups[-1]:
                        # print("No images in range in this group")
                        break
                    else:
                        continue

            if bbox:    
                # unpack bounding box coordinates 
                lx = bbox[0]
                uy = bbox[1]
                rx = bbox[2]
                ly = bbox[3]
                
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
     
            # print("In range")
            sub_group.append(image)
        
        else:
            return
    
    print(f"{len(sub_group)} images in range identified")
    
    # check whether any matching image exists
    if len(sub_group) == 0:
        raise Exception("Empty list provided")    
    # unnest the list of groups to pass to ImageSet method 
    imset_paths = [item for sublist in sub_group for item in sublist]
    # from_grouplist method was created by PNNL
    imgset = imageset.ImageSet.from_grouplist(imset_paths)
    
    return imgset, sub_group


# makes one csv since batch call of original would split into multiple
def make_csv(im_groups, out_dir, bbox=None, altmin=None):
    """
    Create metadata csv for all images being processed.
    
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


    Returns
    -------
    Path of the ImageSet CSV containing necessary fields for filtering

    """
    
    # check whether images are within bounds, if specified
    if bbox or altmin:
       imgset, sub_group = filter_images(im_groups, bbox=bbox, altmin=altmin)
    else:
        # unnest the list of groups to pass to ImageSet method 
        imset_paths = [item for sublist in im_groups for item in sublist]
        imgset = imageset.ImageSet.from_grouplist(imset_paths)
        sub_group = None  # to be able to return when there is a sub_group

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
    
    return csv_out, sub_group


def reflectance(im_groups, out_dir, bbox=None, altmin=None, lwir=True,
                sort_by_wavelength=True, vignette_correct=False, spectral_irr=False):
    """
    Perform corrections on images and convert to reflectance. 
    
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
        Whether to perform the vigentting correction. The default is False.
    spectral_irr : bool, optional
        Whether to use spectral irradiance instead of horizontal. The default is True.

    Returns
    -------
    Path of the ImageSet CSV containing necessary fields for filtering

    """
    
    # check whether images are within bounds, if specified
    if bbox or altmin:
        imgset, _ = filter_images(im_groups, bbox=bbox, altmin=altmin)
    else:
        # unnest the list of groups to pass to ImageSet method 
        imset_paths = [item for sublist in im_groups for item in sublist]
        imgset = imageset.ImageSet.from_grouplist(imset_paths)

    
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
        
        # for troubleshooting
        print(f"Working on: {im_path}")
        
        # add add prefix and suffix
        prefix = os.path.dirname(im_path)[-3:] + '_'
        savebase = prefix + os.path.basename(im_path)[:-5] + 'refl'
        savename = os.path.join(out_dir, savebase)        
        
        
        
        # calculates and directly saves reflectance images for each band
        # and also performs distortion corrections
        # TODO currently coded to use spectral irradiance instead of horizontal
        # TODO need to modifiy source code if want to use horizontal
        # TODO adding option would be good idea
        cap.save_bands_as_refl_float(savename, sort_by_wavelength=sort_by_wavelength, vignette_correct=vignette_correct, spectral_irr=spectral_irr)
        
        
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
            cmd = f'{exiftool_cmd} -tagsFromFile {src_path} -all:all -xmp {dst_path} -q'
            # print(cmd)
            if os.name == 'nt':
                subprocess.check_call(cmd)
            else:
                subprocess.check_call(cmd, shell=True)
            
            # remove duplicate orignal exiftool creates
            os.remove(f"{dst_path}_original")
            
    toc = time.perf_counter()
    print(f"Finished making reflectance images. Execution time: {(toc-tic)/60} minutes")
    
    #return csv_out  # not needed since make_csv function already produces it


def combine_csvs(imageSet_csvs):
    
    # sample csv to pull paths
    csv = imageSet_csvs[0]
    
    # define some variables for outputting csv at the end
    # prefix = os.path.basename(csv).split('.')[-2]
    out_dir = os.path.dirname(csv)
    out_path = os.path.join(out_dir, "combined.csv")
    # read file
    df = pd.concat((pd.read_csv(f, converters={'paths': pd.eval}) for f in imageSet_csvs), ignore_index=True)
    df.to_csv(out_path, index=False)    
    
    return out_path

def altitude_filter(csv_path, altmin=0, n_alt_levels=1):
    """Filter images by altitude using Kmeans clustering and save CSV/plot."""
    
    # read file
    #df = pd.concat((pd.read_csv(f, converters={'paths': pd.eval}) for f in imageSet_csvs), ignore_index=True)
    df = pd.read_csv(csv_path)
    
    # define some variables for outputting csv at the end
    out_dir = os.path.dirname(csv_path)
    
    # subset relevant columns and sort by altitude
    df = df[['altitude', 'paths']]
    df = df.sort_values(by="altitude")
    
    # count the number of 0 altitude tiles
    if 0 in df.altitude.value_counts():
        zero_count = df.altitude.value_counts()[0]
        print(f"There are {zero_count} out of {len(df)} tiles with an altitude of 0.\n\
              {len(df) - zero_count} tiles remain.")
    
    # subset to exclude tiles with altitude == 0
    sub = df[df.altitude > 0].reset_index()

    # subset to exclude tiles with altitude == 0
    if altmin:
        sub = sub[sub.altitude > altmin].reset_index()
    
    
    run = True
    while run:
    
        # use K-Means clustering to group by flying altitude
        n_clusters = n_alt_levels
        cluster_data = sub.altitude.values.reshape(-1,1)
        kmeans_cluster = cluster.KMeans(n_clusters=n_clusters)
        kmeans_cluster.fit(cluster_data)
        cluster_labels = kmeans_cluster.predict(cluster_data)
        cluster_centers = kmeans_cluster.cluster_centers_
        #cluster_labels = kmeans_cluster.labels_
        
        # sort the cluster center indexes their corresponding altitude values
        # create dictionary mapping idexes to altitude category
        idxs = cluster_centers.argsort(axis=0).reshape(-1)
        categories = {idxs[i]: f"level_{i+1}" for i in range(n_clusters)}
    
        # calculate what percent of images each altitude category constitutes
        for i in idxs:
            
            cat = categories[i]  # get the category
            count = len(cluster_labels[cluster_labels==i])
            percent = round(count / len(cluster_labels) * 100, 1)
            
            print(f"{cat} altitude group has {count} tiles. \nThis is {percent}% of the tiles.\n")
            
        
        # add columns to df that give cluster label and corresponding altitude category
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
        #data_clean = data
        data_clean = data[data.dist_to_median <= 20]
        
        # subset to only columns for writting out
        cols_out = ["altitude", "paths", "alt_class"]
        df_out = data_clean[cols_out]
        try:
            df_out["paths"] = df_out["paths"].apply(literal_eval)
        finally:
            df_out = df_out.explode("paths")  # unpack list of lists
        df_out_name = os.path.join(out_dir, "altitude_classes.csv")
        
        # create and save a plot showing the altitude classes
        fig, ax = plt.subplots(figsize=(8,6), dpi=200)
        sns.set_theme(style="whitegrid")
        sns.stripplot(ax=ax, x="PlaceHolder", y='altitude', hue='alt_class', data=data_clean, jitter=.4)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel("Altitude (m)")
        ax.legend(title="Altitude Level")
        ax.legend(bbox_to_anchor=(1.0, 1))#, loc="upper left")
        plt.show()
        
        prompt = input("Are you satisfied with the number of altitude classes? If yes, press ENTER. Otherwise, input the number of classes you'd like to use below:\n")
    
        if prompt == '':
            run = False
        else:
            invalid = True
            while invalid:
                try:
                    n_alt_levels = int(prompt)
                    invalid = False
                except:
                    prompt = input(f"{prompt} is not a number. Input a digit or press ENTER if you are satisfied with the number of altitude classes:\n")
                    n_alt_levels = prompt
                    if prompt == '':
                        invalid = False
                        run = False
                else:
                    print("Redoing clustering with {n_alt_levels} clusters.")
                    
                    
            
    
    # remake figure for saving
    fig, ax = plt.subplots(figsize=(8,6), dpi=200)
    sns.set_theme(style="whitegrid")
    sns.stripplot(ax=ax, x="PlaceHolder", y='altitude', hue='alt_class', data=data_clean, jitter=.4)
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Altitude (m)")
    ax.legend(title="Altitude Level")
    
    # save figure
    figname = os.path.join(out_dir, "altitude_classes_figure.png")
    plt.savefig(figname, dpi=250, format='png', bbox_inches='tight')
    
    df_out.to_csv(df_out_name, index=False)
    
    
def agisoft_make_ortho(out_dir, tiled=True):

    norm_path_in = os.path.normpath(out_dir)
    norm_path_out = os.path.join(norm_path_in, "orthos")
    dir_names = norm_path_in.split(os.sep)
    date = dir_names[-2]
    chunk_str = dir_names[-1]
    out_base = os.path.join(norm_path_out, f"{date}_{chunk_str}")
    
    # make output folder
    try:
        os.mkdir(norm_path_out)
    except:
        print("Cannot create 'orthos' directory, likely because it already exists. Continuing with execution of code.")
            
    alt_csv = os.path.join(norm_path_in, "altitude_classes.csv")
    df = pd.read_csv(alt_csv)
    
    # Make absolute path to images
    make_abs_path = lambda x: os.path.join(norm_path_in, os.path.basename(x))
    df["abs_path"] = df["paths"].apply(make_abs_path)
    
    # get altitude classes
    alt_classes = df.alt_class.unique()
    
    # extract list of files for each altitude class into dictionary
    image_dict = {alt: df.abs_path[df.alt_class == alt].tolist() for alt in alt_classes}
    
    out_project = f"{out_base}.psx"  # project output path
    # create document object
    doc = Metashape.Document()
    doc.save(out_project)
    # create a chunk for each altitude level
    # for i in range(len(image_dict)):
    # this enables big tiff output
    image_compression = Metashape.ImageCompression()
    image_compression.tiff_big = True
    
    # parameters for gradual tiepoint selection
    recunc = 16
    reperr = .55
    #imgcount = 3
    projacc = 16
    
    # initiate lists of dem and ortho paths for use down the pipeline
    dem_list = []
    ortho_list = []
    # idx = 0  # counter to access chunk at each iteration
    for alt, photos in image_dict.items():
        
        chunk = doc.addChunk()
        #chunk = doc.chunks[idx]  # access chunk from doc
        chunk.label = f"Chunk_{alt}"  # given useful label to chunk
        
        chunk.addPhotos(photos)
        doc.save()
        
        print(str(len(chunk.cameras)) + " images loaded")
        
        chunk.matchPhotos(keypoint_limit = 40000, tiepoint_limit = 2000, generic_preselection = True)
        #doc.save()
        
        chunk.alignCameras(adaptive_fitting=True)
        #doc.save()
        
        # gradual tiepoint selection
        # reconstruction uncertainty
        f = Metashape.PointCloud.Filter()
        f.init(chunk, Metashape.PointCloud.Filter.ReconstructionUncertainty)
        f.removePoints(recunc)
        chunk.optimizeCameras(adaptive_fitting=True)
        
        # reprojection error
        f = Metashape.PointCloud.Filter()
        f.init(chunk, Metashape.PointCloud.Filter.ReprojectionError)
        f.removePoints(reperr)
        chunk.optimizeCameras(adaptive_fitting=True)
        
        # not using
        # f = Metashape.PointCloud.Filter()
        # f.init(chunk, Metashape.PointCloud.Filter.ImageCount)
        # f.removePoints(imgcount)
        
        # projection accuracy
        f = Metashape.PointCloud.Filter()
        f.init(chunk, Metashape.PointCloud.Filter.ProjectionAccuracy)
        f.removePoints(projacc)
        chunk.optimizeCameras(adaptive_fitting=True)
        
        # print results of gradual selection
        print("ReprojectionError Level: ")
        print(reperr)
        print("ReconstructionUncertainty Level: ")
        print(recunc)
        # print("ImageCount Level: ")
        # print(imgcount)
        print("ProjectionAccuracy Level: ")
        print(projacc)
        
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
            if tiled:
                chunk.exportRaster(out_ortho, source_data = Metashape.OrthomosaicData,
                                   save_alpha=False, image_compression=image_compression, split_in_blocks=True)
                print(f"{out_ortho} successfully written to disk.")
            else:
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

        chunk.exportReport(f"{out_ortho}_report.pdf")    
        
    # run this to close project?
    Metashape.Document()
    
    if tiled:
        tifs = [t for t in os.listdir(norm_path_out) if t.endswith('.tif')]
        ortho_list = [os.path.join(norm_path_out, i) for i  in tifs  if "ortho" in i]
        
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
        dta = ma.masked_values(dta, 1.)
    
        
        if i <= 5:
            # set scale for optical bands to 1000 for compression
            outband.SetScale(.0001)
            outband.SetOffset(0)
            # change values that were 1.0 in original to no data value
            # apply gain and offset and scale
            dta = dta * 10000
            
    
        elif i == 6:
            # set scale for thermal to 100 for compression
            outband.SetScale(.01)
            outband.SetOffset(0)
            # change values that were 1.0 in original to no data value
            # apply scale (no empirical line fit)
            dta = dta * 100
            
            
        dta = dta.filled(fill_value=65535)
        # write
        outband.WriteArray(dta)
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
    dta = ma.masked_values(dta, 65535)
                

    # set scale for DEM to 10 for compression
    outband.SetScale(.1)
    outband.SetOffset(0)
    # specify unit of DEM
    outband.SetUnitType('meter')
    # change values that were 1.0 in original to no data value
    # apply scale (no empirical line fit)
    dta = dta * 10
        
        
    dta = dta.filled(fill_value=65535)
    # write
    outband.WriteArray(dta)
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
    return path_out


def crop_ortho(ortho_path, out_path, crop_coords):
    
    ortho = gdal.Open(ortho_path, gdal.GA_ReadOnly)
    
    WestBoundCoord = crop_coords[0]
    NorthBoundCoord = crop_coords[1]
    EastBoundCoord = crop_coords[2]
    SouthBoundCoord = crop_coords[3]
    
   
    translateOptionText = f"-projwin {WestBoundCoord} {NorthBoundCoord} {EastBoundCoord} {SouthBoundCoord}"
    translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine(translateOptionText))
    gdal.Translate(out_path, ortho, options=translateoptions)
    
        
def preprocess(parent_dir, out_dir, bbox=None, altmin=None, lwir=True,
                sort_by_wavelength=True, vignette_correct=False, spectral_irr=False):
    
    imageSet_csvs = []
    for image_dir in [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]:
        im_groups = create_nested_list_of_images(image_dir)
        
        # this block splits the processing into chunks
        # circumvents error of undetermined origin when too many images processed at once
        length = len(im_groups)
        calls = int(np.ceil(length / 80))
        print(f"There are {calls} chunks")
        
        # create chunks for processing
        chunks = list(split(im_groups, calls))
        
        # produce metadata CSVs and GeoJSONs
        imageSet_csv_path, sub_group = make_csv(im_groups,
                                        out_dir=out_dir,
                                        bbox=bbox,
                                        altmin=altmin)
        # append csv path to list
        imageSet_csvs.append(imageSet_csv_path)
        
        # process images in chunks
        n_chunk = 1
        for chunk in chunks:
            print(f"Starting on chunk {n_chunk}")
            if any(x in chunk for x in sub_group):
                reflectance(chunk,
                            out_dir=out_dir,
                            bbox=bbox,
                            altmin=altmin,
                            lwir=lwir,
                            sort_by_wavelength=sort_by_wavelength,
                            vignette_correct=vignette_correct,
                            spectral_irr=spectral_irr)
            else:
                 print("No images in range in chunk.")   
            print(f"Finished with chunk {n_chunk}")
            n_chunk += 1
    
    # combine csv for each subfolder into one
    csv_path = combine_csvs(imageSet_csvs)
    
    return csv_path

def agisoft_processing(csv_path, out_dir, altmin=0, n_alt_levels=1, tiled=True):
    
    
    
    print("Plot of altitude classes will be displayed.")
    print("It must be closed for script to continue.")
    # apply altitude filter to create new csv
    altitude_filter(csv_path, altmin=altmin, n_alt_levels=n_alt_levels)
    
    print("Beginning Agisoft orthomosaic production.")
    ortho_dir, ortho_list, dem_list = agisoft_make_ortho(out_dir, tiled=tiled)
    print("Finished Agisoft orthomosaic production.")
    
    df = pd.DataFrame({"ortho_dir": ortho_dir, "orthos": ortho_list, "dems": dem_list})
    csv_out_path = os.path.join(out_dir, "orthos.csv")
    df.to_csv(csv_out_path, index=False)
    
    return csv_out_path
    
    
def post_processing(csv_out_path, crop_coords=None):
    
    print("Beginning post processing.")
    
    df = pd.read_csv(csv_out_path)
    
    ortho_dir = df.loc[0, "ortho_dir"]
    ortho_list = list(df.orthos)
    dem_list = list(df.dems[df.dems.notna()])
    
    out_dir_post_process = os.path.join(ortho_dir, "post_processed")
    try:
        os.mkdir(out_dir_post_process)
    except:
        print("Cannot create 'post_processed' directory, likely because it already exists. Continuing with execution of code.")
    
    for ortho in ortho_list:
        if crop_coords:
            path_out_base = f"{ortho.split('.')[-2]}_crop.tif"
            out_path = os.path.join(out_dir_post_process, path_out_base)
            crop_ortho(ortho, out_path, crop_coords)
            ortho_path = scale_tiff_ortho(out_path, out_dir_post_process)
        else:
            ortho_path = scale_tiff_ortho(ortho, out_dir_post_process)
    
    for dem in dem_list:
        if crop_coords:
            path_out_base = f"{dem.split('.')[-2]}_crop.tif"
            out_path = os.path.join(out_dir_post_process, path_out_base)
            crop_ortho(dem, out_path, crop_coords)
        dem_path = scale_tiff_DEM(dem, out_dir_post_process)
    print("Finished post processing. Open images to find reference tarp coordinates\
           and input those into the last part of the workflow for the empirical line fit.")
           
           