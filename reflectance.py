# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:09:24 2022

@author: gonz509
"""

#import matplotlib.pyplot as plt
#import numpy as np
#import math
#from functools import reduce
#from itertools import groupby
#from osgeo import gdal
import os, glob
import pandas as pd
import numpy as np
import time
import cv2
os.chdir(r"C:\Users\gonz509\GitRepos\imageprocessing")
# must be in proper directory before importing
import micasense.metadata as metadata
import micasense.capture as capture
import micasense.imageutils as imageutils
import micasense.imageset as imageset
from ipywidgets import FloatProgress, Layout
from IPython.display import display
import exiftool
from mapboxgl.utils import df_to_geojson
import subprocess
from itertools import groupby


def decdeg2dms(dd):
   is_positive = dd >= 0
   dd = abs(dd)
   minutes,seconds = divmod(dd*3600,60)
   degrees,minutes = divmod(minutes,60)
   degrees = degrees if is_positive else -degrees
   return (degrees,minutes,seconds)


def reflectance(im_groups, savepath=None, bbox=None, lwir=False, sort_by_wavelength=True):
    
    #TODO include altitude grouping here, need to think through how to add
    # TODO also need to make this work for when there is no bbox, doesn't seem like it does currently?
    
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
            altmin = bbox[4]
            
            # extract geographic coordinates and altitude
            lat, lon, alt = meta.position()
            
            
            # conditions check if image IN bounds
            # check lat bounds
            if (uy < lat) or (ly > lat):
                #print("Out of bounds")
                if image == im_groups[-1]:
                    print("No images in range in this group")
                    break
                else:
                    continue
            # check lon bounds
            if (lx > lon) or (rx < lon):
                #print("Out of bounds")
                if image == im_groups[-1]:
                    print("No images in range in this group")
                    break
                else:
                    continue
            # check altitude minimum
            if alt < altmin:
                print("Too low")
                if image == im_groups[-1]:
                    print("No images in range in this group")
                    break
                else:
                    continue
     
            print("In range")
            sub_group.append(image)
        
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
        #capt = capture.Capture.from_filelist(image)

    
    

    # Save out geojson data so we can open the image capture locations in our GIS
    data, columns = imgset.as_nested_lists()
    df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)
    geojson_data = df_to_geojson(df,columns[3:],lat='latitude',lon='longitude')
    group_str = os.path.dirname(im_groups[0][0])[-3:]
    geojson_name = group_str + "_imageSet.geojson"
    
    with open(os.path.join(savepath,geojson_name),'w') as f:
        f.write(str(geojson_data))
    
    # use warp matrices for each image in the group, since all images are
    # from same flight
    # TODO potential issue if altitude/temperature changes
    # TODO might have to code in an altitude check to recalculate
    # warp matrices when altitude changes notably within same collect
    # TODO 6/21 NEED TO FIX FILE NAMING
    # AND ALSO EXTRACT DESTINATION FILENAME FOR EXIF WRITING BELOW
    tic = time.perf_counter()
    for i, cap in enumerate(imgset.captures):
        
        # create output file name
        im_path = cap.images[0].path  # extract current image path
        # add add prefix and suffix
        prefix = os.path.dirname(im_path)[-3:] + '_'
        savebase = prefix + os.path.basename(im_path)[:-5] + 'refl'
        savename = os.path.join(savepath, savebase)        
        
        
        # cap.dls_irradiance_raw() gives spectral irradiance
        # only need this if you want the images in memory
        # save method below calculates internallly
        #refl_imgs = cap.undistorted_reflectance(cap.dls_irradiance_raw())
        
        # method added by Ilan
        # calculates and directly saves reflectance images for each band
        # TODO also performs distortion corrections etc. as contained in ?????
        cap.save_bands_as_refl_float(savename, sort_by_wavelength=sort_by_wavelength)
        
        
        # copy the metadata of original file to new one
        # get exiftool path
        exiftool_cmd = os.path.normpath(os.environ.get('exiftoolpath'))
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
            dst_path = os.path.join(savepath, dst)
            
            # make command line call
            # copies original metadata to new images
            cmd = f'{exiftool_cmd} -tagsFromFile {src_path} -all:all -xmp {dst_path}'
            print(cmd)
            subprocess.check_call(cmd)
        
    toc = time.perf_counter()
    print(f"Saving time: {(toc-tic)/60} minutes")
    
#%% User set values

# wd
image_dir = r"\\pnl\Projects\UAV_Imagery\aafcam\20211115\002"
#image_path = os.path.join(image_dir,'IMG_1000_1.tif')  # single image


glob_pattern = "*.tif"
glob_path = os.path.join(image_dir, glob_pattern)
paths = glob.glob(glob_path)
#paths = [path for path in paths if not path.endswith("6.tif")]

paths.sort()
im_groups = [list(i) for j, i in groupby(paths, lambda a: a[:-6])]

savepath = r"\\PNL\Projects\UAV_Imagery\Ilan\reflectance\20211115\unstacked\test_set2"
# ULX, ULY, LRX, LRY, altitude minimum
bbox = -97.497701, 36.637810, -97.477929, 36.585851, 500
#bbox = None
lwir = True  # set to true to include in output stack

#%% Call

reflectance(im_groups, savepath=savepath, bbox=bbox, lwir=lwir)
