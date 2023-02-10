# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 08:53:10 2022

@author: gonz509
"""

#%% imports

import os
import numpy as np
import numpy.ma as ma
from sklearn.linear_model import LinearRegression
from osgeo import gdal
import matplotlib.pyplot as plt
import argparse

#%% input paramaters


###########################################
###  I N P U T     P A R A M E T E R S  ###
###########################################

# center_wavelengths=[475, 560, 668, 717, 840]
# dark_tarp_vals=[.111329, .108774, .105177, .103725, .101346]
# bright_tarp_vals=[.47787, .47805, .477821, .477038, .476231]

parser = argparse.ArgumentParser()
# required args
parser.add_argument('ortho_path', help="path for orthomosaic")
parser.add_argument('output_directory', help="output directory")
parser.add_argument('--row_col_bright', default=None, nargs=2, type=int, help="row and column (in that order) of the center pixel for bright reference")
parser.add_argument('--row_col_dark', default=None, nargs=2, type=int, help="row and column (in that order) of the center pixel for dark reference")

# optional args
parser.add_argument('--center_wavelengths', default=[475, 560, 668, 717, 840], nargs="+", type=int, help="the center wavelength for each optical band (in the same order as they are in images to be processed)")
parser.add_argument('--bright_tarp_vals', default=[.47787, .47805, .477821, .477038, .476231], nargs="+", type=float, help="reflectance values of the bright reference tarp for each band")
parser.add_argument('--dark_tarp_vals', default=[.111329, .108774, .105177, .103725, .101346], nargs="+", type=float, help="reflectance values of the dark reference tarp for each band")
parser.add_argument('--other_tiles', default=None, nargs="+", type=str, help="relative paths of any other tiles for the same scene as the ortho_path")

args = parser.parse_args()

# set variables from parser
ortho_path = args.ortho_path  # path to ortho being corrected
out_dir = args.output_directory
row_col_bright = args.row_col_bright
row_col_dark = args.row_col_dark
center_wavelengths= args.center_wavelengths
dark_tarp_vals= args.dark_tarp_vals
bright_tarp_vals= args.bright_tarp_vals
tiles = args.other_tiles



#%% fucntions

# assumes bands in image are in order of wavelength
def empirical_line_fit(image_path,
                       out_dir,
                       center_wavelengths,
                       dark_tarp_vals,
                       bright_tarp_vals,
                       row_col_bright=None, row_col_dark=None, tiles=None):
    
    base_path = os.path.dirname(image_path)
    image_basename = [os.path.basename(image_path)]  # make list for loop later
    if tiles:
        tiles = image_basename + tiles
    else:
        tiles = image_basename
    
    # zip tarp values and make into list of arrays (array = band's values)
    ref_vals = zip(bright_tarp_vals, dark_tarp_vals)
    ref_vals = [np.array(list(band)).reshape((-1, 1)) for band in ref_vals]
    
    # open image with GDAL
    ds = gdal.Open(image_path)
    
    # unpack row/col tarp location tuples
    row_bright, col_bright = row_col_bright  
    row_dark, col_dark = row_col_dark
    
    
    obs = []  # list to store mean observed tarp values
    # initiate loop over the 5 optical bands
    for i in range(1, 6):
        # get band and its values
        # divide by 10,000 b/c input TIFFs are already scaled by that factor
        arr = ds.GetRasterBand(i).ReadAsArray() * .0001
        # extract 3x3 array centered on the tarps and take mean pixels
        bright_ref = arr[row_bright-1: row_bright+2, col_bright-1: col_bright+2]
        bright_ref_mean = np.mean(bright_ref)
        dark_ref = arr[row_dark-1: row_dark+2, col_dark-1: col_dark+2]
        dark_ref_mean = np.mean(dark_ref)
        # append values to list as a tuple
        obs.append((bright_ref_mean, dark_ref_mean))
    
    # convert list of tuples to list of arrays and reshape to match ref_vals
    obs_vals = [np.array(list(band)).reshape((-1, 1)) for band in obs]

    
    # extract bright/dark obs into separate arrays for plotting later on
    obs_bright = [entry[0] for entry in obs]
    obs_dark = [entry[1] for entry in obs]
    
    # run linear regression on each band and store coeffs/intercepts
    models = []
    for band in range(len(center_wavelengths)):
        model = LinearRegression().fit(obs_vals[band], ref_vals[band])
        models.append((model.coef_[0][0], model.intercept_[0]))
    
    # apply empirical line fit to observed tarp values
    elf_bright = [obs[1] * models[obs[0]][0] + models[obs[0]][1] for obs in list(enumerate(obs_bright))]
    elf_dark = [obs[1] * models[obs[0]][0] + models[obs[0]][1] for obs in list(enumerate(obs_dark))]
    
    # shorthand vars for plotting
    x = center_wavelengths
    y1 = bright_tarp_vals
    y2 = dark_tarp_vals
   
    # create plot
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8,6), dpi=200)
    ax.scatter(x, y1, color='darkorange')
    ax.scatter(x, y2, color='black')
    ax.plot(x, y1, color='darkorange', label='Bright Ref')
    ax.plot(x, y2, color='black', label='Dark Ref')
    
    ax.scatter(x, elf_bright, color='#988ED5')
    ax.scatter(x, elf_dark, color='#348ABD')
    ax.plot(x, elf_bright, label='Bright EMP', color='#988ED5')
    ax.plot(x, elf_dark, label="Dark EMP", color='#348ABD')
    
    ax.scatter(x, obs_bright, color='#988ED5')
    ax.scatter(x, obs_dark, color='#348ABD')
    ax.plot(x, obs_bright, label="Bright Obs", linestyle='--', color='#988ED5')
    ax.plot(x, obs_dark, label="Dark Obs", linestyle='--', color='#348ABD')
    
    ax.set_ylim(0, .65)
    ax.set_title("Empircal Line Fit Correction")
    ax.set_ylabel("Reflectance")
    ax.set_xlabel("Wavelength (nm)")
    ax.legend()


    plot_name = os.path.join(out_dir, image_basename[0][:-4] + '_EMP_Plot.png')
    # save the plot or comment out if you don't want to save
    plt.savefig(plot_name, format='png')
    
    ##########################################################
    # apply ELF to image and output as new scaled GeoTIFF
    for tile in tiles:
        if tile != tiles[0]:
            image_path_in = os.path.join(base_path, tile)
            ds = gdal.Open(image_path_in)
        image_path_out = os.path.join(out_dir, tile[:-4] + '_EMP.tif')
        dtype = gdal.GDT_UInt16
        XSize = ds.GetRasterBand(1).XSize
        YSize = ds.GetRasterBand(1).YSize
        
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(image_path_out, XSize, YSize, 6, dtype)#, options=['BIGTIFF=YES'])
        ds_out.SetProjection(ds.GetProjection())
        ds_out.SetGeoTransform(ds.GetGeoTransform())
        
        
        for i in range(1,7):
            inband = ds.GetRasterBand(i)
            outband = ds_out.GetRasterBand(i)

            # define gain and offset; i - 1 b/c gdal doesn't use 0 indexing
            if i < 6:  # needed because thermal band (6) is not included 
                gain = models[i-1][0]
                offset = models[i-1][1]
            
                # set scale for optical bands to 1000 for compression
                outband.SetScale(.0001)
                outband.SetOffset(0)
                #  by 10,000 b/c input TIFFs are already scaled by that factor
                dta = inband.ReadAsArray() * .0001
                dta_ma = ma.masked_values(dta, 1.)
                # change values that were 1.0 in original to no data value
                # apply gain and offset and scale
                dta_ma_EMP_scaled = (dta_ma * gain + offset) * 10000
                dta_EMP_scaled = dta_ma_EMP_scaled.filled(fill_value=65535)
                
                # write
                outband.WriteArray(dta_EMP_scaled)
        
            elif i == 6:
                # # set scale for thermal to 100 for compression
                outband.SetScale(.01)
                outband.SetOffset(0)
                dta = inband.ReadAsArray()
                outband.WriteArray(dta)
                
                # this commented out section is only needed if input data is not already scaled
                # dta_ma = ma.masked_values(dta, 1.)
                # # change values that were 1.0 in original to no data value
                # # apply scale (no empirical line fit)
                # dta_ma_scaled = dta_ma * 100
                # dta_scaled = dta_ma_scaled.filled(fill_value=65535)
        
                
                # # write
                # outband.WriteArray(dta_scaled)
            
            
            # set no data value
            outband.SetNoDataValue(65535)
            outband.FlushCache()
            # compute statistics (Flase means it uses all values)
            ds_out.GetRasterBand(i).ComputeStatistics(False)
        
        # build overviews
        ds_out.BuildOverviews('average', [2, 4, 8, 16, 32, 64])
        
        # required to release variable and finish writing
        ds_out = None
    
    
    
    return models
#%% call

def main():
    
    models = empirical_line_fit(ortho_path,
                                out_dir,
                                center_wavelengths=center_wavelengths,
                                dark_tarp_vals=dark_tarp_vals,
                                bright_tarp_vals=bright_tarp_vals,
                                row_col_bright=row_col_bright,
                                row_col_dark=row_col_dark,
                                tiles=tiles)
        

                                    
if __name__ == '__main__':
    main()    



