# UAS Image Processing with Micasense and Agisoft
Ilan Gonzalez-Hirshfeld, Kristian Nelson, Lexie Goldberger, Jerry Tagestad

## Description

This codebase was developed to provide semi-automatic Python processing of Micasense Altum imagery to produce othomosaics and DEMs. It is designed for execution from a command line interface. It relies on and is adaptated from the open source Micasense imageprocessing library as well as the Agisoft Metashape Python API. It consists of a workflow implemented through either four Python scripts or a more integrated two script workflow. The four steps are: 1) Preprocess the uncorrected Altum imagery to produce reflectance images using code adapted from the Micasense library; 2) Interactively filter and subdivide images into altitude classes based on user input and supervision together with k-means clustering; 3) Process the imagery from each altitude class into orthomosaics and DEMs; 4) Inspect imagery and apply an empirical line fit correction to the orthomosaics.

***NOTE: A valid Metashape license is required for step 3.***

## Setup

This repo leverages [Micasense imageprocessing](https://github.com/micasense/imageprocessing) for preprocessing and the [Metashape Python module](https://agisoft.freshdesk.com/support/solutions/articles/31000148930-how-to-install-metashape-stand-alone-python-module) for orthomosaic/DEM production. Ad hoc modifications of the Micasense library were developed for the particular processing needs of this workflow. Thus, the PNNL fork of that library is used in this implementation: [imageprocessing_PNNL](https://github.com/gnomic-proj/imageprocessing_PNNL). 

***NOTE: Make sure you are using the imageprocessing_PNNL fork listed above and not the original Micasense imageprocessing library.***

### 1. Install micasense module from imageprocessing_PNNL fork

Follow the Micasense installation tutorial found [here](https://github.com/gnomic-proj/imageprocessing_PNNL/blob/master/MicaSense%20Image%20Processing%20Setup.ipynb).

Once again, make sure you are on the imageprocessing_PNNL fork and looking at the installation tutorial in that repository.

The tutorial takes you through the proper installation steps based on OS as well as verifications that the installation has been successful.


### 2. Install Metashape stand-alone Python module

***NOTE: A valid Metashape license is required for this to be successful.***

1. If the micasense conda environment is not already activated, activate it by running the following command in the appropriate conda terminal:

`conda activate micasense`

2. Follow the Agisoft API installation instructions found [here](https://agisoft.freshdesk.com/support/solutions/articles/31000148930-how-to-install-metashape-stand-alone-python-module).

Make sure you are executing the pip install from within the micasense conda environment that was previously set up.


## Workflow

There are two options: the two-script implementation or the four-script implementation. The two-script implementation simply combines the content of the first three scripts of the four-script implementation into one. It is recommended to start by using the two-script implementation. If issues are encountered, the user may wish to rerun certain steps using some or all of the scripts in the four-script implementation. 

This documentation first details the two-script approach. If you know you are going to use the four-script implementation, you can skip ahead to the corresponding section below.

### Two-script Implementation

The workflow detailed here involves two scripts: one to perform all the processing excluding the empirical line fit correction and another to perform the empirical line fit correction. They must be run in that order.

1. Launch the appropriate CLI and cd into the imageprocessing_PNNL directory.
Example:
`cd C:\path\to\repos\imageprocessing_PNNL`

The details of this command will vary based on OS and install location.

2. If the micasense conda environment is not already activated, activate it by running the following command:

`conda activate micasense`

3. Run the `WorkflowPt1Complete.py` script in the CLI using the arguments detailed below.

This script will perform all processing steps prior to the empirical line fit correction. 
***NOTE: The altitude filtering portion of the workflow requires user input to proceed.***
Depending on imagery quantity and processing power, this may occur minutes to several hours after execution initiation. Once the filtering has started, text will be printed to the CLI alerting you. Grouping will be performed based on the user input values at script execution, or else with default vaules. A plot will be displayed showing the distribution of images and their classification into separate groups. Once examined, you MUST close the plot for the code execution to proceed. You will then be prompted whether or not to change the number of groups by entering the new number of groups followed by ENTER, or to accept the current number of groups used by pressing ENTER with no input to proceed to the next step. The goal is to identify the proper number of groups to ensure that images acquired at roughly the same altitude all belong to a single group (and thus orthomosaic). Otherwise, if images from substatially varied acquisition altitudes contribute to a single orthomosaic, issues may be encountered in the orthomosaic production step.

**Required Inputs**:
- Full path to the parent directory containg the Micasense Altum imagery subdirectories to be processed. Usually this is a folder named ####SET, for example 0000SET.
- Full path to the output directory.

**Outputs**

- Corrected reflectance tiles (TIFFs)
- CSVs and GeoJSONs for each imageSet in the input directory, named like {imageSet number}_imageSet.csv/geojson
- `combined.csv` which contains the combined information of the individual imageSet CSVs
- `altitude_classes.csv` containing information about which images belong to which altitude classes
- `altitude_classes_figure.png` showing the distribution of images by altitude
- An Agisoft .psx project file and associated folder ending in .files
- `orthos.csv` containing information about the orthomosaics
- An `orthos` directory containing the initial orthomosaics and DEMs as well as PDF reports of the processing
- A `post_processed` directory within the `orthos` directory that contains scaled (compressed) versions of the orthomosaics and DEMs.

```
usage: WorkflowPt1Complete.py [--bbox] [--crop_coords] [--altmin] [--n_alt_levels] [--no_lwir]
[--no_sbw] [--no_tiled] [--vc] [--spec_irr] parent_dir out_dir

positional arguments:
  parent_dir     Full path to the parent directory containing the Micasense Altum imagery subdirectories
  out_dir        output directory

optional arguments:
 --bbox <ULX> <ULY> <LRX> <LRY>. Bounding box used in preprocessing with coordinates given as: upper-left-X upper-left-Y lower-right-X lower-right-Y
 --crop_coords <ULX> <ULY> <LRX> <LRY>. Cropping box used after orthomosaic production with coordinates given as: upper-left-X upper-left-Y lower-right-X lower-right-Y
 --altmin <altmin>, default=100. Minimum altitude for image to be considered (in metadata units)
 --n_alt_levels <n_alt_levels>, default=1. The number of different flying altitude levels for which to create separate orthomosaics.
 --no_lwir, do not include long wave IR (thermal) band in output
 --no_sbw, do not sort bands by wavelength
 --no_tiled, do not tile the orthomosaics
 --vc, perform vignetting correction in preprocessing (included in Agisoft processing)
 --spec_irr, use spectral irradiance instead of horizontal irradiance

```

Example usage:

`>> python WorkflowPt1Complete.py C:\path\to\images\0000SET C:\path\to\output\directory  --altmin 800`

This will create scaled (i.e. compressed) orthomosaics, excluding all imagery at an altitude below 800. Choosing an appropriate minimum altitude can be important to exclude images that were not intended to contribute to final orthomosaic productions (e.g. calibration imagery, imagery acquired during ascent to lowest stable altitude level, etc.). Remember that user input must always be given during the altitude filtering step, even if "n_alt_levels" is specified.


***NOTE: Steps from this point forward are only necessary if you are performing an empirical line fit correction using a set of light and dark tarps present in the imagery.***

4. If performing empirical line fit correction, open the orthomosaics in "post_processed" directory and identify the pixel row and column value of tarp locations. If your outputs are tiled orthomosaics (the default), you will have to identify which tile for each orthomosaic contains the tarps.

5. Perform empirical line fit correction to produce final product by running `Workflow_EmpiricalLineFit.py`.

**Required Inputs**:
- Full path to orthomosaic containing tarps for applying correction. 
- Full path to the output directory.
- The row and column value of the pixel most closely centered on the bright tarp
- The row and column value of the pixel most closely centered on the dark tarp

**Outputs**

- Empirical line fit corrected orthomosaics/tiles (TIFFs)
- A PNG plot showing the correction values

```
usage: Workflow_EmpiricalLineFit.py [--center_wavelengths] [--bright_tarp_vals] [--dark_tarp_vals] [--other_tiles] ortho_path out_dir --row_col_bright --row_col_dark

positional arguments:
  ortho_path     Full path to the parent directory containing the Micasense Altum imagery subdirectories
  out_dir        output directory

required tick arguments:
--row_col_bright <ROW> <COLUMN>. The row and column value of the pixel most closely centered on the bright tarp.
--row_col_dark <ROW> <COLUMN>. The row and column value of the pixel most closely centered on the dark tarp.

optional arguments:
 --center_wavelengths <BAND_1> <BAND_2> <BAND_3> <BAND_4> <BAND_5>, default=0.47787 0.47805 0.477821 0.477038 0.476231. Τhe center wavelength (nm) as integers for each optical band (in the same order as they are in images to be processed). Default values correspond to the Micasense Altum imager. If needed, more bands can be specified by adding additional values separated by spaces.
 --bright_tarp_vals <BAND_1> <BAND_2> <BAND_3> <BAND_4> <BAND_5>, default=0.47787 0.47805 0.477821 0.477038 0.476231. Reflectance values of the bright reference tarp for each band. Default values correspond to the Micasense Altum imager. If needed, more bands can be specified by adding additional values separated by spaces.
 --dark_tarp_vals <BAND_1> <BAND_2> <BAND_3> <BAND_4> <BAND_5>, default=0.111329 0.108774 0.105177 0.103725 0.101346. Reflectance values of the dark reference tarp for each band. Default values correspond to the Micasense Altum imager. If needed, more bands can be specified by adding additional values separated by spaces.
 --other_tiles <FILENAME>* Filenames (not full paths!) of any other tiles for the same scene as ortho_path. Any number of filenames can be specified separated by a space.

```

Example usage:

`>> python Workflow_EmpiricalLineFit.py C:\path\to\post_processed\ortho_path_with_tarps C:\path\to\output\directory  --row_col_bright 961 1002 --row_col_dark 945 995 --other_tiles other_tile_1.tif other_tile_2.tif other_tile 3.tif`

This will calculate the empirical line fit correction using the orthomosaic tile specifed in the first positional argument and apply it to that image as well as to the orthomosaic tiles indicated by the "other_tiles" argument. The script will use row/column values of 961/1002 and 945/995 to identify bright and dark tarp locations, respectively.


### Four-script Implementation

***NOTE: This is an alternative and equivalent workflow to the one detailed above. It should be used in its entirety in place of the above workflow (Two-script Implementaion) or else select portions may be used for ad hoc workflow tailoring/troubleshooting.***

The workflow detailed here involves four scripts corresponding to the four steps given in the description portion of this document. Running any individual script presupposes and requires the prior execution of any preceding scripts in the workflow.

1. Launch the appropriate command line interface and cd into the imageprocessing_PNNL directory.
Example:
`cd C:\path\to\repos\imageprocessing_PNNL`

The details of this command will vary based on OS and install location.

2. If the micasense conda environment is not already activated, activate it by running the following command:

`conda activate micasense`

3. Run the Workflow_preprocess.py script in the CLI using the arguments detailed below. This script will apply basic corrections to the Altum imagery and convert them to reflectance TIFFs.



**Required Inputs**:
- Full path to the parent directory containg the Micasense Altum imagery subdirectories to be processed. Usually this is a folder named ####SET, for example 0000SET.
- Full path to the output directory.

**Outputs**

- Corrected reflectance tiles (TIFFs)
- CSVs and GeoJSONs for each imageSet in the input directory, named like {imageSet number}_imageSet.csv/geojson
- combined.csv which contains the combined information of the individual imageSet CSVs. The CSV path will be printed to CLI, to be used as input for the `Workflow_agisoft` script.

```
usage: Workflow_preprocess.py [--bbox] [--altmin] [--no_lwir]
[--no_sbw] [--vc] [--spec_irr] parent_dir out_dir

positional arguments:
  parent_dir     Full path to the parent directory containing the Micasense Altum imagery subdirectories
  out_dir        output directory

optional arguments:
 --bbox <ULX> <ULY> <LRX> <LRY>. Bounding box used in preprocessing with coordinates given as: upper-left-X upper-left-Y lower-right-X lower-right-Y
 --altmin <altmin>, default=100. Minimum altitude for image to be considered (in metadata units)
 --no_lwir, do not include long wave IR (thermal) band in output
 --no_sbw, do not sort bands by wavelength
 --vc, perform vignetting correction in preprocessing (included in Agisoft processing)
 --spec_irr, use spectral irradiance instead of horizontal irradiance

```

Example usage:

`>> python Workflow_preprocess.py C:\path\to\images\0000SET C:\path\to\output\directory  --altmin 800`

This will create reflectance images, excluding all imagery at an altitude below 800.

4. Run the Workflow_agisoft.py script in the CLI using the arguments detailed below. This script will perform altitude grouping/filtering and produce orthomosaics for each altitude level.

***NOTE: A valid Metashape license is required for this to be successful.***

***NOTE: The altitude filtering portion of the workflow requires user input to proceed.***
Depending on imagery quantity and processing power, this may occur minutes to several hours after execution start. Once the filtering has started, text will be printed to the CLI alerting you. Grouping will be performed based on the user input values at script execution, or else with default vaules. A plot will be displayed showing the distribution of images and their classification into separate groups. Once examined, you MUST close the plot for the code execution to proceed. You will then be prompted whether or not to change the number of groups by entering the new number of groups followed by ENTER, or to accept the current number of groups used by pressing ENTER with no input to proceed to the next step. The goal is to identify the proper number of groups to ensure that images acquired at roughly the same altitude all belong to a single group (and thus orthomosaic). Otherwise, issues may be encountered in the orthomosaic production step if images from substatially varied acquisition altitudes are used to produce a single orthomosaic.

**Required Inputs**:
- Full path to the `combined.csv` produced in the last step and printed to the CLI.
- Full path to the output directory.

**Outputs**
- `altitude_classes.csv` containing information about which images belong to which altitude classes
- `altitude_classes_figure.png` showing the distribution of images by altitude
- an Agisoft .psx project file and associated folder ending in .files
- `orthos.csv` containing information about the orthomosaics. The CSV path will be printed to CLI, to be used as input for the `Workflow_postprocess` script.
- an "orthos" directory containing the initial orthomosaics and DEMs as well as PDF reports of the processing

```
usage: Workflow_agisoft.py [--altmin] [--n_alt_levels] [--no_tiled] csv_path out_dir

positional arguments:
  csv_path     Full path to the CSV containing the combined information about the preprocessed images (produced in by previous script)
  out_dir      output directory

optional arguments:
 --altmin <altmin>, default=100. Minimum altitude for image to be considered (in metadata units)
 --n_alt_levels <n_alt_levels>, default=1. The number of different flying altitude levels for which to create separate orthomosaics.
 --no_tiled, do not tile the orthomosaics

```

Example usage:

`>> python Workflow_agisoft.py C:\path\to\CSV\combined.csv C:\path\to\output\directory  --altmin 800 --n_alt_levels 7`

This will create unscaled orthomosaics, excluding all imagery at an altitude below 800, and sort them into seven altitude groups. Choosing an appropriate minimum altitude can be important to exclude images that were not intended to contribute to final orthomosaic productions (e.g. calibration imagery, imagery acquired during ascent to lowest stable altitude level, etc.). Remember that user input must always be given during the altitude filtering step, even if `n_alt_levels` is specified.

5. Run the Workflow_postprocess.py script in the CLI using the arguments detailed below. This script will perform altitude grouping/filtering and produce orthomosaics for each altitude level.

**Required Inputs**:
- Full path to the orthos.csv produced in the last step and printed to the CLI.
- Full path to the output directory.

**Outputs**
- a `post_processed` directory within the `orthos` directory that contains scaled (compressed) versions of the orthomosaics and DEMs.

#TODO 23-1-18 NEED TO CHANGE THIS TO NOT HAVE out_dir (to reflect code) and rewrite docs to match
usage: Workflow_postprocess.py [--crop_coords] csv_path out_dir

positional arguments:
  csv_path     Full path to the CSV containing the combined information about the orthomosaics (produced in by previous script)
  out_dir      output directory

optional arguments:
 --crop_coords <ULX> <ULY> <LRX> <LRY>. Cropping box used with coordinates given as: upper-left-X upper-left-Y lower-right-X lower-right-Y

```

Example usage:

`>> python Workflow_postprocess.py C:\path\to\CSV\orthos\ortho.csv C:\path\to\output\directory`

This will create scaled orthomosaics and DEMs. If the `crop_coords` were specified, it would also crop the orthomosaics to those coordintes.

6. If performing empirical line fit correction, open the orthomosaics in "post_processed" directory and identify the pixel row and column value of tarp locations. If your outputs are tiled orthomosaics (the default), you will have to identify which tile for each orthomosaic contains the tarps.

7. Perform empirical line fit correction to produce final product by running `Workflow_EmpiricalLineFit.py`.

**Required Inputs**:
- Full path to orthomosaic containing tarps for applying correction. 
- Full path to the output directory.
- The row and column value of the pixel most closely centered on the bright tarp
- The row and column value of the pixel most closely centered on the dark tarp

**Outputs**

- Empirical line fit corrected orthomosaics/tiles (TIFFs)
- A PNG plot showing the correction values

```
usage: Workflow_EmpiricalLineFit.py [--center_wavelengths] [--bright_tarp_vals] [--dark_tarp_vals] [--other_tiles] ortho_path out_dir --row_col_bright --row_col_dark

positional arguments:
  ortho_path     Full path to the parent directory containing the Micasense Altum imagery subdirectories
  out_dir        output directory

required tick arguments:
--row_col_bright <ROW> <COLUMN>. The row and column value of the pixel most closely centered on the bright tarp.
--row_col_dark <ROW> <COLUMN>. The row and column value of the pixel most closely centered on the dark tarp.

optional arguments:
 --center_wavelengths <BAND_1> <BAND_2> <BAND_3> <BAND_4> <BAND_5>, default=0.47787 0.47805 0.477821 0.477038 0.476231. Τhe center wavelength (nm) as integers for each optical band (in the same order as they are in images to be processed). Default values correspond to the Micasense Altum imager. If needed, more bands can be specified by adding additional values separated by spaces.
 --bright_tarp_vals <BAND_1> <BAND_2> <BAND_3> <BAND_4> <BAND_5>, default=0.47787 0.47805 0.477821 0.477038 0.476231. Reflectance values of the bright reference tarp for each band. Default values correspond to the Micasense Altum imager. If needed, more bands can be specified by adding additional values separated by spaces.
 --dark_tarp_vals <BAND_1> <BAND_2> <BAND_3> <BAND_4> <BAND_5>, default=0.111329 0.108774 0.105177 0.103725 0.101346. Reflectance values of the dark reference tarp for each band. Default values correspond to the Micasense Altum imager. If needed, more bands can be specified by adding additional values separated by spaces.
 --other_tiles <FILENAME>* Filenames (not full paths!) of any other tiles for the same scene as ortho_path. Any number of filenames can be specified separated by a space.

```

Example usage:

`>> python Workflow_EmpiricalLineFit.py C:\path\to\post_processed\ortho_path_with_tarps C:\path\to\output\directory  --row_col_bright 961 1002 --row_col_dark 945 995 --other_tiles other_tile_1.tif other_tile_2.tif other_tile 3.tif`

This will calculate the empirical line fit correction using the orthomosaic tile specifed in the first positional argument and apply it to that image as well as to the orthomosaic tiles indicated by the "other_tiles" argument. The script will use row/column values of 961/1002 and 945/995 to identify bright and dark tarp locations, respectively.