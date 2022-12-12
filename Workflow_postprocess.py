# -*- coding: utf-8 -*-

from WorkflowFunctions import *

#%% Input parameters


###########################################
###  I N P U T     P A R A M E T E R S  ###
###########################################

# NOTE: the file structure must be as follows:
# the parent_directory should contain all the individual chunk folders and no others

parser = argparse.ArgumentParser()
# required args
parser.add_argument('csv_path_ortho', help="path of ortho csv from Agisoft step")

# optional args
parser.add_argument('--bbox', default=None, nargs="+", type=float, help="bounding box used in preprocessing with coordinates given as: ULX ULY LRX LRY")
parser.add_argument('--crop_coords', default=None, nargs="+", type=float, help="cropping bounding box used after orthomosaic production with coordinates given as: ULX ULY LRX LRY")
parser.add_argument('--altmin', default=100, type=int, help="Minmal altitude for image to be considered (in metadata units)")
parser.add_argument('--n_alt_levels', default=1, type=int, help="The number of different flying altitude levels for which to create separate orthomosaics")
parser.add_argument('--no_lwir', action='store_false', help="do not include long wave IR (thermal) band in output")
parser.add_argument('--no_sbw', action='store_false', help="do not sort bands by wavelength")
parser.add_argument('--vc', action='store_true', help="do not perform vignetting correction in preprocessing")
# although the arguments with the 'no' prefix indicate NOT to do something,
# a True value means it WILL do it and a false value means it WILL NOT.
# the default values are True, meaning if you don't add the flag it WILL
# perform those things
# the vc argument is the converse (i.e. using the flag DOES perform it)
parser.set_defaults(no_lwir=True)
parser.set_defaults(no_sbw=True)
parser.set_defaults(vc=False)
args = parser.parse_args()

# set variables from parser
csv_path_ortho = args.csv_path_ortho
altmin = args.altmin  # the minimum altitude for images to be included in processing
n_alt_levels = args.n_alt_levels
# a bounding box with the coordinates ordered as: ULX, ULY, LRX, LRY
bbox = args.bbox
crop_coords = args.crop_coords
lwir = args.no_lwir  # whether or not to include thermal band in processing
sort_by_wavelength = args.no_sbw  # whether or not to include thermal band in processing
vignette_correct = args.vc  # whether to apply vignetting correction


#################################################################
#################################################################


    
#%% Main

def main():
    
    post_processing(csv_path_ortho, crop_coords=crop_coords)

                                    
if __name__ == '__main__':
    main()    