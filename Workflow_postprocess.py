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
parser.add_argument('--crop_coords', default=None, nargs="+", type=float, help="cropping bounding box used after orthomosaic production with coordinates given as: ULX ULY LRX LRY")

args = parser.parse_args()

# set variables from parser
csv_path_ortho = args.csv_path_ortho
# a cropping box with the coordinates ordered as: ULX, ULY, LRX, LRY
crop_coords = args.crop_coords


#################################################################
#################################################################


    
#%% Main

def main():
    
    post_processing(csv_path_ortho, crop_coords=crop_coords)

                                    
if __name__ == '__main__':
    main()    