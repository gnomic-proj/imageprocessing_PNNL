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
parser.add_argument('csv_path', help="path of combined csv from preprocess step")
parser.add_argument('output_directory', help="output directory")

# optional args
parser.add_argument('--altmin', default=100, type=int, help="Minmal altitude for image to be considered (in metadata units)")
parser.add_argument('--n_alt_levels', default=1, type=int, help="The number of different flying altitude levels for which to create separate orthomosaics")
parser.add_argument('--no_tiled', action='store_false', help="Do not tile the orthomosaics.")
# although the arguments with the 'no' prefix indicate NOT to do something,
# a True value means it WILL do it and a false value means it WILL NOT.
# the default values are True, meaning if you don't add the flag it WILL
# perform those things
args = parser.parse_args()
parser.set_defaults(no_tiled=True)

# set variables from parser
csv_path = args.csv_path
out_dir = args.output_directory
altmin = args.altmin  # the minimum altitude for images to be included in processing
n_alt_levels = args.n_alt_levels
tiled = args.no_tiled


#################################################################
#################################################################


    
#%% Main

def main():

    csv_out_path = agisoft_processing(csv_path, out_dir, altmin=altmin, n_alt_levels=n_alt_levels)
    
    print("The CSV path to be used as input for the 'Workflow_postprocess script is:")
    print(csv_out_path)
         
if __name__ == '__main__':
    main()    