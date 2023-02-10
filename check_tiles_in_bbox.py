import os, glob
import pandas as pd
from itertools import groupby

# must be in proper directory before importing
import micasense.metadata as metadata
import micasense.imageset as imageset


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


def identify_tiles_in_bbox(im_groups, bbox):
    """
    Filter images spatially and/or by altitude.\
        
    Parameters
    ----------
    im_groups : list
        Nested list of full paths of tiles pertaining to each capture.
    bbox : sequence, optional
        Bounding box tuple with the coordinates ordered as: ULX, ULY, LRX, LRY.
        The default is None.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing information about identified images
    sub_group: list
        list of image paths for identified images
    
    """
    
    sub_group = []
    for image in im_groups:
        
        # get image metadata
        meta = metadata.Metadata(image[0])#, exiftoolPath=exiftoolPath)
            
        
        # extract geographic coordinates and altitude
        lat, lon, alt = meta.position()
        if (lat is None) or (lon is None):
            print(f'One or more of the GPS coordinate values is null (lat={lat}, lon={lon}). Proceeding to next image.')
            continue
        
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

    print(f"{len(sub_group)} images in range identified")
    
    # check whether any matching image exists
    if len(sub_group) == 0:
        print("Empty list provided. Likely means no images in range.")
        return None, None  # to match tuple size of successful run
    # unnest the list of groups to pass to ImageSet method 
    imset_paths = [item for sublist in sub_group for item in sublist]
    # from_grouplist method was created by PNNL
    imgset = imageset.ImageSet.from_grouplist(imset_paths)
    
    data, columns = imgset.as_nested_lists()
    df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)

    return df, sub_group 

def make_csv_of_tiles_in_bbox(parent_dir, out_dir, bbox):
    dfs = []
    for image_dir in [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]:
        im_groups = create_nested_list_of_images(image_dir)    
    
        df, sub_group = identify_tiles_in_bbox(im_groups, bbox)
        if df is not None:
            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    # keep only base name of image (i.e. no band number) and drop _refl suffix
    df_all.paths = df_all["paths"].apply(lambda x: x[0].split('_refl')[0])
    
    csv_name = "tiles_in_bounds.csv"
    csv_out = os.path.join(out_dir, csv_name)
    
    df_all.to_csv(csv_out, index=False)

#%% RUN CODE

####################### USER VALUES ############################################
# bounding box for tile search
bbox = -97.48650656264057,36.605540630497984,-97.47493149725285,36.59725352873046
# directory containing tiles to search
parent_dir = r"C:\example\path\to\tiles\0000SET"
# output directory for csv containing name and info of tiles within bounds
out_dir = r"C:\example\output\directory\path"
###############################################################################
make_csv_of_tiles_in_bbox(parent_dir, out_dir, bbox)

# output will be file named "tiles_in_bounds.csv"
