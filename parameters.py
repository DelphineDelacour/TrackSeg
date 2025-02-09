""" Path """
Data_folder = "Data/"
# path to the .tif to analyze
Path_tiff= Data_folder +"DATA.tif"

""" Segmentation """
diameter = 100 # cell diameter in pixels
model_type = 'tissuenet' # tissuenet or cyto or nucleus
channels = [0,0] 
# First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue). 
# Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue). 
# For instance, to segment grayscale images, input [0,0]. 
# To segment images with cells in green and nuclei in blue, input [2,3].

net_avg = False # increase the precision but increase computing times
augment = False # increase the precision but increase computing times
cellprob_threshold = -12 # See cellpose paper, we use this value because every pixels belong to a cell
flow_threshold = -12 # See cellpose paper, we use this value because every pixels belong to a cell
batch_size = 8

""" Velocities and correlations """
nb_frames_to_analyze = 4
treshold_lineage = 1 # Keep only track longer than this in frames

""" Triangular order """
resize_dim = (512,512) # Reduce the img size for faster computation (You have to keep the ANISOTROPY of the original images)
size_dilatation = 5 # Kernel size for the dilatation procedure to find neigbors
threshold_area = 50 # Area treshold (in pixels) (after the resizing!)
