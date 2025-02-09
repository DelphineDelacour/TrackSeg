
The folder "Data" contain the data in .tif format, the analysis folder and an .xml files which contain the trackmate tracking results
The folder "functions" contain the differents functions call by the main
To set the parameters use the "parameters.py" files
To launch the differents function use the "main.py" files

Require library: 
cellpose
cv2
numpy
tqdm
copy
scipy

Functions:

"appply_cellpose": Apply Cellpose to the .tif data file and create the segmentation in the Analysis folder

"compute_area_and_inertia": Compute and return the area, strain and inertia tensor component of each detected cells in lists

"velocities": Transform the .xml from stardist to an array and return the velocities in a list
	      A part is based from Chenyu Jin code (at : https://forum.image.sc/t/reading-trackmate-xml-file-in-python/59262/2)
 
"autocorrelation_velocity": Compute and return the velocity autocorrelation as an array

"spatial_correlation_velocity": Compute and return the velocity spatial correlation as an array

"compute_hexatic_order_exp": Compute and return the hexatic order and the number of neigbors of each cells as array

"compute_triangular_order_exp": Compute and return the triangular order of each cells as array
