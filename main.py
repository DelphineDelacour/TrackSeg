import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import parameters

from functions.segmentation import appply_cellpose

from functions.morphology import compute_area_and_inertia

from functions.velocities_and_correlations import velocities
from functions.velocities_and_correlations import autocorrelation_velocity
from functions.velocities_and_correlations import spatial_correlation_velocity

from functions.hexatic_order import compute_hexatic_order_exp

from functions.triangular_order import compute_triangular_order_exp



""" Segmentation """
appply_cellpose(parameters.Path_tiff,parameters.diameter,parameters.model_type,parameters.channels,parameters.net_avg,parameters.augment,parameters.flow_threshold,parameters.cellprob_threshold,parameters.batch_size)


""" Morphology """
L_cell_area,L_Ixx,L_Ixy,L_Iyy,L_strain = compute_area_and_inertia(parameters.Path_tiff,parameters.nb_frames_to_analyze)


""" Velocities """
V = velocities(parameters.Data_folder,parameters.nb_frames_to_analyze,parameters.treshold_lineage)


""" Velocity autocorrelation """
auto_corr = autocorrelation_velocity(parameters.Data_folder,parameters.nb_frames_to_analyze,parameters.treshold_lineage)


""" Spatial correlation """
spatial_corr = spatial_correlation_velocity(parameters.Data_folder,parameters.nb_frames_to_analyze,parameters.treshold_lineage)


""" Hexatic order """
L_hexatic, L_hexatic_nb_of_el =  compute_hexatic_order_exp(parameters.Path_tiff,parameters.nb_frames_to_analyze,parameters.resize_dim,parameters.threshold_area,parameters.size_dilatation)

# To compute a mean hexatic order you have to compute the mean of the norm divide by the number of each neigbors
import numpy as np
mean_hexatic = np.nanmean(np.abs(np.divide(L_hexatic,L_hexatic_nb_of_el)))


""" Triangular order """
Ltriangular = compute_triangular_order_exp(parameters.Path_tiff,parameters.nb_frames_to_analyze,parameters.resize_dim,parameters.size_dilatation,parameters.threshold_area)

# To compute a mean triangular order you have to compute the sum of the norm of the elements divide by their number
import numpy as np
mean_triangular = np.sum(np.abs(Ltriangular))/len(Ltriangular)