import os 
from tqdm import tqdm
from cellpose import  models, io
import numpy as np


def appply_cellpose(Path_tiff,diameter,model_type,channels,net_avg,augment,flow_threshold,cellprob_threshold,batch_size):
    
    '''
    This function split the tiff and apply cellpose on each frames
    '''
    
    Path_analysis = Path_tiff.replace('.tif', '')
    
    # Create the folders
    try:
        os.mkdir(Path_analysis + "_Analysis")
    except:
        print("Warning, process_mask directory:'"+Path_analysis+ "_Analysis"+"' already exist")
        
    try:
        os.mkdir(Path_analysis + "_Analysis/split_data")
    except:
        print("Warning, process_mask directory:'"+Path_analysis+ "_Analysis/split_data"+"' already exist")

    # Split and save the data
    print(Path_tiff)
    im = io.imread(Path_tiff)
    if len(np.shape(im))>2:
        le = im.shape[0]
    else:
        le = 1
    for k in range(le):
        if len(np.shape(im))>2:
            im_ = im[k,:,:]
        if len(np.shape(im))<3:
            im_ = im[:,:]
        io.imsave(Path_analysis+ "_Analysis/split_data/"+'{:04d}'.format(k)+".tif",im_)
        

    Split_data_folder = Path_analysis + "_Analysis/split_data/"
    
    # Apply Cellpose network               
    image_names = io.get_image_files(Split_data_folder,'_masks', imf=None)
    model = models.CellposeModel(model_type=model_type,gpu = True)             
        
    count = 0
    for image_name in tqdm(image_names):
              image = io.imread(image_name)
              out = model.eval(image, diameter=diameter,
                                          channels = channels,
                                          net_avg=net_avg,
                                          augment=augment,
                                          flow_threshold=flow_threshold,
                                          cellprob_threshold=cellprob_threshold,
                                          invert=False,
                                          batch_size=batch_size)
              masks, flows = out[:2]
              
              # Save the prediction
              io.imsave(Split_data_folder+'{:04d}'.format(count)+"_masks.tif",masks)
              count = count + 1
              
    print("\n Segmentation is done")