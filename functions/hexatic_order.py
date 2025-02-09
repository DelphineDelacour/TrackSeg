import cv2 
import numpy as np
from copy import copy
from tqdm import tqdm


def compute_hexatic_order_exp(Path_tiff,nb_frames,resize_dim,threshold_area,size_dilatation):
    
    '''
    This function compute the tissue hexatic order
    '''
    
    L_hexatic = []
    L_hexatic_nb_of_el = []
    
    Path_analysis = Path_tiff.replace('.tif', '')
      
    for k in tqdm(range(nb_frames)):
        
        # Load img
        im = cv2.imread(Path_analysis + "_Analysis/split_data/"+'{:04d}'.format(k)+'_masks.tif', cv2.IMREAD_UNCHANGED)
        im = cv2.resize(im,resize_dim,interpolation = cv2.INTER_NEAREST)
        im = np.array(im)
        uni = np.unique(im)
        uni = uni[np.where(uni !=0)]
        
        # Go through each color
        for j in (range(len(uni))):
            im_ = copy(im)
            im_copy = copy(im_)
            
            # Isolate the object
            im_[np.where(im_ != uni[j])] = 0
            im_[np.where(im_ == uni[j])] = 1
                        
            # Tresh and dilate the object
            ret, bw_img = cv2.threshold(im_,0,255,cv2.THRESH_BINARY)   
            se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (size_dilatation,size_dilatation))
            mask = cv2.morphologyEx(bw_img, cv2.MORPH_DILATE, se2)/255
            
            # Find object centroids
            cnt, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            M = cv2.moments(cnt[0])
            cellx = int(M["m10"] / M["m00"])
            celly = int(M["m01"] / M["m00"])    
            
            # Find the neigbourgs color
            indxs = np.unique(im_copy[np.where(mask == 1)])
            
            # Go trought the neiborgs and compute hexatic order 
            hex_order = np.array([0])
            nb_neigb_hex =0
            for ind in indxs:
                if ind != 0:
                    if ind != uni[j]:
            
                        copy_seg = copy(im)      
                        copy_seg[np.where(im != ind)] = 0
                        copy_seg[np.where(im == ind)] = 1  
                        
                        ret, th_seg = cv2.threshold(copy_seg, 0, 255, 0) 
                        contors, hierarchy = cv2.findContours(np.uint8(th_seg), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Compute hexatic order with each neigbourgs
                        try:
                                M = cv2.moments(contors[0])
                                x_neigb = int(M["m10"] / M["m00"])
                                y_neigb = int(M["m01"] / M["m00"])         
                                angl = np.arctan2((x_neigb-cellx),(y_neigb-celly))                            
                                hex_order = hex_order + np.array([np.exp(0+1j*6*angl)]) 
                                nb_neigb_hex = nb_neigb_hex +1 
                                
                        except:
                                pass
            if nb_neigb_hex !=0:                                               
                L_hexatic_nb_of_el.append(nb_neigb_hex)
                L_hexatic.append(hex_order[0])

    return np.array(L_hexatic),np.array(L_hexatic_nb_of_el)