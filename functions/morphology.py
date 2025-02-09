import cv2    
import numpy as np   
from copy import copy   
from tqdm import tqdm
from numpy import linalg as LA


def compute_inertia(mask_def):
    
    '''
    This function compute the inertia matrix components of an image
    '''
    
    mask = copy(mask_def)
    
    th, im_th = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0]
        
    if len(contours) >0:
        
            # If there is multiple contours (error) , we keep the bigger one
            idx = 0
            L_area = np.zeros((len(contours),1))
            for j in range(len(contours)):
                L_area[j] =  cv2.contourArea(contours[j])
            idx = np.argmax(L_area)
            cont = contours[idx]
            
            # Inertia computation
            xtot = cont[:,0,0]
            ytot = cont[:,0,1]
            
            x_center = np.mean(xtot)
            y_center = np.mean(ytot)
            
            x_centered = xtot - x_center
            y_centered = ytot - y_center
            
            Ixx = np.mean( np.multiply(x_centered,x_centered))
            Iyy = np.mean( np.multiply(y_centered,y_centered))
            Ixy = np.mean( np.multiply(x_centered,y_centered))
    
            return Ixx,Ixy,Iyy
    
    else:
        return 9999999,9999999,9999999
    
    
def compute_strain(Ixx,Ixy,Iyy):
    
    '''
    This function compute the strain from the inertia matrix
    '''
        
    Inertia = np.array([[Ixx,Ixy],[Ixy,Iyy]])
    
    eVa, eVe = LA.eig(Inertia)
    
    if eVa[0] > eVa[1]:
        a = eVa[0]
        b = eVa[1]
    else:
        a = eVa[1]
        b = eVa[0]
                    
    return np.log(a/b)/2

def compute_area_and_inertia(Path_tiff,nb_frames):
    
    '''
    This function compute the cells area, shape(Inertia matrix) and strain
    '''
        
    L_cell_area = []
    L_Ixx = []
    L_Ixy = []
    L_Iyy = []
    L_strain = []
    
    Path_analysis = Path_tiff.replace('.tif', '')
      
    for k in tqdm(range(nb_frames)):
        
            # Load img
            im = cv2.imread(Path_analysis + "_Analysis/split_data/"+'{:04d}'.format(k)+'_masks.tif', cv2.IMREAD_UNCHANGED)
            im = np.array(im)
            uni = np.unique(im)
            uni = uni[np.where(uni !=0)]
            
            # Go trought each objects
            for iii in range(len(uni)):
                
                im_ = copy(im)
                
                # Isolate the object
                im_[np.where(im != uni[iii])] = 0
                im_[np.where(im == uni[iii])] = 1
                
                # Tresh the object
                ret, thresh = cv2.threshold(im_, 0, 255, 0)      
                contours, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Compute area
                copy_seg = copy(im)        
                copy_seg[np.where(im != uni[iii])] = 0
                copy_seg[np.where(im == uni[iii])] = 1
                cell_area = np.sum(copy_seg)
                
                # Compute inertia
                Ixx,Ixy,Iyy = compute_inertia(copy_seg)
                
                # Add to list
                if Ixx !=9999999:
                    L_cell_area.append(cell_area)
                    L_Ixx.append(Ixx)
                    L_Ixy.append(Ixy)
                    L_Iyy.append(Iyy)
                    
                    L_strain.append(compute_strain(Ixx,Ixy,Iyy))
                    
                
    return L_cell_area,L_Ixx,L_Ixy,L_Iyy,L_strain