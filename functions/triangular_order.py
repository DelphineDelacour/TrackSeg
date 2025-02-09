import cv2 
import numpy as np
from copy import copy
import matplotlib.pyplot as plt 
from tqdm import tqdm


def compute_vertices(Path_tiff,frames,resize_dim,size_dilatation,threshold_area):
    
    """ 
    Return a binary image that contain the vertices
    """
    
    # Definition of the structural element for the dilatation
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (size_dilatation,size_dilatation))
    
    # Load the segmented image
    Path_analysis = Path_tiff.replace('.tif', '')
    seg_name = Path_analysis + "_Analysis/split_data/"+'{:04d}'.format(frames)+'_masks.tif'
    seg =  cv2.imread(seg_name, cv2.IMREAD_UNCHANGED)
    
    # Resize to save computation times
    seg = cv2.resize(seg,resize_dim,interpolation = cv2.INTER_NEAREST)
        
    # Find all the "color" of each mask
    uni = np.unique(seg)
    uni = uni[np.where(uni != 0)]
    
    # Initialize the image that will contain vertices
    img_vertex = np.zeros(np.shape(seg))
    
    # Go trought each segmented cells 
    for k in range(len(uni)):
        
        # Copy to avoid reloading 
        seg_ = copy(seg)
        
        # Isolate the cell
        seg_[np.where(seg_ != uni[k])] = 0
        seg_[np.where(seg_ == uni[k])] = 1

        # We apply an area treshold
        if np.sum(seg_)>threshold_area:
            
            # Tresh the object and dilate it
            ret, thresh = cv2.threshold(seg_, 0, 255, 0)      
            mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, se)/255
            
            # Find the cells neigbourgs indexes
            indxs = np.unique(seg[np.where(mask == 1)])
            indxs = indxs[np.where(indxs != 0)]
            
            # Image that will contain intersection of dilated objects
            seg_inter = np.zeros(np.shape(seg))
            
            # Go trought neibouring j-cells of the cell k
            for j in range(len(indxs)):
                
                # Copy to avoid reload
                seg_ = copy(seg)
                
                # Isolate the neigbouring j-cell
                seg_[np.where(seg_ != indxs[j])] = 0
                seg_[np.where(seg_ == indxs[j])] = 1
                
                # Tresh the object and dilate it
                ret, thresh = cv2.threshold(seg_, 0, 255, 0)      
                mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, se)/255
                
                # Add the image to the intersection image
                seg_inter = seg_inter + mask
                        
            # If we have an intersection of cells during the dilatation
            if np.max(seg_inter)>1:
                # Set the value of the intersection to 1, 0 elewhere in the img
                seg_inter[np.where(seg_inter!=np.max(seg_inter))] = 0
                seg_inter[np.where(seg_inter==np.max(seg_inter))] = 1
                
                # We add the intersection to the image that contain vertices
                img_vertex = img_vertex + seg_inter
      
    # We set value of all vertices to 1
    img_vertex[np.where(img_vertex !=0)] = 1
    
    ## Show the computed image
    # plt.imshow(img_vertex)
    # plt.show()
    
    return img_vertex
    
    
    
def compute_triangular_order(Path_tiff,frames,resize_dim,size_dilatation,img_vertex):
    
    """ 
    Return a list containing the triangular order parameters of each vertex of the img
    """
    
    # List initialization
    Ltriangular = []
    
    # Definition of the structural element for the dilatation
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (size_dilatation,size_dilatation))
    
    # Load the data
    Path_analysis = Path_tiff.replace('.tif', '')
    seg_name = Path_analysis + "_Analysis/split_data/"+'{:04d}'.format(frames)+'_masks.tif'
    seg =  cv2.imread(seg_name, cv2.IMREAD_UNCHANGED)
    
    # Resize to save computation times
    seg = cv2.resize(seg,resize_dim,interpolation = cv2.INTER_NEAREST)
    
    # Tresh the vertices image to find every vertex
    ret, thresh = cv2.threshold(img_vertex, 0, 255, 0)      
    contours, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Go trought each vertex
    for cont in contours:

        try:
            
            # Copy to avoid reloading 
            img_vertex_ = copy(img_vertex)
            
            # Compute the centroid of the vertex
            M = cv2.moments(cont)
            vx = int(M["m10"] / M["m00"])
            vy = int(M["m01"] / M["m00"])
            
            # Create a binary image that contain the vertex
            ZE = np.zeros(np.shape(seg))
            cv2.drawContours(ZE, [cont[0]], 0, 255, 5) 
            
            ## Show the isolated vertex image
            # plt.imshow(ZE)
            # plt.show()
            
            # Find the neigbourgs cells indexes (cell to which the vertex belong)
            indxs = np.unique(seg[np.where(ZE == 255)])
            indxs = indxs[np.where(indxs != 0)]
            
            # Dilate each neigbourg cell and find their intersection
            intersection = np.zeros(np.shape(seg))
            
            for j in range(len(indxs)):
                
                # Copy to avoid reloading 
                seg_ = copy(seg)
                
                # isolate the cells
                seg_[np.where(seg != indxs[j])] = 0
                seg_[np.where(seg == indxs[j])] = 1
                
                # Tresh the object and dilate it
                ret, thresh = cv2.threshold(seg_, 0, 255, 0)      
                mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, se)/255
                
                # Add it to the intersection
                intersection = intersection + mask
                
            # If we have at least 1 intersection
            if np.max(intersection)>1:
                # Keep the intersection area and set to 0 the rest
                intersection[np.where(intersection<2)] = 0
                intersection[np.where(intersection>=2)] = 1
                    
            ## Show the intersection
            # plt.imshow(intersection)
            # plt.show()
            
            
            # Isolated the vertices that belong to this intersection
            img_vertex_[np.where(intersection == 0)] = 0
            
            ## Show the vertex and the neibgours vertices
            # plt.imshow(img_vertex_)
            # plt.show()
            
            # Get ride of the center vertex in this image:
            # Isolate the center and dilate it
            ret, thresh = cv2.threshold(ZE/255, 0, 255, 0)      
            mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, se)/255
            # Delete the center
            img_vertex_ = img_vertex_ - mask
            # Ensure none negative value
            img_vertex_[np.where(img_vertex_ == -1)] = 0
            
            ## Show only the neibgours vertices
            # plt.imshow(img_vertex_)
            # plt.show()
                
            # Get the centroids of those neibgours vertices
            ret, thresh = cv2.threshold(img_vertex_, 0, 255, 0)      
            C, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            # If some pixels had stay from the center vertex:
            # We delete object smaller than half the bigger vertex
            # other methods are possible but this one work
            
            # Get objects area and define the threshold
            L_area = np.zeros((len(C),1))
            for j in range(len(C)):
                L_area[j] =  cv2.contourArea(C[j])
            th_area = np.max(L_area)/2        
            
            
            # Get centroid of neigb vertices
            Lvx_neigb = []
            Lvy_neigb = []
            # Go trought vertices
            for c in C:
                if cv2.contourArea(c) > th_area: # ensure that this is a vertex
                    # Get its centroid
                    M = cv2.moments(c)
                    vxn = int(M["m10"] / M["m00"])
                    vyn = int(M["m01"] / M["m00"])
                    
                    # Add it to a list
                    Lvx_neigb.append(vxn)
                    Lvy_neigb.append(vyn)
                    
                    
            # Now we have the centroid of the vertex (vx,vy) and its neibgours (Lvx,Lvy)
        
            # Computation of the tiangular order associated to this vertex:
            triangular = 0
            # If we have at least 2 neigb (get rid of vertices in the borders)
            if len(Lvx_neigb)>2:
                # for each neigb
                for kkk in range(len(Lvx_neigb)):
                    # angle computation
                    angle = np.arctan2(Lvy_neigb[kkk]-vy,Lvx_neigb[kkk]-vx)
                    # sum
                    triangular = triangular +np.exp(0+1j*3*angle)
                # Divide by 3 and add to the list
                triangular = triangular/3
                Ltriangular.append(triangular)
            
        except:
            pass
    
    return Ltriangular
    


def compute_triangular_order_exp(Path_tiff,nb_frames,resize_dim,size_dilatation,threshold_area):
    
    """ 
    Return a list containing the triangular order parameters of each vertex of the exp
    """
    
    # List initialization
    Ltriangular = []
    
    # Go trought frames
    for frames in tqdm(range(nb_frames)):
        
        # Compute and return the images of the vertices
        img_vertex = compute_vertices(Path_tiff,frames,resize_dim,size_dilatation,threshold_area)
        # Compute the associated triangular order parameters of each vertex 
        Ltriangular_ = compute_triangular_order(Path_tiff,frames,resize_dim,size_dilatation,img_vertex)
        # Concatenate the lists
        Ltriangular = Ltriangular + Ltriangular_
        
    # Return the full list
    return Ltriangular