from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist


import xml.etree.ElementTree as et
def loadxmlTrajs(xmlfile):
    # FROM Chenyu Jin at : https://forum.image.sc/t/reading-trackmate-xml-file-in-python/59262/2
    """ 
    Load xml files into a python dictionary with the following structure:
        tracks = {'0': {'nSpots': 20, 'trackData': numpy.array(t, x, y, z) }}
    Tracks should be xml file from 'Export tracks to XML file',
    that contains only track info but not the features.
    Similar to what 'importTrackMateTracks.m' needs.
    """
    tree = et.parse(xmlfile);
    try:
        tree = et.parse(xmlfile);
    except OSError:
        print('Failed to read XML file {}'.format(xmlfile) )
    root =  tree.getroot()
    # print(root.attrib)  # or extract metadata
    nTracks = int(root.attrib['nTracks'])
    tracks = {}
    for i in range(nTracks):
        trackIdx = str(i)
        tracks[trackIdx] = {}
        nSpots = int(root[i].attrib['nSpots'])
        tracks[trackIdx]['nSpots'] = nSpots
        trackData = np.array([ ]).reshape(0, 4)
        for j in range(nSpots):
            t = float(root[i][j].attrib['t'])
            x = float(root[i][j].attrib['x'])
            y = float(root[i][j].attrib['y'])
            z = float(root[i][j].attrib['z'])
            spotData = np.array([t, x, y, z])
            trackData = np.vstack((trackData, spotData))
        tracks[trackIdx]['trackData'] = trackData
    return tracks,nTracks
    # Dump the dictionary with json or pickle it as you want
    
def velocities(Data_folder,nb_frames,treshold_lineage):

    xmlfile =  Data_folder + "Tracks.xml"
    tracks,nTracks = loadxmlTrajs(xmlfile)

    V = []
    
    for i in (range(nTracks)):
            
            arr = tracks[str(i)]['trackData']
            
            # Filter too short lineages
            if tracks[str(i)]['nSpots']>treshold_lineage: 
    
                dx = np.concatenate(([0],arr[:,1]))-np.concatenate((arr[:,1],[0]))
                dx = dx[1:-1]
                
                dy = np.concatenate(([0],arr[:,2]))-np.concatenate((arr[:,2],[0]))
                dy = dy[1:-1]
                
                for kk in range(len(dy)):
                    V.append(np.sqrt(dx[kk]*dx[kk]+dy[kk]*dy[kk]))
                
    return V


def autocorrelation_velocity(Data_folder,nb_frames,treshold_lineage):

    xmlfile =  Data_folder + "/Tracks.xml"
    tracks,nTracks = loadxmlTrajs(xmlfile)
        
    Corr = np.zeros((nTracks,nb_frames))
        
    for i in (range(nTracks)):
            
            arr = tracks[str(i)]['trackData']
            
            # Filter too short lineages
            if tracks[str(i)]['nSpots']>treshold_lineage: 
    
                dx = np.concatenate(([0],arr[:,1]))-np.concatenate((arr[:,1],[0]))
                dx = dx[1:-1]
                
                dy = np.concatenate(([0],arr[:,2]))-np.concatenate((arr[:,2],[0]))
                dy = dy[1:-1]
                
                for kk in range(len(dy)):
                    Corr[i,kk] = dx[0]*dx[kk]+dy[0]*dy[kk]
                
    Corr[Corr == 0] = np.nan
    Corr = np.nanmean(Corr, axis=0)

    return Corr
            

def spatial_correlation_velocity(Data_folder,nb_frames,treshold_lineage):
    
    step_dist = 10
    
    xmlfile =  Data_folder + "/Tracks.xml"
    tracks,nTracks = loadxmlTrajs(xmlfile)
      
    Lx = []
    Ly = []
    Ldx = []
    Ldy = []
    Lframe = []
        
    for i in (range(nTracks)):
        # Filter too short lineages
        arr = tracks[str(i)]['trackData']
        if tracks[str(i)]['nSpots']>treshold_lineage: 
            dx = np.concatenate(([0],arr[:,1]))-np.concatenate((arr[:,1],[0]))
            dx = dx[1:-1]
            dy = np.concatenate(([0],arr[:,2]))-np.concatenate((arr[:,2],[0]))
            dy = dy[1:-1]
            for kk in range(len(dy)):
                Ldx.append(dx[kk])
                Ldy.append(dy[kk])
                Lframe.append(arr[kk,0])
                Lx.append(arr[kk,1])
                Ly.append(arr[kk,2])
                    
    Lx = np.array(Lx)
    Ly = np.array(Ly)
    Ldx = np.array(Ldx)
    Ldy = np.array(Ldy)
    Lframe = np.array(Lframe) 
        
    Corr_mean_f = np.zeros((int(np.max(Lframe)),step_dist))
        
    for f in tqdm(range(int(np.max(Lframe)))):
        L_x = Lx[np.where(Lframe == f)]
        L_y = Ly[np.where(Lframe == f)]
        L_dx = Ldx[np.where(Lframe == f)]
        L_dy = Ldy[np.where(Lframe == f)]
        A = np.array([L_x,L_y]).T
        DIST = cdist(A,A)
        Coor_f = np.zeros((np.shape(DIST)[0],step_dist))
        for i in range(np.shape(DIST)[0]):
            dist = DIST[i]
            dx = L_dx[i]
            dy = L_dy[i]
            for j in range(step_dist):
                wh1 = np.where(dist>step_dist*(j))
                wh2 = np.where(dist<step_dist*(j+1))
                DX = L_dx[wh1 and wh2]
                DY = L_dy[wh1 and wh2]
                Corr = 0
                for kij in range(len(DX)):
                    Corr = Corr + dx*DX[kij]+dy*DY[kij]
                Corr = Corr/len(DX)
                Coor_f[i,j] = Corr
                    
        Corr_mean_f[f,:] = np.mean(Coor_f, axis=0)
    Corr_mean = np.mean(Corr_mean_f, axis=0)
    
    return Corr_mean