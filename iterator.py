from osgeo import gdal
import numpy as np
def iterator():
    path = iface.activeLayer().source()
    layer = iface.activeLayer()
    
    data_source = gdal.Open(path)
    #extract one band, because we don't need 3d matrix
    band = data_source.GetRasterBand(1)
    #read matrix as numpy array
    
    raster = band.ReadAsArray().astype(np.float)
    #a = band.ReadAsArray().astype(np.float)
    #print(raster)
    threshold = 222 #poziom odniesienia - wyzsze od 222
    
    #change no data to nan value
    raster[raster == band.GetNoDataValue()] = np.nan
    raster2 = raster[np.logical_not(np.isnan(raster))]

    (y_index, x_index) = np.nonzero(raster > threshold)
    #print(y_index, x_index)
    
    #To demonstate this compare a.shape to band.XSize and band.YSize
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = data_source.GetGeoTransform()
    
    x_coords = x_index * x_size + upper_left_x + (x_size / 2) #add half the cell size
    y_coords = y_index * y_size + upper_left_y + (y_size / 2) #to centre the point
    z_coords = np.asarray(raster2).reshape(-1)
    
    '''print(x_coords)
    print(y_coords)
    print(z_coords)'''
    #all = np.dstack((y_index, x_index))
    #print(all)
    list_of_all = np.stack((y_index, x_index,x_coords,y_coords,z_coords), axis=-1)
    print(list_of_all)
    #get the minimal value to stretching method
    minimal = np.nanmin(raster)
    
    entire_matrix = np.stack((x_coords,y_coords,z_coords), axis=-1)
    #print(entire_matrix)
    
    
iterator()