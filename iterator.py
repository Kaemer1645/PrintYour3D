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


    threshold = 222 #poziom odniesienia - wyzsze od 222
    
    #change no data to nan value
    raster[raster == band.GetNoDataValue()] = np.nan
    raster2 = raster[np.logical_not(np.isnan(raster))]
    #print(raster)
    #print(raster)
    #print('---')
    #print(raster2)
    (y_index, x_index) = np.nonzero(raster > threshold)
    #print(y_index, x_index)
    
    #To demonstate this compare a.shape to band.XSize and band.YSize
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = data_source.GetGeoTransform()
    
    x_coords = x_index * x_size + upper_left_x + (x_size / 2) #add half the cell size
    y_coords = y_index * y_size + upper_left_y + (y_size / 2) #to centre the point
    z_coords = np.asarray(raster2).reshape(-1)
    
    #print(x_coords)
    #print(y_coords)
    #print(z_coords)
    #print(len(x_coords),len(y_coords),len(z_coords))

    #all = np.dstack((y_index, x_index))
    #print(all)
    #list_of_all = np.stack((y_index, x_index,x_coords,y_coords,z_coords), axis=-1)
    #print(list_of_all)
    #get the minimal value to stretching method
    minimal = np.nanmin(raster)
    
    entire_matrix = np.stack((x_coords,y_coords,z_coords), axis=-1)
    #print(entire_matrix)
    
    #boundaries
    
    #add outer
    bounder = np.pad(raster, pad_width = 1, mode='constant', constant_values=0)
    bounder[bounder == 0] = np.nan
    bounder_inner = np.roll(bounder, 1, axis = 0) * np.roll(bounder, -1, axis = 0) * np.roll(bounder, 1, axis = 1) * np.roll(bounder, -1, axis = 1)
    is_inner = (np.isnan(bounder_inner) == False)
    b = bounder
    b[is_inner] = np.nan
    b[~np.isnan(b)] = 200
    #print(b)
    boundary_real = b[1:-1,1:-1]
    #print('----')
    #print(boundary_real)
    
    boundary_real_2 = boundary_real[np.logical_not(np.isnan(boundary_real))]

    #create boundary coordinates
    (y_index_boundary, x_index_boundary) = np.nonzero(boundary_real==200)
    
    
    x_coords_boundary = x_index_boundary * x_size + upper_left_x + (x_size / 2) #add half the cell size
    y_coords_boundary = y_index_boundary * y_size + upper_left_y + (y_size / 2) #to centre the point
    z_coords_boundary = np.asarray(boundary_real_2).reshape(-1)
    #print(x_coords_boundary)
    #print(y_coords_boundary)
    #print(z_coords_boundary)
    #print(len(x_coords_boundary),len(y_coords_boundary),len(z_coords_boundary))
    boundary_the_end = np.stack((x_coords_boundary,y_coords_boundary,z_coords_boundary), axis=-1)
    #print(boundary_the_end)
    we_are_the_champions = np.concatenate((entire_matrix,boundary_the_end))
    print(we_are_the_champions)
iterator()