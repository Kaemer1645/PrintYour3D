# import from main libraries
import numpy as np
import os
import processing
from osgeo import gdal

# import from qgis library
from qgis.utils import iface
from qgis.PyQt.QtWidgets import QProgressBar, QProgressDialog
from qgis.PyQt.QtCore import QTimer
from qgis.core import QgsProject, QgsMapLayer

# import delaunay algorithm
from scipy.spatial import Delaunay

# import from numpy-stl library
from .stl import mesh

# import matplotlib to create graph
import matplotlib.pyplot as plt

# that class have got entire algorithm to create a model

class Create_model:
    def __init__(self, dlg, raster_layer):
        """This constructor need Qt Dialog  class as dlg and need current layer to execute the algorithm
        in current_layer parameter"""

        self.raster_layer = raster_layer
        self.bar = QProgressBar()
        self.dlg = dlg

        self.border = []

        self.X = []
        self.Y = []
        self.Z = []
        self.faces = []
        self.points = []

    def iterator(self):
        """Method to get values of raster with outer snapped to the reference level"""
        path_to_raster_layer = self.raster_layer.source()

        data_source = gdal.Open(path_to_raster_layer)

        # extract one band, because we don't need 3d matrix
        band = data_source.GetRasterBand(1)

        # read matrix as numpy array
        raster = band.ReadAsArray().astype(np.float)

        threshold = self.dlg.dsbDatum.value()  # get raster base lavel

        # change no data to nan value
        raster[raster == band.GetNoDataValue()] = np.nan
        raster2 = raster[np.logical_not(np.isnan(raster))]  # w raster2 nie mam nanow i sa to tylko wartosci wysokosci
        (y_index, x_index) = np.nonzero(raster >= threshold)

        # get the minimal value to stretching method
        self.minimal = np.nanmin(raster)

        # To demonstate this compare a.shape to band.XSize and band.YSize
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = data_source.GetGeoTransform()

        x_coords = x_index * x_size + upper_left_x + (x_size / 2)  # add half the cell size
        y_coords = y_index * y_size + upper_left_y + (y_size / 2)  # to centre the point
        raster3 = raster2[raster2 >= threshold]
        z_coords = np.asarray(raster3).reshape(-1)

        entire_matrix = np.stack((x_coords, y_coords, z_coords), axis=-1)

        # add outer
        bounder = np.pad(raster, pad_width=1, mode='constant', constant_values=np.nan)
        bounder_inner = (
                    np.roll(bounder, 1, axis=0) * np.roll(bounder, -1, axis=0) * np.roll(bounder, 1, axis=1) * np.roll(
                bounder, -1, axis=1) * np.roll(np.roll(bounder, 1, axis=0), 1, axis=1)
                    * np.roll(np.roll(bounder, 1, axis=0), -1, axis=1) * np.roll(np.roll(bounder, -1, axis=0), 1,
                                                                                 axis=1) * np.roll(
                np.roll(bounder, -1, axis=0), -1, axis=1))
        is_inner = (np.isnan(bounder_inner) == False)
        b = bounder
        b[is_inner] = np.nan
        b[~np.isnan(b)] = threshold
        boundary_real = b[1:-1, 1:-1]

        boundary_real_2 = boundary_real[np.logical_not(np.isnan(boundary_real))]

        # create boundary coordinates
        (y_index_boundary, x_index_boundary) = np.nonzero(boundary_real == threshold)

        # print(len(boundary_real_2))
        x_coords_boundary = x_index_boundary * x_size + upper_left_x + (x_size / 2)  # add half the cell size
        y_coords_boundary = y_index_boundary * y_size + upper_left_y + (y_size / 2)  # to centre the point
        z_coords_boundary = np.asarray(boundary_real_2).reshape(-1)

        boundary_the_end = np.stack((x_coords_boundary, y_coords_boundary, z_coords_boundary), axis=-1)
        boundary_the_end = np.repeat(boundary_the_end, 10, axis=0)

        self.entire_mat_with_heights = np.concatenate((entire_matrix, boundary_the_end))
        # entire_mat_with_heights[entire_mat_with_heights[:, [0,1,2]].argsort()]
        self.entire_mat_with_heights = self.entire_mat_with_heights[np.argsort(self.entire_mat_with_heights[:, 2])]

    def delaunay(self):
        """This is Delaunay algorithm from Scipy lib
        This is needed to create vertices and faces which will be going to executed in creating STL model"""

        self.x = self.entire_mat_with_heights[:, 0]
        self.y = self.entire_mat_with_heights[:, 1]
        self.z = self.entire_mat_with_heights[:, 2]

        # print(self.x.shape)
        self.tri = Delaunay(np.array([self.x, self.y]).T)
        for vert in self.tri.simplices:
            self.faces.append([vert[0], vert[1], vert[2]])
        for i in range(self.x.shape[0]):
            self.points.append([self.x[i], self.y[i], self.z[i]])

    def saver(self):
        """ Create STL model """

        # Define the vertices
        vertices = np.array(self.points)
        # Define the triangles
        facess = np.array(self.faces)
        # Create the mesh
        self.figure = mesh.Mesh(np.zeros(facess.shape[0], dtype=mesh.Mesh.dtype))
        all_percentage = len(facess)
        value = self.bar.value()
        for i, f in enumerate(facess):
            if value < all_percentage:
                value = value + 1
                self.bar.setValue(value)
            else:
                self.timer.stop()
            for j in range(3):
                self.figure.vectors[i][j] = vertices[f[j], :]

        filename = self.dlg.lineEdit.text()
        self.figure.save(filename)

    def create_graph(self):
        """ Visualize model by Matplotlib lib"""

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_trisurf(self.x, self.y, self.z, triangles=self.tri.simplices, cmap=plt.cm.Spectral)
        ax.set_title('3D_Graph')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def stretching(self):
        """ This method stretching the entire model to the given Datum level"""
        for cords in self.entire_mat_with_heights:
            if cords[2] > self.minimal:
                height_stretched = cords[2] - float(self.dlg.dsbDatum.value())
                height_stretched = height_stretched * self.dlg.sbStretch.value()
                height_stretched += float(self.dlg.dsbDatum.value())
                cords[2] = height_stretched

    def loading(self):  # stworzyc calkowicie na nowo
        """ Loading progress bar """

        self.dialog = QProgressDialog()
        self.dialog.setWindowTitle("Loading")
        self.dialog.setLabelText("That's your progress")
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.dialog.setBar(self.bar)
        self.dialog.setMinimumWidth(300)
        self.dialog.show()
        self.timer = QTimer()
        self.timer.timeout.connect(self.saver)
        self.timer.start(1000)

    def shape(self, direction):
        """ This algorithm convert ShapeFile to the .tif file.
        To created that process I used a lot of GDAL processing algorithms
         and this gave me possibility to convert the shape to .tif file,
         drape them to the correct elevation and merge this new raster to the current main tif """

        self.plugin_dir = direction
        buffer_distance = self.dlg.dsbBuffer.value()
        output_data_type = self.dlg.sbOutputData.value()
        output_raster_size_unit = 0
        no_data_value = 0
        layer2 = self.dlg.cmbSelectShape.currentLayer()
        shape_dir = os.path.join(self.plugin_dir, 'TRASH')

        layer_ext_cor = self.dlg.cmbSelectLayer.currentLayer()
        provider = layer_ext_cor.dataProvider()
        extent = provider.extent()
        xmin = extent.xMinimum()
        ymax = extent.yMaximum()
        xmax = extent.xMaximum()
        ymin = extent.yMinimum()
        cord_sys = layer_ext_cor.crs().authid()
        coords = "%f,%f,%f,%f " % (xmin, xmax, ymin, ymax) + '[' + str(cord_sys) + ']'
        rows = layer_ext_cor.height()
        cols = layer_ext_cor.width()
        set_shp_height = self.dlg.sbHeight.value()

        processing.run("native:buffer",
                       {'INPUT': layer2, 'DISTANCE': buffer_distance, 'SEGMENTS': 5, 'END_CAP_STYLE': 0,
                        'JOIN_STYLE': 0,
                        'MITER_LIMIT': 2, 'DISSOLVE': False,
                        'OUTPUT': os.path.join(shape_dir,
                                               'shape_bufor.shp')})  #### tu poprawic ten dissolve \/ zredukowac ten na dole

        processing.run("native:dissolve", {
            'INPUT': os.path.join(shape_dir, 'shape_bufor.shp'),
            'FIELD': [],
            'OUTPUT': os.path.join(shape_dir, 'shape_dissolve.shp')})

        processing.run("qgis:generatepointspixelcentroidsinsidepolygons",
                       {'INPUT_RASTER': self.dlg.cmbSelectLayer.currentLayer().dataProvider().dataSourceUri(),
                        'INPUT_VECTOR': os.path.join(shape_dir, 'shape_dissolve.shp'),
                        'OUTPUT': os.path.join(shape_dir, 'shape_points.shp')})

        processing.run("native:setzfromraster",
                       {'INPUT': os.path.join(shape_dir, 'shape_points.shp'),
                        'RASTER': self.dlg.cmbSelectLayer.currentLayer().dataProvider().dataSourceUri(),
                        'BAND': 1, 'NODATA': 0, 'SCALE': 1,
                        'OUTPUT': os.path.join(shape_dir, 'shape_drape.shp')})

        layer3 = iface.addVectorLayer(os.path.join(shape_dir, 'shape_drape.shp'), "Shape_Drape", "ogr")
        if not layer3:
            print("Layer failed to load!")
        field_name = "height"
        field_namess = [field.name() for field in layer3.fields()]
        i = 0
        for l in range(100):
            i += 1
            if field_name in field_namess:
                print('Exist')
                field_name = field_name + str(i)
                continue
            else:
                print('Doesn\'t Exist')
                break

        processing.run("qgis:fieldcalculator", {
            'INPUT': os.path.join(shape_dir, 'shape_drape.shp'),
            'FIELD_NAME': field_name, 'FIELD_TYPE': 0, 'FIELD_LENGTH': 10, 'FIELD_PRECISION': 3, 'NEW_FIELD': True,
            'FORMULA': 'z($geometry)+' + str(set_shp_height), 'OUTPUT': os.path.join(shape_dir, 'shape_drape_c.shp')})

        processing.run("gdal:rasterize", {
            'INPUT': os.path.join(shape_dir, 'shape_drape_c.shp'),
            'FIELD': field_name, 'BURN': 0, 'UNITS': output_raster_size_unit, 'WIDTH': cols, 'HEIGHT': rows,
            'EXTENT': coords, 'NODATA': no_data_value, 'OPTIONS': '', 'DATA_TYPE': output_data_type,
            'INIT': None, 'INVERT': False,
            'OUTPUT': os.path.join(shape_dir, 'shape_to_raster.tif')})
        iface.addRasterLayer(os.path.join(shape_dir, 'shape_to_raster.tif'), "Shape_to_Raster")
        QgsProject.instance().removeMapLayers([layer3.id()])

        processing.run("gdal:merge", {
            'INPUT': [self.dlg.cmbSelectLayer.currentLayer().dataProvider().dataSourceUri(),
                      os.path.join(shape_dir, 'shape_to_raster.tif')],
            'PCT': False, 'SEPARATE': False, 'NODATA_INPUT': None, 'NODATA_OUTPUT': None, 'OPTIONS': '', 'DATA_TYPE': 5,
            'OUTPUT': os.path.join(shape_dir, 'merged.tif')})
        iface.addRasterLayer(os.path.join(shape_dir, 'merged.tif'), "Raster_With_Shape")

        shape_to_raster = QgsProject.instance().mapLayersByName('Shape_to_Raster')
        QgsProject.instance().removeMapLayers([shape_to_raster[0].id()])
