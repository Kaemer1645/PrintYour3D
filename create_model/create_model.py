#import from main libraries
import numpy as np
import os
import processing

#import delaunay algorithm
from scipy.spatial import Delaunay

#import from numpy-stl library
from .stl import mesh

#import from qgis library
from qgis.utils import iface
from qgis.PyQt.QtWidgets import QProgressBar, QProgressDialog
from qgis.PyQt.QtCore import QTimer
from qgis.core import QgsProject, QgsMapLayer

#import matplotlib to create graph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri




#that class have got entire algorithm to create a model

class Create_model:

    def __init__(self, dlg, current_layer):

        self.layers = current_layer
        self.bar = QProgressBar()
        self.dlg = dlg

        self.list = []
        self.list_of_all=[]
        self.border = []

        self.X = []
        self.Y = []
        self.Z = []
        self.faces = []
        self.points = []

    def iterator(self):

        self.layer = self.dlg.cmbSelectLayer.currentLayer()

        #get the extent of layer
        provider = self.layer.dataProvider()
        extent = provider.extent()
        xmin = extent.xMinimum()
        ymax = extent.yMaximum()
        rows = self.layer.height()
        cols = self.layer.width()
        xsize = self.layer.rasterUnitsPerPixelX()
        ysize = self.layer.rasterUnitsPerPixelY()
        xinit = xmin + xsize / 2
        yinit = ymax - ysize / 2
        block = provider.block(1, extent, cols, rows)

        #iterate the raster to get the values of pixels
        k=1
        for i in range(rows):
            for j in range(cols):
                x = xinit + j * xsize
                y = yinit
                k += 1
                if block.value(i, j)  == -3.4028234663852886e+38: #, block.value(zewL,j) , block.value(zewP,j)]
                    self.list_of_all.append([i,j,x,y,block.value(i,j)])
                elif block.value(i,j)>=self.dlg.dsbDatum.value():
                    self.list.append([x, y, block.value(i, j)])
                    self.list_of_all.append([i,j,x,y,block.value(i,j)])
                else:
                    self.list_of_all.append([i,j,x,y,-3.4028234663852886e+38])
            xinit = xmin + xsize / 2
            yinit -= ysize

        #get minimal value to stretching method
        height=[]
        for searching in self.list:
                height.append(searching[2])
        self.minimal=min(height)

        colrow=[]
        rowcol=[]
        for pixelo in self.list_of_all:
            rowcol.append(pixelo)
            if pixelo[1]==j:
                colrow.append(rowcol)
                rowcol=[]
        for pixel in self.list_of_all:
            if pixel[4] !=-3.4028234663852886e+38:
                if pixel[0]==0 or pixel[1]==0 or pixel[0]==i or pixel[1]==j:
                    pixel[4] = float(self.dlg.dsbDatum.value())
                    self.border.append([pixel[2],pixel[3],pixel[4]])
                    self.list = self.border + self.list
                else:
                    wii=pixel[0]
                    kol=pixel[1]
                    pixel[4]=float(self.dlg.dsbDatum.value())
                    condition1 = colrow[wii-1][kol][4]
                    condition2 = colrow[wii][kol-1][4]
                    condition3 = colrow[wii+1][kol][4]
                    condition4 = colrow[wii][kol+1][4]

                    if condition1> -3.4028234663852886e+38 or condition2> -3.4028234663852886e+38 or condition3> -3.4028234663852886e+38 or condition4 > -3.4028234663852886e+38:
                        if condition1 == -3.4028234663852886e+38or condition2== -3.4028234663852886e+38 or condition3== -3.4028234663852886e+38 or condition4 == -3.4028234663852886e+38:
                            self.border.append([pixel[2],pixel[3],pixel[4]])
                            self.list=self.border+self.list

        return self.list, self.minimal


    def delaunay(self):

        for x in self.list:
            x_cord = x[0]
            self.X.append(x_cord)
            y_cord = x[1]
            self.Y.append(y_cord)
            z_cord = x[2]
            self.Z.append(z_cord)
        self.x = np.array(self.X)
        self.y = np.array(self.Y)
        self.z = np.array(self.Z)

        # Delaunay algorithm

        self.tri = Delaunay(np.array([self.x, self.y]).T, qhull_options=None)
        for vert in self.tri.simplices:
            self.faces.append([vert[0], vert[1], vert[2]])
        for i in range(self.x.shape[0]):
            self.points.append([self.x[i], self.y[i], self.z[i]])

        return self.faces, self.points, self.x, self.y, self.z, self.tri

    def saver(self):
        # Define the vertices of the cube
        vertices = np.array(self.points)
        # Define the triangles composing the cube
        facess = np.array(self.faces)
        # Create the mesh
        self.cube = mesh.Mesh(np.zeros(facess.shape[0], dtype=mesh.Mesh.dtype))
        all_percentage = len(facess)
        value = self.bar.value()
        for i, f in enumerate(facess):
            if value < all_percentage:
                value = value + 1
                self.bar.setValue(value)
            else:
                self.timer.stop()
            for j in range(3):
                self.cube.vectors[i][j] = vertices[f[j], :]

        filename=self.dlg.lineEdit.text()
        self.cube.save(filename)
        return self.cube

    def create_graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_trisurf(self.x, self.y, self.z, triangles=self.tri.simplices, cmap=plt.cm.Spectral)
        ax.set_title('3D_Graph')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def stretching(self):
        for cords in self.list:
            if cords[2] > self.minimal:
                height_stretched = cords[2] - float(self.dlg.dsbDatum.value())
                height_stretched = height_stretched * self.dlg.sbStretch.value()
                height_stretched += float(self.dlg.dsbDatum.value())
                cords[2] = height_stretched
        return self.list

    def shape(self, direction):
        self.plugin_dir = direction
        buffer_distance = self.dlg.dsbBuffer.value()
        output_data_type = self.dlg.sbOutputData.value()
        output_raster_size_unit = self.dlg.sbOutputSizeUnit.value()
        no_data_value = self.dlg.sbNoData.value()
        layer2 = self.dlg.cmbSelectShape.currentLayer()
        shape_dir = os.path.join(self.plugin_dir, 'TRASH')

        #selectedLayerIndex = self.dlg.cmbSelectLayer.currentIndex()
        #layer_ext_cor = self.layers[selectedLayerIndex].layer()
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
        # field_index = layer4.fields().indexFromName(field_name)
        field_namess = [field.name() for field in layer3.fields()]  # jak podac tutaj plik shape_drape
        i = 0
        for l in range(100):
            i += 1
            if field_name in field_namess:
                print('Exist')
                field_name = field_name + str(i)
                continue
            else:
                print('Doesn\'t Exist')
                # field_name == field_name
                break

        processing.run("qgis:fieldcalculator", {
            'INPUT': os.path.join(shape_dir, 'shape_drape.shp'),
            'FIELD_NAME': field_name, 'FIELD_TYPE': 0, 'FIELD_LENGTH': 10, 'FIELD_PRECISION': 3, 'NEW_FIELD': True,
            'FORMULA': 'z($geometry)+2', 'OUTPUT': os.path.join(shape_dir, 'shape_drape_c.shp')})

        processing.run("gdal:rasterize", {
            'INPUT': os.path.join(shape_dir, 'shape_drape_c.shp'),
            'FIELD': field_name, 'BURN': 0, 'UNITS': output_raster_size_unit, 'WIDTH': cols, 'HEIGHT': rows,
            # width heighy ustawic automatycznie do glownej rozdzielczosci
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
        #print(shape_to_raster[0].id())
        QgsProject.instance().removeMapLayers([shape_to_raster[0].id()])
    def loading(self):
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



