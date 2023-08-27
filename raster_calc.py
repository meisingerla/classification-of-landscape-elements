from qgis.core import *
from osgeo import gdal, gdalconst
import helpers_preprocessing #as ...
import numpy as np
import paths
import matplotlib.pyplot as plt

'''Generates a new GeoTiff out of an existing raster, where all white values (255 in all 3 bands) are set np.nan.
Nan-values are considered to be NoDataValues and are not visible in QGIS.
Add generated geotiff in QGIS: Layer -> Add Layer -> Add Raster Layer...
Set color range of the bands: Right click on the added raster -> Properties -> Symbology: set min=0 and max=255 for all 3 bands.'''

raster = gdal.Open(paths.raster2_knobloch, gdalconst.GA_Update)
band1 = raster.GetRasterBand(1)
band2 = raster.GetRasterBand(2)
band3 = raster.GetRasterBand(3)
array1 = band1.ReadAsArray()
array2 = band2.ReadAsArray()
array3 = band3.ReadAsArray()
new_array1 = np.zeros(array1.shape)
new_array2 = np.zeros(array1.shape)
new_array3 = np.zeros(array1.shape)
for i in range(array1.shape[0]):
    print(i, '/', array1.shape[0]-1)
    for j in range(array1.shape[1]):
        if array1[i][j]==255 and array2[i][j]==255 and array3[i][j]==255:
            new_array1[i][j] = np.nan
            new_array2[i][j] = np.nan
            new_array3[i][j] = np.nan
        else:
            new_array1[i][j] = array1[i][j]
            new_array2[i][j] = array2[i][j]
            new_array3[i][j] = array3[i][j]


print(new_array1)
fn = '..\\Luftbilder-qgis\\Rhein_Knoblochsaue\\upper_raster_nan.tif'
driver = gdal.GetDriverByName('GTiff')
ds = driver.Create(fn, xsize=raster.RasterXSize, ysize=raster.RasterYSize, bands=3, eType=gdal.GDT_Float32)
new_band1 = ds.GetRasterBand(1).WriteArray(new_array1)
new_band2 = ds.GetRasterBand(2).WriteArray(new_array2)
new_band3 = ds.GetRasterBand(3).WriteArray(new_array3)

ds.SetGeoTransform(raster.GetGeoTransform())
ds.SetSpatialRef(raster.GetSpatialRef())
ds.SetProjection(raster.GetProjection())

ds = None

#Dauer: 2,5h
raster = gdal.Open(paths.raster1_knobloch, gdalconst.GA_Update)
band1 = raster.GetRasterBand(1)
band2 = raster.GetRasterBand(2)
band3 = raster.GetRasterBand(3)
array1 = band1.ReadAsArray()
array2 = band2.ReadAsArray()
array3 = band3.ReadAsArray()
new_array1 = np.zeros(array1.shape)
new_array2 = np.zeros(array1.shape)
new_array3 = np.zeros(array1.shape)
for i in range(array1.shape[0]):
    print(i, '/', array1.shape[0]-1)
    for j in range(array1.shape[1]):
        if array1[i][j]==255 and array2[i][j]==255 and array3[i][j]==255:
            new_array1[i][j] = np.nan
            new_array2[i][j] = np.nan
            new_array3[i][j] = np.nan
        else:
            new_array1[i][j] = array1[i][j]
            new_array2[i][j] = array2[i][j]
            new_array3[i][j] = array3[i][j]


print(new_array1)
fn = '..\\Luftbilder-qgis\\Rhein_Knoblochsaue\\lower_raster_nan.tif'
driver = gdal.GetDriverByName('GTiff')
ds = driver.Create(fn, xsize=raster.RasterXSize, ysize=raster.RasterYSize, bands=3, eType=gdal.GDT_Float32)
new_band1 = ds.GetRasterBand(1).WriteArray(new_array1)
new_band2 = ds.GetRasterBand(2).WriteArray(new_array2)
new_band3 = ds.GetRasterBand(3).WriteArray(new_array3)

ds.SetGeoTransform(raster.GetGeoTransform())
ds.SetSpatialRef(raster.GetSpatialRef())
ds.SetProjection(raster.GetProjection())

ds = None


#Probe: Eckkoordinaten des Rasters befinden sich trotzdem an nan-Werten, 
# also erkennt Programm keinen Fehler, wenn landscape element im nan-Bereich liegt.
# Daher in get_raster_of_landscape_element noch nan-Abfrage eingebaut.
raster1 = gdal.OpenEx('..\\Luftbilder-qgis\\Rhein_Knoblochsaue\\test3.tif')
raster2 = gdal.OpenEx(paths.raster1_knobloch)
raster3 = gdal.OpenEx(paths.raster2_knobloch)
raster = [raster1, raster2, raster3]
vector = gdal.OpenEx('..\\Luftbilder-qgis\\Rhein_Knoblochsaue\\manual_mapping.shp')
searching_critical_landscape_element=1
vector.ResetReading()
landscape_element = vector.GetNextFeature()[0] #liefert ein osgeo.ogr.Feature ([1] liefert zugehörige osgeo.ogr.Layer, bei mir immer gleich)
while searching_critical_landscape_element:
    idx_ref = landscape_element.GetFieldIndex('Ref')
    ref = landscape_element.GetField(idx_ref)
    if ref==54:
        idx_descriptio = landscape_element.GetFieldIndex('descriptio')
        descriptio = landscape_element.GetField(idx_descriptio)
        landscape_element_coordinates = helpers_preprocessing.get_landscape_element_coords(landscape_element)
        raster_of_landscape_element = helpers_preprocessing.get_raster_of_landscape_element(raster, landscape_element_coordinates, vector)
        landscape_element_indices = helpers_preprocessing.coords2pixels(landscape_element_coordinates, raster_of_landscape_element, vector)
        info = {'landscape element object': landscape_element, 'vector': vector, 'raster': raster_of_landscape_element, 'descriptio': descriptio, 'ref': ref, 'coordinates': landscape_element_coordinates, 'indices': landscape_element_indices}
        critical_landscape_element = info
        searching_critical_landscape_element=0
    landscape_element = vector.GetNextFeature()[0]
    
image_tile = helpers_preprocessing.cutout_around_landscape_element(raster_of_landscape_element, critical_landscape_element['indices'], (256,256))
image_tile = image_tile.astype('uint8')
plt.figure()
plt.imshow(image_tile)
plt.show()
