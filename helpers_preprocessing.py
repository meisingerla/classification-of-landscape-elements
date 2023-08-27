from osgeo import gdal
#from qgis.core import *
import logging
import numpy as np
import math
from osgeo import osr

'''Methods for pre-processing of the data'''

# Coordinate systems:
# x-y coordinate system: Position in a raster or vector layer described by coordinates; Origin in the bottom left corner, orientation of x and y depends on the orientation of the CRS.
# a-b coordinate system: Position in a pixel array described by indices; Origin in the top left corner, a runs horizontal to the right and b runs vertical downwards.
#                        Access in the array: array[b_start:b_end, a_start:a_end]

def open_raster(file_name, band_number=1):
    """From Flusstools Lib: Opens a raster file and accesses its bands.

    Args:
        file_name (str): The raster file directory and name.
        band_number (int): The Raster band number to open (default: ``1``).

    Returns:
        osgeo.gdal.Dataset: A raster dataset a Python object.
        osgeo.gdal.Band: The defined raster band as Python object.
    """
    gdal.UseExceptions()
    # open raster file or return None if not accessible
    try:
        raster = gdal.Open(file_name)
    except RuntimeError as e:
        logging.error("Cannot open raster.")
        print(e)
        return math.nan, math.nan
    # open raster band or return None if corrupted
    try:
        raster_band = raster.GetRasterBand(band_number)
    except RuntimeError as e:
        logging.error("Cannot access raster band.")
        logging.error(e)
        return raster, math.nan
    return raster, raster_band


def raster2array(raster, band_number, tile_origin, tile_size):
    """Modified version from Flusstools Lib: Extracts a numpy ``ndarray`` of the desired section from a raster.

    Args:
        raster (osgeo.gdal.Dataset): .tif file opened in gdal
        band_number (int): The raster band number to open
        tile_origin (tuple): Indices of the top left corner of the desired section (a-b coordinate system)
        tile_size (tuple): Shape of the numpy ``ndarray`` to be extracted

    Returns:
        list: three-elements of [``osgeo.DataSet`` of the original raster,
        ``numpy.ndarray`` following the input characteristics where no-data values are replaced with ``np.nan``,
        ``osgeo.GeoTransform`` of the original raster]
    """
    # open the band (see above)
    try:
        band = raster.GetRasterBand(band_number)
    except RuntimeError as e:
        logging.error("Cannot access raster band.")
        logging.error(e)
        return raster, math.nan
    
    try:
        # read array data from band
        band_array = band.ReadAsArray(tile_origin[0], tile_origin[1], tile_size[0], tile_size[1])
    except AttributeError:
        logging.error("Could not read array of raster band type=%s." %
                      str(type(band)))
        return raster, band, math.nan
    try:
        # overwrite NoDataValues with np.nan
        band_array = np.where(
            band_array == band.GetNoDataValue(), np.nan, band_array)
    except AttributeError:
        logging.error(
            "Could not get NoDataValue of raster band type=%s." % str(type(band)))
        return raster, band, math.nan
    # return the array and GeoTransformation used in the original raster
    return raster, band_array, raster.GetGeoTransform()


def get_corners(raster, vector):
    '''Determines the coordinates of two corners of the raster and transforms them into the CRS of the vector.

    Args:
        raster (osgeo.gdal.Dataset): .tif file opened with gdal
        vector (osgeo.gdal.Dataset): .shp file opened with gdal

    Returns:
        coordinates_top_left (tuple): Coordinates of the top left corner of the raster expressed in the vector CRS (x-y coordinate system)
        coordinates_bottom_right (tuple): Coordinates of the bottom right corner of the raster expressed in the vector CRS (x-y coordinate system)
    '''
    #determine corner coordinates
    geo_transform = raster.GetGeoTransform()
    top_left_x = geo_transform[0]
    top_left_y = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = geo_transform[5]
    bottom_right_x = top_left_x + (raster.RasterXSize * pixel_width)
    bottom_right_y = top_left_y + (raster.RasterYSize * pixel_height)

    #transform coordinates from the CRS of the raster to the CRS of the vector.
    source = raster.GetSpatialRef()
    target = vector.GetLayer().GetSpatialRef()
    transform = osr.CoordinateTransformation(source, target)
    top_left_x_transformed, top_left_y_transformed, z = transform.TransformPoint(top_left_x, top_left_y)
    bottom_right_x_transformed, bottom_right_y_transformed, z = transform.TransformPoint(bottom_right_x, bottom_right_y)
    coordinates_top_left = (top_left_x_transformed, top_left_y_transformed)
    coordinates_bottom_right = (bottom_right_x_transformed, bottom_right_y_transformed)
    return coordinates_top_left, coordinates_bottom_right


def coords2pixels(coordinates, raster, vector):
    '''Converts vector coordinates (x-y coordinate system) into indices (a-b coordinate system) of the associated raster array.
    
    Args:
        coordinates (tuple): Coordinates of a landscape element in the vector (x-y coordinate system)
        raster (osgeo.gdal.Dataset): .tif file opened with gdal
        vector (osgeo.gdal.Dataset): .shp file opened with gdal
    
    Returns:
        indices_pixel (tuple): Indices of the landscape element's position in the array deriving from the raster
    '''
    coordinates_top_left, coordinates_bottom_right = get_corners(raster, vector)
    
    pixel_size_a = raster.RasterXSize-1
    pixel_size_b = raster.RasterYSize-1
    coordinates_size_x = coordinates_bottom_right[0]-coordinates_top_left[0]
    coordinates_size_y = coordinates_top_left[1]-coordinates_bottom_right[1]
    if coordinates[0]<coordinates_top_left[0] or coordinates[0]>coordinates_bottom_right[0]:
        raise Exception("Coordinate is outside the area of the .tif file")
    if coordinates[1]<coordinates_bottom_right[1] or coordinates[1]>coordinates_top_left[1]:
        raise Exception("Coordinate is outside the area of the .tif file")
    index_pixel_a = (coordinates[0]-coordinates_top_left[0])*pixel_size_a/coordinates_size_x
    index_pixel_b = pixel_size_b-(coordinates[1]-coordinates_bottom_right[1])*pixel_size_b/coordinates_size_y
    indices_pixel = (int(index_pixel_a), int(index_pixel_b))

    return indices_pixel


def cutout_around_landscape_element(raster, indices_pixel, tile_size):
    '''Provides an image (in form of an array of pixel values) of the selected landscape element, cut out of the whole raster.
    If the tile around the landscape element transcends the data, the function returns an image of the desired shape, but along the data boundary.
    
    Args:
        raster (osgeo.gdal.Dataset): .tif file opened with gdal
        indices_pixel (tuple): Indices of the landscape element's position in the raster array (a-b coordinate system)
        tile_size (tuple): Shape of the required image tile

    Returns:
        image_tile (np.array): Pixel values of the required tile containing the landscape element
    '''
    # so landscape element immer mittig. Wird so belassen, da Markierung des landscape elements nicht unbedingt mittig und landscape element gewisse Ausmaße hat
    raster_size = (raster.RasterXSize, raster.RasterYSize)

    if indices_pixel[0]<0 or indices_pixel[0]>=raster_size[0] or indices_pixel[1]<0 or indices_pixel[1]>=raster_size[1]:
        raise Exception("Pixel indices of the landscape element lie outside of the data")

    if tile_size[0]>raster_size[0] or tile_size[1]>raster_size[1]:
        raise Exception("Image tile is larger than original image")

    if indices_pixel[0]-math.floor(tile_size[0]/2)<0:
        source_indices_pixel_a = 0
        print('Desired left edge transcends the data, pushed left edge to the left data boundary')
    elif indices_pixel[0]-math.floor(tile_size[0]/2)+tile_size[0]>raster_size[0]:
        source_indices_pixel_a = raster_size[0]-tile_size[0]
        print('Desired right edge transcends the data, pushed right edge to the right data boundary')
    else:
        source_indices_pixel_a = indices_pixel[0]-math.floor(tile_size[0]/2)

    if indices_pixel[1]-math.floor(tile_size[1]/2)<0:
        source_indices_pixel_b = 0
        print('Desired upper edge transcends the data, pushed upper edge to the upper data boundary')
    elif indices_pixel[1]-math.floor(tile_size[1]/2)+tile_size[1]>raster_size[1]:
        source_indices_pixel_b = raster_size[1]-tile_size[1]
        print('Desired lower edge transcends the data, pushed lower edge to the lower data boundary')
    else:
        source_indices_pixel_b = indices_pixel[1]-math.floor(tile_size[1]/2)
    
    source_indices_pixel = (source_indices_pixel_a, source_indices_pixel_b)

    
    band_colors = [('red', 1), ('green', 2), ('blue', 3)]
    from collections import OrderedDict
    image_tiles = OrderedDict()
    for (band_color, idx) in band_colors:
        _, image_tiles[band_color], _ = raster2array(raster, idx, source_indices_pixel, tile_size)
    image_tile = np.dstack(tuple(image_tiles.values()))
    #Alternative 1:
    # band_colors = ['red', 'green', 'blue']
    # image_tiles = []
    # for idx, band_color in enumerate(band_colors):
        # _, image_tile_one_color, _ = raster2array(raster, idx+1, source_indices_pixel, tile_size)
        # image_tiles.append(image_tile_one_color)
    # image_tile = np.dstack(tuple(image_tiles))
    #Alternative 2:
    # band_colors = ['red', 'green', 'blue']
    # image_tile = np.empty(tile_size)
    # for idx, _ in enumerate(band_colors):
    #     _, image_tile_one_color, _ = raster2array(raster, idx+1, source_indices_pixel, tile_size)
    #     image_tile = np.dstack((image_tile, image_tile_one_color))
    # image_tile = np.dsplit(image_tile, [1])[1]
    #Alternative 3:
    # _, image_tile_red, _ = raster2array(raster, 1, source_indices_pixel, tile_size)
    # _, image_tile_green, _ = raster2array(raster, 2, source_indices_pixel, tile_size)
    # _, image_tile_blue, _ = raster2array(raster, 3, source_indices_pixel, tile_size)
    # image_tile = np.dstack([image_tile_red, image_tile_green, image_tile_blue])

    return image_tile


def get_raster_of_landscape_element(list_of_rasters, landscape_element_coordinates, vector):
    '''Returns the raster on which the landscape element is located.
    If landscape element is inside more than one raster, the user gets informed but only one raster is returned.

    Args:
        list_of_rasters (list of osgeo.gdal.Dataset): List of all rasters in the project
        landscape_element_coordinates (tuple): Coordinates of the landscape element in the vector (x-y coordinate system)
        vector (osgeo.gdal.Dataset): .shp file opened with gdal, containing the landscape element

    Returns:
        corresponding_raster (osgeo.gdal.Dataset): Raster on which the landscape element is located
    '''
    corresponding_raster_found=0
    for raster in list_of_rasters:
        
        coordinates_top_left_corner_raster, coordinates_bottom_right_corner_raster = get_corners(raster, vector)
        if coordinates_bottom_right_corner_raster[1]<landscape_element_coordinates[1]<coordinates_top_left_corner_raster[1]:
            if coordinates_top_left_corner_raster[0]<landscape_element_coordinates[0]<coordinates_bottom_right_corner_raster[0]:
                #if corresponding_raster_found == 1:
                #    print('Landscape element is inside more than one raster')
                source_indices_pixel = coords2pixels(landscape_element_coordinates, raster, vector)
                _, image_tile, _ = raster2array(raster, 1, source_indices_pixel, (1,1))
                if not(math.isnan(image_tile[0][0])):
                    corresponding_raster = raster
                    corresponding_raster_found = 1
    if not(corresponding_raster_found):
        raise Exception('Landscape element is outside all known rasters')
    #print(corresponding_raster.GetDescription())

    return corresponding_raster

def get_landscape_element_coords(landscape_element):
    '''Returns the coordinates of the landscape element's position in the vector CRS.

    Args:
        landscape element (osggeo.ogr.Feature): Point in an vector layer

    Returns:
        landscape_element_coords (tuple):  (x-y coordinate system)
    '''
    geom = landscape_element.GetGeometryRef()
    if geom.GetGeometryName()!='POINT':
        raise Exception("Landscape element has no Point geometry")
    landscape_element_coords = (geom.GetX(), geom.GetY())

    return landscape_element_coords