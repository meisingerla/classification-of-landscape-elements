from qgis.core import *
import matplotlib.pyplot as plt
from classes_data_preprocessing import GISProject
from classes_data_preprocessing import FloodplainLandscapeElements

#Probe: Für raster=[raster1, raster2] und raster=[raster2, raster1] erscheint nie ein weißes Bild, sondern immer richtige Abbildung des landscape elements
#Für raster=[raster3, raster4] und raster=[raster4, raster3] erscheint bei entsprechenden landscape elements ein weißes Bild (z.B. ref 54)

plt.ion()
raster1 = '..\\Luftbilder-qgis\\Rhein_Knoblochsaue\\lower_raster_nan.tif'
raster2 = '..\\Luftbilder-qgis\\Rhein_Knoblochsaue\\upper_raster_nan.tif'
raster3 = '..\\Luftbilder-qgis\\Rhein_Knoblochsaue\\2D Basis 50m _shifted_compressed.tif'
raster4 = '..\\Luftbilder-qgis\\Rhein_Knoblochsaue\\Tag 1 3D Basis 50m_ shifted_Compressed.tif'
vector = '..\\Luftbilder-qgis\\Rhein_Knoblochsaue\\manual_mapping.shp'

rasters_knobloch = [raster1, raster2]
vectors_knobloch = [vector]

knobloch = GISProject(rasters_knobloch, vectors_knobloch)
knobloch_driftwood = FloodplainLandscapeElements(knobloch, 'driftwood')
for i in range(len(knobloch_driftwood)):
    plt.figure()
    plt.imshow(knobloch_driftwood[i][0])
    plt.title('ref %i' %knobloch_driftwood.get_info(i)['ref'])

plt.ioff()
plt.show()