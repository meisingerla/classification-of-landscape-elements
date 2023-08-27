#from qgis.core import *
from osgeo import gdal
import helpers_preprocessing
import os
import logging
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
#import helpers_CNN_visualize

'''Classes related to pre-processing of geospatial data provided as QGIS projects:
GISProject(): Merging of related rasters and vectors (usually belonging to the same QGIS project).
FloodplainProject(): Conflation of several GISProject to create a dataset consisting of more than one QGIS project.
FloodplainLandscapeElements(): Input data for a neural network for image recognition. Landscape elements derive from GISProject or FloodplainProject and their labels belong to the class_names
'''

class GISProject():
    '''Merging of related rasters and vectors (usually belonging to the same QGIS project).

    Attributes:
        filenames_rasters (list of str): List of all relevant raster filenames (with directory, ending on ''.tif'')
        filenames_vectors (list of str): List of all relevant vector filenames (with directory, ending on ''.shp'')
    '''

    def __init__(self, filenames_rasters, filenames_vectors):
        self.rasters = []
        for filename_raster in filenames_rasters:
            if gdal.OpenEx(filename_raster).RasterCount<=0:
                raise Exception('%s is no raster' %filenames_rasters)
            self.rasters.append(gdal.OpenEx(filename_raster))
        self.vectors = []
        for filename_vector in filenames_vectors:
            vector = gdal.OpenEx(filename_vector)
            if vector.GetLayerCount()<=0:
                raise Exception('%s is no vector' %filenames_vectors)
            elif vector.GetLayerCount()>1:
                raise Exception('Vector has more than one layer. The Code is not intended for this case')
            self.vectors.append(vector)


    def __call__(self, *args, **kwargs):
        # aufzurufen mit object()
        # prints class structure information to console
        print("Class Info: <type> = GISProject (%s)" % os.path.dirname(__file__))
        print(dir(self))

    # Methoden aus helpers übernehmen macht keinen Sinn, beziehen sich (fast) alle auf ein raster und einen Vektor (kein self)
    # -> bleiben in helpers_preprocessing.
    
    def get_all_landscape_elements(self):
        '''Creates a list of all landscape elements contained in self.vectors

        Args:
            None

        Returns:
            list_of_landscape_elements (list): List of dictionaries containing information about all landscape elements in self.vectors
                Keys of dictionary: 'landscape element object', 'vector', 'raster', 'label', 'class', 'ref', 'coordinates', 'indices'
        '''
        list_of_landscape_elements = []
        for vector in self.vectors:
            vector.ResetReading()
            landscape_element = vector.GetNextFeature()[0] #liefert ein osgeo.ogr.Feature ([1] liefert zugehörige osgeo.ogr.Layer, bei mir immer gleich)
            while landscape_element is not None:
            #Alternative zu while und GetNextFeature():
            #for i in range(vector.GetLayer().GetFeatureCount()):
                #landscape_element = vector.GetLayer().GetFeature(i)
                try:
                    #idx_ref = landscape_element.GetFieldIndex('ref')
                    #ref = landscape_element.GetField(idx_ref)
                    ref = landscape_element.GetField('ref')
                except KeyError as e:
                    logging.error("Landscape element has no field named 'ref'")
                    print(e)
                try:
                    label = landscape_element.GetField('label')
                except KeyError as e:
                    logging.error("Landscape element has no field named 'label'")
                    print(e)
                landscape_element_coordinates = helpers_preprocessing.get_landscape_element_coords(landscape_element)
                raster_of_landscape_element = helpers_preprocessing.get_raster_of_landscape_element(self.rasters, landscape_element_coordinates, vector)
                landscape_element_indices = helpers_preprocessing.coords2pixels(landscape_element_coordinates, raster_of_landscape_element, vector)
                info = {'landscape element object': landscape_element, 'vector': vector, 'raster': raster_of_landscape_element, 'label': label, 'class': label, 'ref': ref, 'coordinates': landscape_element_coordinates, 'indices': landscape_element_indices}
                #nur hinzufügen, wenn wenig nan?
                list_of_landscape_elements.append(info)
                landscape_element = vector.GetNextFeature()[0]
        return list_of_landscape_elements
    
    def __eq__(self, other):
        rasters_self = set([raster.GetDescription() for raster in self.rasters])
        rasters_other = set([raster.GetDescription() for raster in other.rasters])
        vectors_self = set([vector.GetDescription() for vector in self.vectors])
        vectors_other = set([vector.GetDescription() for vector in other.vectors])
        if rasters_self==rasters_other and vectors_self==vectors_other:
            return True
        else:
            return False
            

class FloodplainProject():
    '''Conflation of several GISProject to create a dataset consisting of more than one QGIS project.

    Attributes:
        gisprojects (list of GISProject): List of all GISProject relevant for desired dataset
    '''
    def __init__(self, gisprojects):
        self.gisprojects = gisprojects

    def __call__(self, *args, **kwargs):
        # aufzurufen mit object()
        # prints class structure information to console
        print("Class Info: <type> = FloodplainProject (%s)" % os.path.dirname(__file__))
        print(dir(self))

    def get_all_landscape_elements(self):
        '''Creates a list of all landscape elements contained in the vectors of each self.gisprojects

        Args:
            None

        Returns:
            list_of_all_landscape_elements (list): List of dictionaries containing information about all landscape elements in the vectors of self.gisprojects
                Keys of dictionary: 'landscape element object', 'project', 'vector', 'raster', 'label', 'class', 'ref', 'coordinates', 'indices'
        '''
        list_of_landscape_elements = []
        for gisproject in self.gisprojects:
            list_of_landscape_elements.extend(gisproject.get_all_landscape_elements())
        return list_of_landscape_elements
    
    def __eq__(self, other):
        if len(self.gisprojects)==len(other.gisprojects):
            count_max=len(self.gisprojects)
            count=0
            for gisproject_self in self.gisprojects:
                for gisproject_other in other.gisprojects:
                    if gisproject_self==gisproject_other:
                        count+=1 #funktioniert nicht, wenn gleiches gisproject mehrmals in FloodplainProject. Muss an anderer Stelle unterbunden werden.
            if count==count_max:
                return True
            else:
                return False
        else:
            return False


class FloodplainLandscapeElements(Dataset):
    '''Input data for a neural network for image recognition.
    Landscape elements derive from GISProject or FloodplainProject and are labeled.

    Attributes: (constructed in init-method)
        geospatial_project (GISProject or FloodplainProject): Provides geospatial information about the data
        class_names (list of str): Classes that the neural network should classify
        negated_class_names (list of str): If a class name starts with 'no ', create a negated (and thus existing) class name without the 'no '. Otherwise enter 'None'.
        image_shape (tuple, default (224, 224)): Shape of the image tile
        transform (torchvision.transforms, default None): Transformation of the data (e.g. crop, normalize)
        length (list of int): Number of landscape elements of the type listed accordingly in class_names, which should be contained in the FloodplainLandscapeElements instance.
        list_all_described_landscape_elements (list): List of dictionaries containing information about all landscape elements of the types self.decriptios.
            If the class name starts with 'no ' all landscape elements which are not of the type following after 'no ' are addressed.
            Keys of dictionary: 'landscape element object', 'vector', 'raster', 'label', 'class', 'ref', 'coordinates', 'indices'
    '''

    def __init__(self, geospatial_project, class_names, image_shape=(224,224), transform=None, length=None):
        '''Constructor.

        Args:
            geospatial_project (GISProject or FloodplainProject): Provides geospatial information about the data
            class_names (str or list of str): Classes of the data. Only one class_name not useful for classification, but for analyzing the dataset.
            image_shape (tuple, default (224, 224)): Shape of the image tile
            transform (torchvision.transforms.transforms.Compose, default None): Transformation of the data (e.g. crop, normalize)
            length (int or list of int): Number of landscape elements of the type listed accordingly in class_names, which should be contained in the FloodplainLandscapeElements instance.
        
        Returns:
            Attributes mentioned in the class description
        '''
        self.geospatial_project = geospatial_project
        # if class_names is just one string, put it in a list
        if isinstance(class_names, str):
            self.class_names = [class_names]
        elif isinstance(class_names, list):
            self.class_names = class_names
        else:
            raise Exception('class_names has wrong type, should be ''str'' or ''list''')
        # if a class_name starts with 'no ', create a negated class_name without the 'no '
        self.negated_class_names = []
        for class_name in self.class_names:
            if class_name[0:3]=='no ':
                self.negated_class_names.append(class_name[3:])
            else:
                self.negated_class_names.append(None)
        self.image_shape = image_shape
        self.transform = transform
        # if length is just one tuple or None, put it in a list
        if isinstance(length, int):
            if len(self.class_names)!=1:
                raise Exception('Length of some class_names are missing. If you want them to have the maximum possible length, enter None. For example: [None, 100, None] for three class_names')
            self.length = [length]
        elif isinstance(length, list):
            self.length = length
        elif length==None:
            self.length = [None]*len(self.class_names)
        else:
            raise Exception('length has wrong type, should be ''int'' or ''list''.')
        # check if as many landscape elements exist as requested in length
        if self.length!=[None]*len(self.class_names):
            max_number_described_landscape_element = []
            for i, number in enumerate(self.length):
                if self.class_names[i]!='everything else':
                    max_number_described_landscape_element.append(len(self.get_described_landscape_elements(self.class_names[i])))
                    if number:
                        if number>max_number_described_landscape_element[i]:
                            raise Exception('So many landscape elements of the described type do not exist. Maximum length is %d' %len(self.get_described_landscape_elements(self.class_names[i])))
            if 'everything else' in self.class_names:
                    i = self.class_names.index('everything else')
                    if self.length[i]:
                        number_all_landscape_elements = len(self.geospatial_project.get_all_landscape_elements())
                        if self.length[i]>number_all_landscape_elements-sum(max_number_described_landscape_element):
                            raise Exception('So many landscape elements of type "everything else" do not exist. Maximum length is %d' %(number_all_landscape_elements-sum(max_number_described_landscape_element)))
        self.list_all_described_landscape_elements = self.get_all_described_landscape_elements()

    def __call__(self, *args, **kwargs):
        # aufzurufen mit object()
        # prints class structure information to console
        print("Class Info: <type> = FloodplainLandscapeElements (%s)" % os.path.dirname(__file__))
        print(dir(self))

    def get_all_described_landscape_elements(self):
        '''Creates a list of all landscape elements of the types listed in self.class_names.

        Args:
            None

        Returns:
            all_described_landscape_elements (list): List of dictionaries containing information about all landscape elements of the types self.decriptios.
                If a class name starts with 'no ' all landscape elements which are not of the type following after 'no ' are addressed.
                Keys of dictionary: 'landscape element object', 'vector', 'raster', 'label', 'class', 'ref', 'coordinates', 'indices'
        '''
        all_described_landscape_elements = []
        for class_name in self.class_names:
            if class_name!='everything else':
                described_landscape_elements = self.get_described_landscape_elements(class_name)
                all_described_landscape_elements.extend(described_landscape_elements)
        # folgendes könnte auch in for-Schleife, aber dann wäre Position von 'everything else' entscheidend
        if 'everything else' in self.class_names:
            described_landscape_elements_ee = self.get_everything_else_landscape_elements(all_described_landscape_elements)
            all_described_landscape_elements.extend(described_landscape_elements_ee)
        return all_described_landscape_elements
    
    def get_everything_else_landscape_elements(self, all_described_landscape_elements):
        '''Creates a list of all landscape elements of type 'everything else'

        Args:
            all_described_landscape_elements (list): List of the landscape elements (dict of properties) of all types listed in class_names (except of type 'everything else')

        Returns:
            described_landscape_elements (list): List of dictionaries containing information about all landscape elements of type 'everything else'.
                Keys of dictionary: 'landscape element object', 'vector', 'raster', 'label', 'class', 'ref', 'coordinates', 'indices'
        '''
        described_landscape_elements = []
        idx = self.class_names.index('everything else')
        all_landscape_elements = self.geospatial_project.get_all_landscape_elements()
        coord_of_all_described_landscape_elements = [described_landscape_element['coordinates'] for described_landscape_element in all_described_landscape_elements]
        for landscape_element in all_landscape_elements:
            if landscape_element['coordinates'] not in coord_of_all_described_landscape_elements:
                # Prüfen ob landscape element selbst in described_landscape_elements liegt funktioniert nicht, da landscape_element['landscape element object'] nicht konstant.
                # Stattdessen auf coord prüfen, ist am genauesten. Noch zusätzlichen Test einbauen?
                landscape_element['class'] = 'everything else'
                described_landscape_elements.append(landscape_element)
        if self.length[idx]:
            described_landscape_elements = described_landscape_elements[:self.length[idx]]
        return described_landscape_elements
    
    def get_described_landscape_elements(self, class_name):
        '''Creates a list of all landscape elements beloning to the entered class.

        Args:
            class_name (str): class to which the label of the data belongs

        Returns:
            described_landscape_elements (list): List of dictionaries containing information about all landscape elements of type decriptio.
                If class_name starts with 'no ' all landscape elements which are not of the type following after 'no ' are addressed.
                Keys of dictionary: 'landscape element object', 'vector', 'raster', 'label', 'class', 'ref', 'coordinates', 'indices'
        '''
        landscape_elements = self.geospatial_project.get_all_landscape_elements()
        idx = self.class_names.index(class_name)
        described_landscape_elements = []
        for landscape_element in landscape_elements:
            if self.negated_class_names[idx]:
                if landscape_element['class']!=self.negated_class_names[idx]:
                    landscape_element['class'] = class_name
                    described_landscape_elements.append(landscape_element)
            else:
                if landscape_element['class']==class_name:
                    described_landscape_elements.append(landscape_element)
        if self.length[idx]:
            described_landscape_elements = described_landscape_elements[:self.length[idx]]
        return described_landscape_elements

    def __len__(self):
        '''Returns the number of landscape elements belonging to one of the classes in self.class_names

        Args:
            None

        Returns:
            number_described_landscape_elements (int): number of landscape elements of type self.class_names
        '''
        number_described_landscape_elements = len(self.list_all_described_landscape_elements)
        return number_described_landscape_elements

    def __iter__(self):
        '''Generates an Iterator.
        
        Args:
            None
            
        Returns:
            None
        '''
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        '''Provides an image and the class of one landscape element. Call with self[idx].

        Args:
            idx (int): Index to the list of all landscape elements belonging to the FloodplainLandscapeElements instance

        Returns:
            image_tile (np.array): Pixel np.array including idx-th landscape element of FloodplainLandscapeElements instance
            class_name (str): class of the regarding landscape element
            idx (int): Index of the regarding landscape element in the list of all landscape elements belonging to the FloodplainLandscapeElements instance
        '''
        all_described_landscape_elements = self.list_all_described_landscape_elements #besser einmal speichern als zweimal aufrufen?
        if idx>=len(all_described_landscape_elements):
            raise Exception('idx is out of range')
        landscape_element = all_described_landscape_elements[idx]
        image_tile = helpers_preprocessing.cutout_around_landscape_element(landscape_element['raster'], landscape_element['indices'], self.image_shape)
        image_tile = image_tile.astype('uint8')
        if self.transform:
            #plt.figure()
            #plt.imshow(image_tile)
            #plt.show()
            image_tile = self.transform(image_tile)
            #plt.figure()
            #plt.imshow(helpers_CNN_visualize.imshow(image_tile))
            #plt.show()
        return image_tile, landscape_element['class'], idx
    
    def get_info(self,idx):
        '''Provides information of one landscape element.

        Args:
            idx (int): Index to the list of all landscape elements belonging to the FloodplainLandscapeElements instance
        
        Returns:
            landscape_element (dict): Contains all relevant information about the regarding landscape element.
                Keys: 'landscape element object', 'vector', 'raster', 'label', 'class', 'ref', 'coordinates', 'indices'
        '''
        if idx>=len(self.list_all_described_landscape_elements):
            raise Exception('idx is out of range')
        landscape_element = self.list_all_described_landscape_elements[idx]
        return landscape_element
    
    def __add__(self, other):
        '''Combines two FloodplainLandscapeElements objects of the same geospatial_project, image_shape and transformation, 
        but with different class names and desired corresponding length.
        Callable via self + other.

        Args:
            other (FloodplainLandscapeElements): FloodplainLandscapeElements object of the same geospatial_project, image_shape and transformation as shape, 
                but with different class names and desired corresponding length.
        
        Returns:
            combined FloodplainLandscapeElements object
        '''
        if self.geospatial_project!=other.geospatial_project:
            raise Exception('Arguments have different geospatial projects.')
        combined_class_names = self.class_names+other.class_names
        if self.image_shape!=other.image_shape:
            raise Exception('Arguments have different image shapes.')
        image_shape = self.image_shape
        if self.transform!=other.transform:
            raise Exception('Arguments have different transforms.Compose objects. If both have the same arguments, please take the same object')
            #only checks for the same object (id), not for the same content
        transform = self.transform
        combined_length = self.length+other.length
        return FloodplainLandscapeElements(self.geospatial_project, combined_class_names, image_shape=image_shape, transform=transform, length=combined_length)
    
    def get_mean_and_std(self):
        '''Computes the mean and the standard deviation of the image dataset of the class instance before performing the transformation.
        Return values can be used for Normalization:
        Init instance of FloodplainLandscapeElements with transform=transfomrs.ToTensor(), execute get_mean_and_std(self) on it and change the class attribute transform of the instance to a transformation with the corresponding normalization.

        Args:
            None

        Returns:
            total_mean (np.array of shape (1,3)): mean of each rgb channel of the whole image dataset
            total_std (np.array of shape (1,3)): standard deviation of each rgb channel of the whole image dataset
        '''
        images = [helpers_preprocessing.cutout_around_landscape_element(landscape_element['raster'], landscape_element['indices'], self.image_shape) for landscape_element in self.list_all_described_landscape_elements]
        rgb_sum = np.array([0.0, 0.0, 0.0])
        rgb_sum_sq = np.array([0.0, 0.0, 0.0])
        for i,image in enumerate(images):
            if len(np.argwhere(np.isnan(image)))<=self.image_shape[0]*self.image_shape[1]*3/3:
                image = np.where(np.isnan(image),0,image)
            else:
                print(self.list_all_described_landscape_elements[i])
                raise Exception('More than one third of the image is nan. Check whether the pixels should set as white pixels.')
            rgb_sum += image.sum(axis=(0,1))/255
            rgb_sum_sq += ((image/255)**2).sum(axis=(0,1))
        count = len(images) * self.image_shape[0] * self.image_shape[1]
        total_mean = rgb_sum / count
        total_var  = (rgb_sum_sq / count) - (total_mean ** 2)
        total_std  = np.sqrt(total_var)
        return total_mean, total_std