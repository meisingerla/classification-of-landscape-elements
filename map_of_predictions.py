from __future__ import print_function, division
import helpers_CNN
import torch
from osgeo import ogr
import helpers_CNN
import helpers_testing
import os
import torch.nn as nn

'''Generates shape file with right and wrong predictions'''
'''Possible bugs: Segmentation fault (core dumped) or Aborted (core dumped).
Both are related to change of vector and can be fixed by correct closure of the objects (=None).'''
'''Way to fix: Run once, then probably bug Segmentation fault (core dumped). Delete the 4 generated files on the computer
and then run the file again and again without changing anything, until hopefully no more error appears.'''
'''If landscape element is None Type: vector_with_pred exists, but copy of landscape element was interrupted. Delete file and try again.'''

torch.manual_seed(42)

class_names = ['large wood', 'no large wood']

model_name = 'model-May24_22-19-21_iws-juck.pth'
model_path = '../models/' + model_name
model = helpers_testing.load_model(model_name)
if model.fc.out_features!=len(class_names):
    raise Exception('Model was trained on an other number of classes.')

dataloaders = helpers_CNN.get_dataloaders_no_test(class_names, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
landscape_element = None
layer = None
vector = None
raster_field = None
vector_with_pred = None
vector = None
driver = None
driver = ogr.GetDriverByName('ESRI Shapefile')
for x in ['val', 'train']:
    # execute forward pass, for explanation see helpers_CNN def test_model
    dataloader = dataloaders[x]
    model.eval()
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(dataloader, 0):
            print('batch', i)
            inputs, labels, item_idcs = data
            inputs = inputs.to(device)
            targets = tuple(class_names.index(label) for label in labels)
            targets = torch.as_tensor(targets)
            targets = targets.to(device)
            
            outputs = model(inputs)
            outputs_normed = nn.functional.softmax(outputs.data, dim=1)
            probs, predicted = torch.max(outputs.data, 1)
            
            correct_indices = (targets==predicted).nonzero(as_tuple=True)[0]
            incorrect_indices = (targets!=predicted).nonzero(as_tuple=True)[0]
            last_item_idx=0
            for i,item_idx in enumerate(item_idcs):
                # get considered landscape element
                landscape_element_FID = dataloader.dataset.get_info(item_idx)['landscape element object'].GetFID()
                print('landscape_element_FID', landscape_element_FID)
                print('item_idx', item_idx)
                if last_item_idx==386:
                    print()
                # erstelle neuen vector (neues shape file) oder greife auf altes zu:
                vector_name = dataloader.dataset.get_info(item_idx)['vector'].GetDescription()
                if vector_name == '../Luftbilder-qgis/Fischpass_4962_Frauenstein_alles/manual_mapping_31255.shp':
                    vector = driver.Open(vector_name,1)
                    print('number of landscape elements', vector.GetLayer(0).GetFeatureCount())
                    vector_with_pred_name = '/predictions4_' + model_name.split('.')[0] + '.shp'
                    vector_with_pred_path = '../Luftbilder-qgis/' + vector_name.split('/')[2] + vector_with_pred_name
                    if os.path.exists(vector_with_pred_path):
                        vector_with_pred = driver.Open(vector_with_pred_path, 1)
                    else:
                        vector_with_pred = driver.CopyDataSource(vector, vector_with_pred_path)
                        print('vector was copied')
                    print('number of landscape elements', vector_with_pred.GetLayer(0).GetFeatureCount())
                    # If not already present, create a new field named 'corr_pred'
                    layer = vector_with_pred.GetLayer(0) # assumption: vector has only one layer
                    landscape_element = layer.GetFeature(landscape_element_FID) #landscape element refs sind gleich -> passt
                    print('ref', landscape_element.GetField('ref'))
                    if layer.FindFieldIndex('dataloader', 0)==-1:
                        raster_field = ogr.FieldDefn('dataloader', ogr.OFTString)
                        #raster_field.SetWidth(10)
                        layer.CreateField(raster_field)
                    landscape_element.SetField('dataloader', x)
                    print('was set', x)
                    if layer.FindFieldIndex('corr_pred', 0)==-1:
                        raster_field = ogr.FieldDefn('corr_pred', ogr.OFTInteger)
                        layer.CreateField(raster_field)
                    # Set value of the field 'corr_pred' to 1 if the landscape element was predicted right, otherwise set it to 0
                    if i in correct_indices:
                        landscape_element.SetField('corr_pred', 1)
                    elif i in incorrect_indices:
                        landscape_element.SetField('corr_pred', 0)
                    else:
                        print('something is wrong')
                    # Save changes
                    last_item_idx = item_idx
                    layer.SetFeature(landscape_element)
                    vector_with_pred = None
                    vector = None
                    layer = None
                    raster_field = None
                elif vector_name == '../Luftbilder-qgis/Fischpass_4962_Frauenstein_0202/manual_mapping_5684.shp':
                    vector = driver.Open(vector_name,1)
                    print('number of landscape elements', vector.GetLayer(0).GetFeatureCount())
                    vector_with_pred_name = '/predictions4_' + model_name.split('.')[0] + '.shp'
                    vector_with_pred_path = '../Luftbilder-qgis/' + vector_name.split('/')[2] + vector_with_pred_name
                    if os.path.exists(vector_with_pred_path):
                        vector_with_pred = driver.Open(vector_with_pred_path, 1)
                    else:
                        vector_with_pred = driver.CopyDataSource(vector, vector_with_pred_path)
                        print('vector was copied')
                    print('number of landscape elements', vector_with_pred.GetLayer(0).GetFeatureCount())
                    # If not already present, create a new field named 'corr_pred'
                    layer = vector_with_pred.GetLayer(0) # assumption: vector has only one layer
                    landscape_element = layer.GetFeature(landscape_element_FID) #landscape element refs sind gleich -> passt
                    print('ref', landscape_element.GetField('ref'))
                    if layer.FindFieldIndex('dataloader', 0)==-1:
                        raster_field = ogr.FieldDefn('dataloader', ogr.OFTString)
                        #raster_field.SetWidth(10)
                        layer.CreateField(raster_field)
                    landscape_element.SetField('dataloader', x)
                    print('was set', x)
                    if layer.FindFieldIndex('corr_pred', 0)==-1:
                        raster_field = ogr.FieldDefn('corr_pred', ogr.OFTInteger)
                        layer.CreateField(raster_field)
                    # Set value of the field 'corr_pred' to 1 if the landscape element was predicted right, otherwise set it to 0
                    if i in correct_indices:
                        landscape_element.SetField('corr_pred', 1)
                    elif i in incorrect_indices:
                        landscape_element.SetField('corr_pred', 0)
                    else:
                        print('something is wrong')
                    # Save changes
                    last_item_idx = item_idx
                    layer.SetFeature(landscape_element)
                    vector_with_pred = None
                    vector = None
                    layer = None
                    raster_field = None
                else:
                    print('Landscape elements of ', vector_name, 'are not considered')
            landscape_element = None
            layer = None
            vector = None
            raster_field = None