from __future__ import print_function, division
import helpers_testing
import helpers_CNN
import helpers_CNN_uniform
import helpers_CNN_visualize
import paths
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
#import tikzplotlib
from classes_data_preprocessing import GISProject
from classes_data_preprocessing import FloodplainLandscapeElements
from torchvision import transforms
import numpy as np

'''Methods to investigate the roughness of the landscape environment'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataset_knoblochsaue(class_names, balanced=False):
    '''Merges the rasters and vectors from Rhein_Knoblochsaue to one normalized FloodplainLandscapeElements.
    For class_names=['large wood', 'no large wood'], the number of no large wood landscape elements can be limited to equal the number of large wood landscape elements by setting balanced=True.

    Args:
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network.
        balanced (bool, default=False): If true, the number of landscape elements of type 'no large wood' matches the number of landscape elements of type 'large wood'. Otherwise, all data is used.
            Balancing is only possible for class names=["large wood", "no large wood"] or class names=["large wood", "everything else"].

    Returns:
        dataset (FloodplainLandscapeElements): Dataset containing the landscape elements from the fish_bypass_complete and fish_bypass_0202 project
    '''
    rasters_knobloch = [paths.raster2_knobloch, paths.raster1_knobloch]
    vectors_knobloch = [paths.vector2_knobloch, paths.vector3_knobloch]
    project = GISProject(rasters_knobloch, vectors_knobloch)
    
    if balanced:
        if set(class_names)=={'large wood', 'no large wood'} or set(class_names)=={'large wood', 'everything else'}:
            test_images_large_wood = FloodplainLandscapeElements(project, class_names[0], transform=transforms.ToTensor())
            length_large_wood=len(test_images_large_wood)
            dataset = FloodplainLandscapeElements(project, class_names, image_shape=(224,224), transform=transforms.ToTensor(), length=[None, length_large_wood])
        else:
            raise Exception('Balancing is only possible for class names=["large wood", "no large wood"] or class names=["large wood", "everything else"].')
    else:
        dataset = FloodplainLandscapeElements(project, class_names, image_shape=(224,224), transform=transforms.ToTensor())
    mean, std = dataset.get_mean_and_std()
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset.transform=data_transforms
    return dataset

def get_dataset_fish_bypass(class_names):
    '''Merges the rasters and vectors from Fischpass_Frauenstein_alles to one normalized FloodplainLandscapeElements.

    Args:
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network.

    Returns:
        dataset (FloodplainLandscapeElements): Dataset containing the landscape elements from the fish_bypass_complete and fish_bypass_0202 project
    '''
    rasters_fish_bypass = [paths.raster1_fish_bypass]
    vectors_fish_bypass = [paths.vector1_fish_bypass]
    project = GISProject(rasters_fish_bypass, vectors_fish_bypass)
    
    dataset = FloodplainLandscapeElements(project, class_names, image_shape=(224,224), transform=transforms.ToTensor())
    mean, std = dataset.get_mean_and_std()
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset.transform=data_transforms
    return dataset

def get_smallest_std(current_class_name, item_idx, std, smallest_std, desired_class_name=None):
    '''Checks whether the standard deviation of the considered data point is smaller than the smallest standard deviation measured for the previous data points.

    Args:
        current_class_name (str): class name of the considered data point
        item_idx (int): item idx of the considered data point
        std (torch.Tensor): standard deviation of the considered data point
        smallest_std (dict<torch.Tensor>): item_idx (callable with key 'index') and standard deviation (callable with key 'value') of a previous data points with the smallest std
        desired_class_names (str, default=None): class name for which the standard deviation should be examined. If None, data points of all class_names are considered.

    Returns:
        smallest_std (dict<torch.Tensor>): item_idx and std of the data point with the smallest std, including the current data points in the measurement
    '''
    if desired_class_name:
        if current_class_name==desired_class_name:
            if sum(std)<sum(smallest_std['value']):
                smallest_std['index'] = item_idx
                smallest_std['value'] = std
    else:
        if sum(std)<sum(smallest_std['value']):
            smallest_std['index'] = item_idx
            smallest_std['value'] = std
    return smallest_std

def get_largest_std(current_class_name, item_idx, std, largest_std, desired_class_name=None):
    '''Checks whether the standard deviation of the considered data point is larger than the largest standard deviation measured for the previous data points.

    Args:
        current_class_name (str): class name of the considered data point
        item_idx (int): item idx of the considered data point
        std (torch.Tensor): standard deviation of the considered data point
        largest_std (dict<torch.Tensor>): item_idx (callable with key 'index') and standard deviation (callable with key 'value') of a previous data point with the largest std
        desired_class_names (str, default=None): class name for which the standard deviation should be examined. If None, data points of all class_names are considered.

    Returns:
        largest_std (dict<torch.Tensor>): item_idx and std of the data point with the largest std, including the current data point in the measurement
    '''
    if desired_class_name:
        if current_class_name==desired_class_name:
            if sum(std)>sum(largest_std['value']):
                largest_std['index'] = item_idx
                largest_std['value'] = std
    else:
        if sum(std)>sum(largest_std['value']):
            largest_std['index'] = item_idx
            largest_std['value'] = std
    return largest_std

def get_std_each_image(class_names, dataset, count, smallest_std, largest_std, writer):
    '''Calculates the standard deviation for each image in dataset.

    Args:
        class_names (list<str>): classes of landscape elements that should be compared
        dataset (FloodplainLandscapeElements): considered and prepared landscape elements
        count (dict<int>): Contains the number of landscape elements per class
        smallest_std (dict<torch.Tensor>): item_idx (callable with key 'index') and standard deviation (callable with key 'value') of the data point with the smallest std
        largest_std (dict<torch.Tensor>): item_idx (callable with key 'index') and standard deviation (callable with key 'value') of the data point with the largest std
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard

    Returns:
        std_per_data_point (dict<torch.Tensor>): standard deviation for each image assigned to the class names
    '''
    std_per_data_point = {x: torch.zeros(count[x],3).to(device) for x in class_names}
    k={x: 0 for x in class_names}
    # Iterate over the data and calculate std per image
    for data in dataset:
        # Get images with belonging class
        image, class_name, item_idx = data #image ist normiert, daher nicht in 0 bis 1
        image = image.to(device)

        pixels = 224*224
        var = ((image**2).sum(axis=[1,2]) / pixels) - ((image.sum(axis=[1,2])/pixels) ** 2)
        std = torch.sqrt(var)
        std_per_data_point[class_name][k[class_name]] = std
        k[class_name]+=1
        # find large wood image with smallest/largest std:
        #smallest_std = get_smallest_std(class_name, item_idx, std, smallest_std, 'large wood')
        #largest_std = get_largest_std(class_name, item_idx, std, largest_std, 'large wood')
        smallest_std = get_smallest_std(class_name, item_idx, std, smallest_std)
        largest_std = get_largest_std(class_name, item_idx, std, largest_std)
    print('smallest_std:', smallest_std, ',', 'ref =', dataset.get_info(smallest_std['index'])['ref'], ', class =', dataset.get_info(smallest_std['index'])['class'])
    writer.add_image('Image with smallest std', helpers_CNN_visualize.imshow_own_norm(dataset[smallest_std['index']][0], dataset), dataformats='HWC')
    print('largest_std:', largest_std, ',', 'ref =', dataset.get_info(largest_std['index'])['ref'], ', class =', dataset.get_info(largest_std['index'])['class'])
    writer.add_image('Image with largest std', helpers_CNN_visualize.imshow_own_norm(dataset[largest_std['index']][0], dataset), dataformats='HWC')
    return std_per_data_point

def plot_std_per_data_point(std_per_data_point, dataset, name_dataset, writer):
    '''Plots the collected standard deviation per image.
    Large wood data points are marked red, no large wood data points are marked blue.
    Works only for class_names=['large wood', 'no large wood'].

    Args:
        std_per_data_point (dict<torch.Tensor>): standard deviation for each image assigned to the class names
        dataset (FloodplainLandscapeElements): considered and prepared landscape elements
        name_dataset (str): Name of the dataset for the records
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard
    
    Returns:
        None
    '''
    # Record coordinates that should be plotted.
    # x records the number of the image, y its standard deviation.
    # Three consecutive values refer to the same image to represent the r,g and b values.
    class_names = std_per_data_point.keys()
    if set(class_names)!={'large wood', 'no large wood'}:
        raise Exception('Code is only suitable for classes "large wood" and "no large wood".')
    x = {class_name: [] for class_name in class_names}
    y = {class_name: [] for class_name in class_names}
    k=0
    for class_name in class_names:
        for std in std_per_data_point[class_name]:
            for std_per_color in std:
                x[class_name].append(k)
                y[class_name].append(std_per_color)
            k+=1
    print('large wood: mean=', np.mean(y['large wood']), 'std=', np.std(y['large wood']))
    print('no large wood: mean=', np.mean(y['no large wood']), 'std=', np.std(y['no large wood']))
    # Print distribution of standard deviation per image
    fig=plt.figure()
    plt.plot(x['large wood'], y['large wood'], 'r*', markersize=1)
    plt.plot(x['no large wood'], y['no large wood'], 'b*', markersize=1)
    plt.xlim(0,len(dataset))
    plt.ylim(0,1.35)
    plt.xlabel('image')
    plt.ylabel('standard deviation')
    plt.title(name_dataset)
    #plt.legend(['large wood', 'no large wood'])
    #plt.show()
    #tikzplotlib.save("roughness_knobloch_rasters_reordered.tex")
    writer.add_figure('distribution of standard deviation per image', fig, 0)

def get_average_std(std_per_data_point, count, class_names):
    '''Calculates the average std for each class_name and over all classes.

    Args:
        std_per_data point (dict<torch.Tensor>): standard deviation for each image assigned to the class names
        count (dict<int>): Contains the number of landscape elements per class
        class_names (list<str>): classes of landscape elements that should be compared

    Returns:
        None
    '''
    average_std = {x: torch.Tensor([0.0, 0.0, 0.0]).to(device) for x in class_names}
    average_std_overall = 0
    for key in class_names:
        average_std[key] = std_per_data_point[key].sum(axis=0) / count[key]
        print(key, ': average std (average of std per image)=', average_std[key])
        average_std_overall += average_std[key]*count[key]
    average_std_overall = average_std_overall/(sum(count.values()))
    print('average std overall =', average_std_overall)

def initialize_lw_data_point(dataset):
    '''Calls the first large wood element to initialize the smallest and largest std.

    Args:
        dataset (FloodplainLandscapeElements): considered and prepared landscape elements

    Returns:
        smallest_std (dict<torch.Tensor>): item_idx (callable with key 'index') and standard deviation (callable with key 'value') of the data point with the smallest std
        largest_std (dict<torch.Tensor>): item_idx (callable with key 'index') and standard deviation (callable with key 'value') of the data point with the largest std
    '''
    # Initialize first values to find images with smallest/largest mean and std
    image, class_name, item_idx = next(iter(dataset))
    if class_name!='large wood':
        raise Exception('Should be large wood')
    pixels=224*224
    var = ((image**2).sum(axis=[1,2]) / pixels) - ((image.sum(axis=[1,2])/pixels) ** 2)
    std = torch.sqrt(var)
    smallest_std_lw = {'index': item_idx, 'value': std}
    largest_std_lw = {'index': item_idx, 'value': std}
    return smallest_std_lw, largest_std_lw

def evaluate_roughness_lw(name_dataset):
    '''Plots the standard deviation for each image of the dataset and calculates the average std for large wood and no large wood images.

    Args:
        name_dataset (str): Name of the dataset that should be considered. Either 'knoblochsaue' or 'fish bypass'.

    Returns:
        None
    '''
    print(name_dataset)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    # Initialize class_names, dataset and values necessary for calculation of std
    class_names = ['large wood', 'no large wood']
    if name_dataset=='knoblochsaue':
        dataset = get_dataset_knoblochsaue(class_names, balanced=False)
    elif name_dataset=='fish bypass':
        dataset = get_dataset_fish_bypass(class_names)
    else:
        raise Exception('"name dataset" has to be either "knoblochsaue" or "fish bypass".')
    dataloader = DataLoader(dataset, shuffle=False)
    count = {x: helpers_CNN_uniform.get_number_landscape_elements(dataloader, x) for x in class_names}
    smallest_std, largest_std = initialize_lw_data_point(dataset)
    # Calculate plot std per image
    std_per_data_point = get_std_each_image(class_names, dataset, count, smallest_std, largest_std, writer)
    plot_std_per_data_point(std_per_data_point, dataset, name_dataset, writer)
    get_average_std(std_per_data_point, count, class_names)