import numpy as np
import torch
from torch.utils.data import DataLoader
import math

'''Methods to uniform data and check distribution of data'''

def manual_sampling(ids):
    '''Arranges ids that landscape elements of large wood and no large wood are more or less alternating.
    It would be sampler=train_ids_mixed, sampler=val_ids_mixed in dataloaders and len(trainloader.sampler) instead of len(trainloader.sampler.indices).
    Is nonsense for training, because batches should vary. Reproducibility solved by torch.manual_seed() instead.
    
    Args:
        ids (list of int):

    Return:
        ids_mixed (list of int):
    '''
    ids_first_half = ids[:math.ceil(len(ids)/2)]
    ids_second_half = ids[math.ceil(len(ids)/2):]
    ids_mixed = [None] * len(ids)
    for i,item in enumerate(ids_first_half):
        ids_mixed[i*2] = item
    for i,item in enumerate(ids_second_half):
        ids_mixed[i*2+1] = item
    return ids_mixed

def set_two_fixed_subsets(training_dataset, batch_size, ratio_val=0.2):
    '''The training set is divided into one training and one validation dataset.
    Every int(1/ratio_val)th landscape element of the training_dataset is assigned to the validation dataset and all other landscape elements are assigned to the training set.
    This determined (not random) assignment is reasonable for this specific training_dataset, because the first half of landscape elements are all of class "large wood" and the second half of landscape elements are all of class "no large wood".

    Args:
        training_dataset (FloodplainLandscapeElements): Dataset containing the landscape elements intended for training the network
        ratio_val (float, default 0.2): Portion of the training_dataset that is used for validation

    Returns:
        dataloaders (set of torch.utils.data.DataLoader): Set with the keys "train" and "val", each containing the assigned landscape elements grouped into batches
    '''
    dataset_indices = list(range(len(training_dataset)))
    val_ids = np.arange(0,len(training_dataset),int(1/ratio_val)).tolist()
    train_ids = [x for x in dataset_indices if x not in val_ids]
    subsampler = {'train': torch.utils.data.SubsetRandomSampler(train_ids),
                  'val': torch.utils.data.SubsetRandomSampler(val_ids)} #shuffles
    dataloaders = {x: DataLoader(training_dataset, batch_size=batch_size, sampler=subsampler[x]) for x in ['train', 'val']}
    #print('val_ids_mixed', val_ids_mixed)
    #print('train_ids_mixed', train_ids_mixed)
    #print('riprap training', get_number_landscape_elements(dataloaders['train'], 'riprap'))
    #print('riprap val', get_number_landscape_elements(dataloaders['val'], 'riprap'))
    #print('count_large_wood_train', count_large_wood_train)
    #print('count_no_large_wood_train', count_no_large_wood_train)
    #print('count_large_wood_val', count_large_wood_val)
    #print('count_no_large_wood_val', count_no_large_wood_val)
    return dataloaders

def get_number_landscape_elements_each_class(dataloaders):
    '''Computes the distribution of the landscape elements belonging to the classes "large wood" and "no large wood" in the training and validation dataset.

    Args:
        dataloaders (set of torch.utils.data.DataLoader): Set with the keys "train" and "val", each containing the assigned landscape elements grouped into batches.

    Returns:
        count_large_wood_train (int): Number of landscape elements of the type "large wood" in the training dataset
        count_no_large_wood_train (int): Number of landscape elements of the type "no large wood" in the training dataset
        count_large_wood_val (int): Number of landscape elements of the type "large wood" in the validation dataset
        count_no_large_wood_val (int): Number of landscape elements of the type "no large wood" in the validation dataset
    '''
    count_large_wood_train = 0
    count_no_large_wood_train = 0
    for batch in iter(dataloaders['train']):
        count_large_wood_train+=batch[1].count('large wood')
        count_no_large_wood_train+=batch[1].count('no large wood')
    count_large_wood_val = 0
    count_no_large_wood_val = 0
    for batch in iter(dataloaders['val']):
        count_large_wood_val+=batch[1].count('large wood')
        count_no_large_wood_val+=batch[1].count('no large wood')
    return count_large_wood_train, count_no_large_wood_train, count_large_wood_val, count_no_large_wood_val

def get_number_landscape_elements(dataloader, class_name):
    '''Returns the number of landscape elements of the entered class in the dataloader.

    Args:
        dataloader (torch.utils.data.DataLoaders): dataloader that contains the data entered in the model
        class_name (str): class of the landscape elements that is be counted
    
    Returns:
        count_landscape_element (int): number 
    '''
    count_landscape_element = 0
    for batch in iter(dataloader):
        count_landscape_element+=batch[1].count(class_name)
    return count_landscape_element

def get_location_distr(dataloader):
    '''Determines the number of landscape elements that stem from Koblochsaue and fish bypass, respectively.

    Args:
        dataloader (torch.utils.data.DataLoaders): dataloader that contains the data entered in the model

    Returns:
        None
    '''
    dataset = dataloader.dataset
    sum_knobloch=0
    sum_fish_bypass=0
    for data in dataloader:
        _, _, item_idcs = data
        for item_idx in item_idcs:
            print(dataset.get_info(item_idx)['label'])
            if 'Knobloch' in dataset.get_info(item_idx)['raster'].GetDescription():
                sum_knobloch+=1
            elif 'Fischpass' in dataset.get_info(item_idx)['raster'].GetDescription():
                sum_fish_bypass+=1
    print('Number of landscape elements from Rhein_Knoblochsaue:', sum_knobloch)
    print('Number of landscape elements from Fischpass_4962_Frauenstein', sum_fish_bypass)
    #in 'test': für split='separated' immer Rhein_Knoblochsaue, für split='mixed' sowohl Knoblochsaue als auch Fischpass