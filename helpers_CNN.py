import numpy as np
from datetime import datetime
import torch
from sklearn.model_selection import KFold
import time
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import googlenet, GoogLeNet_Weights
from classes_data_preprocessing import GISProject
from classes_data_preprocessing import FloodplainProject
from classes_data_preprocessing import FloodplainLandscapeElements
import paths
import helpers_testing
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

'''Methods for hyperparametertuning, training and testing.'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show_in_tensorboard(inp, current_batch_idx, writer):
    inp = inp.cpu().numpy().transpose((2, 3, 0, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    now = datetime.now()
    tag = now.strftime("%d/%m/%Y %H:%M:%S")
    #print(refs) #sind nicht in der Reihenfolge wie die Bilder erscheinen
    writer.add_images('batch ' + str(current_batch_idx), inp, 0, dataformats='HWNC')

def names_to_numbers(class_names, targets_str):
    '''Transformation of the string targets to numbers.
    The numbers are indices of the list class_names, so a reverse transformation is easily possible.

    Args:
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network
        targets_str (tuple): List of class of each occuring landscape element
    
    Returns:
        targets_int (torch.Tensor): Tensor of numbers that indices the class of each occuring landscape element
    '''
    targets_int = tuple(class_names.index(target_str) for target_str in targets_str)
    targets_int = torch.as_tensor(targets_int)
    return targets_int

def number_to_name(class_names, target_int):
    return class_names[target_int]


def train_model(trainloader, class_names, learning_params, epoch, writer, fold=None):
    '''One training epoch of the model based on the data in trainloader.

    Args:
        trainloader (torch.utils.data.DataLoader): Contains batches of the landscape elements which are used to train the model
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network
            So far only suitable for two entries
        learning_params (FixedFeatureExtractor): Parameters (model, loss_function, optimizer, scheduler) of the CNN GoogLeNet set to execute transfer learning
        epoch (int): Current epoch of the training process
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard (transmits loss and accuracy)
        fold (int, default=None): Current fold of the k_fold_cross_validation, None for hold_out method
    
    Returns:
        learning_params.model (torchvision.models): Model after the training, i.e. with updated parameters
    '''
    # Set the mode to train (e.g. dropout is activated)
    learning_params.model.train()
    # Set current loss value of this epoch
    current_loss = 0.0
    correct_count = 0
    # Iterate over the DataLoader for training data (each loop handles one batch)
    for i, data in enumerate(trainloader):
        # Get inputs (images), targets (class_names) and item_idcs
        inputs, targets_str, _ = data
        inputs = inputs.to(device)
        # count_large_wood = targets_str.count('large wood')
        # print('count_large_wood train', count_large_wood)
        # count_no_large_wood = targets_str.count('no large_ wood')
        # print('count_no_large_wood train', count_no_large_wood)
        # Show all images of each batch
        #show_in_tensorboard(inputs, i, writer)
        #print('showed batch')
        # Convert targets_str (tensor of str) to targets (tensor of int)
        targets = names_to_numbers(class_names, targets_str)
        targets = targets.to(device)
        # Zero the gradients (actually set is to None). Otherwise the current gradient is accumulated (summed) to the gradient from the last backward passes.
        learning_params.optimizer.zero_grad()
        # Compute gradients. Wouldn't these be computed anyway?
        with torch.set_grad_enabled(True):
            # Perform forward pass
            if learning_params.use_auxiliary:
                outputs, aux1, aux2 = learning_params.model(inputs) #Rehienfolge der returns von hinten nach vorne durchs Netz?
            else:
                outputs = learning_params.model(inputs)
            outputs_normed = nn.functional.softmax(outputs.data, dim=1)
            _, predictions = torch.max(outputs_normed, 1) #wählt max aus jeder Zeile (diese class ist wahrscheinlicher) und zugehöriger index entspricht der predicted class
            # Compute loss
            if learning_params.use_auxiliary:
                loss = learning_params.loss_function(outputs, targets) + 0.3 * learning_params.loss_function(aux1, targets) + 0.3 * learning_params.loss_function(aux2, targets)
            else:
                loss = learning_params.loss_function(outputs, targets) #type(tensor), 0-dim, contains one value which is callable by .item()
            # Perform backward pass
            loss.backward()
            # Perform optimization -> update weights
            learning_params.optimizer.step()
        # Loss statistics
        current_loss += loss.item() * inputs.size(0) #summing up deaveraged loss from each batch (deaveraging by inputs.size(0) (number of landscape elements in batch) to get exact value for the last (not full) batch)
        # Accuracy statistics
        correct_indices = (targets==predictions).nonzero(as_tuple=True)[0]
        correct_count += len(correct_indices)
    #change the learning rate
    learning_params.scheduler.step()
    #print(learning_params.optimizer.param_groups[0]['lr'])
    # Print loss of this epoch
    try:
        length_trainloader = len(trainloader.sampler.indices)
    except AttributeError as e:
        length_trainloader = len(trainloader.sampler)
    train_loss = current_loss / length_trainloader
    if fold is not None:
        writer.add_scalar("Loss/train/fold"+str(fold), train_loss, epoch)
    else:
        writer.add_scalar("Loss/train", train_loss, epoch)
    # Print accuracy of this epoch
    train_acc = correct_count / length_trainloader
    if fold is not None:
        writer.add_scalar("Accuracy/train/fold"+str(fold), train_acc, epoch)
    else:
        writer.add_scalar("Accuracy/train", train_acc, epoch)
    return learning_params.model


def test_model(testloader, class_names, learning_params, epoch, writer, phase, acc_per_fold=None):
    '''Validation of the model developed in the current epoch based on the data in testloader.

    Args:
        testloader (torch.utils.data.DataLoader): Contains batches of the landscape elements which are used to validate the model
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network
        learning_params (FixedFeatureExtractor): Parameters (model, loss_function, optimizer, scheduler) of the CNN GoogLeNet set to execute transfer learning
        epoch (int): Current epoch of the training process
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard (transmits loss and accuracy)
        phase (str): Defines for the writer if the test is applied on 'validation' or 'test1', 'test2', ... data
        acc_per_fold (list of ints, default=None): Accuracy of the final model of each fold of cross validation, None for hold_out method

    Returns:
        None
    '''
    if acc_per_fold is not None:
        if None in acc_per_fold:
            current_fold=acc_per_fold.index(None)-1
        else:
            current_fold=len(acc_per_fold)-1
                
    # Evaluation for this fold
    correct_count = 0
    current_loss = 0.0
    # Set the mode to evaluation (e.g. dropout is deactivated)
    learning_params.model.eval()
    # Switch off autograd, so no gradients are computed. Saves memory
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, targets_str, item_idcs = data #data[2] enthält nur Werte, die auch in val_ids vorkommen -> passt
            # count_large_wood = targets_str.count('large wood')
            # print('count_large_wood val', count_large_wood)
            # count_no_large_wood = targets_str.count('no large wood')
            # print('count_no_large_wood val', count_no_large_wood)
            inputs = inputs.to(device)
            targets = names_to_numbers(class_names, targets_str)
            targets = targets.to(device)

            # zero the parameter gradients
            learning_params.optimizer.zero_grad()

            #with torch.set_grad_enabled(False): #würde outputs, probs und loss enthalten. Not necessary within torch.no_grad()?

            # Generate outputs
            outputs = learning_params.model(inputs) #durch model.eval() kommt nur noch ein return Wert, auch bei use_auxiliary=True
            outputs_normed = nn.functional.softmax(outputs.data, dim=1)

            # Set total and correct
            probs, predictions = torch.max(outputs_normed, 1) #outputs und outputs.data ist eig gleich, siehe Erklärung oben
            loss = learning_params.loss_function(outputs, targets) #so laut https://www.kaggle.com/code/ivankunyankin/googlenet-inception-from-scratch-using-pytorch/notebook. Nicht vergleichbar mit train loss??

            # Loss statistics
            current_loss += loss.item() * inputs.size(0) #summing up deaveraged loss from each batch (deaveraging by inputs.size(0) (number of landscape elements in batch) to get exact value for the last (not full) batch)
            # Accuracy statistics
            correct_indices = (targets==predictions).nonzero(as_tuple=True)[0]
            correct_count += len(correct_indices) #correct += (predictions == targets).sum().item()

        # Print loss of this epoch
        try:
            length_testloader = len(testloader.sampler.indices)
        except AttributeError as e:
            length_testloader = len(testloader.sampler)
        val_loss = current_loss / length_testloader
        if acc_per_fold is not None:
            writer.add_scalar("Loss/validation/fold"+str(current_fold), val_loss, epoch)
        else:
            writer.add_scalar("Loss/" + phase , val_loss, epoch)
        # Print accuracy of this epoch
        val_acc = correct_count / length_testloader
        if acc_per_fold is not None:
            writer.add_scalar("Accuracy/validation/fold"+str(current_fold), val_acc, epoch)
            acc_per_fold[current_fold]=val_acc
        else:
            writer.add_scalar("Accuracy/" + phase, val_acc, epoch)

def hold_out(dataloaders, class_names, learning_params, num_epochs, writer):
    '''Transfer learning of a CNN using a fixed division in training and validation dataset.

    Args:
        dataloaders (dict of torch.utils.data.DataLoaders): Includes Dataloaders for 'train', 'val' and 'test' phase
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network
        learning_params (FixedFeatureExtractor): Parameters (model, loss_function, optimizer, scheduler) of the CNN GoogLeNet set to execute transfer learning
        num_epochs (int): Number of epochs that will be run through during training
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard
    
    Returns:
        None
    '''
    #mit dataloaders[''].dataset lässt sich FloodplainLandscapeElements aufrufen, dabei keine Unterteilung in train, val und test.
    #gut für Abruf von info: dataloaders[''].dataset.get_info(i). schlecht um Länge des testsets zu finden
    print('Starting training')
    start_training = time.time()
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        start_epoch = time.time()
        #sth_vorher = copy.deepcopy(learning_params.model.inception4e.branch1.conv.weight)
        #fc2_vorher = copy.deepcopy(learning_params.model.aux2.fc2.weight)
        #fc1_vorher = copy.deepcopy(learning_params.model.aux2.fc1.weight)
        #conv_conv_vorher = copy.deepcopy(learning_params.model.aux2.conv.conv.weight)
        #fc_vorher = copy.deepcopy(learning_params.model.inception4e.branch1.conv.weight)
        torch.manual_seed(42)
        model = train_model(dataloaders['train'], class_names, learning_params, epoch, writer) #nur learning_params.model.fc.weight ändern sich
        #print('Gewichte von fc2 wurden nicht verändert:', torch.all(fc2_vorher==learning_params.model.aux2.fc2.weight)) #False
        #print('Gewichte von fc1 wurden nicht verändert:', torch.all(fc1_vorher==learning_params.model.aux2.fc1.weight)) #False
        #print('Gewichte von conv.conv wurden nicht verändert:', torch.all(conv_conv_vorher==learning_params.model.aux2.conv.conv.weight)) #False
        #print('Gewichte von fc wurden nicht verändert:', torch.all(fc_vorher==learning_params.model.fc.weight)) #False
        #print('Gewichte von sth wurden nicht verändert:', torch.all(sth_vorher==learning_params.model.inception4e.branch1.conv.weight)) #True
        print('Starting validation')
        torch.manual_seed(42)
        test_model(dataloaders['val'], class_names, learning_params, epoch, writer, 'validation')
        if 'test' in dataloaders.keys():
            test_model(dataloaders['test'], class_names, learning_params, epoch, writer, 'test')
        #print('Weights nach dem Testen:', learning_params.model.fc.weight)
        time_val = time.time() - start_epoch
        print(f'Epoch has finished in {time_val // 60:.0f}m {time_val % 60:.0f}s')
    time_training = time.time() - start_training
    print(f'Training process has finished in {time_training // 3600:.0f}h {(time_training % 3600) // 60:.0f}m {time_training % 60:.0f}s')
    # Evaluation of the trained model
    print('Evaluate the trained model')
    torch.manual_seed(42)
    helpers_testing.test_final_model(dataloaders['val'], class_names, model, writer, 'validation')
    if 'test' in dataloaders.keys():
        helpers_testing.test_final_model(dataloaders['test'], class_names, model, writer, 'test')
    # Saving the model
    print('Saving the model')
    save_path = '../models/model-' + writer.log_dir.split('/')[1] + '.pth'
    torch.save(model.state_dict(), save_path)
    

def k_fold_cross_val(training_dataset, class_names, learning_params, k_folds, num_epochs, batch_size, writer):
    '''The training_dataset is split into k_folds equal sized subsets.
    k_folds-1 of these subsets are used to train the model and the remaining dataset is used to validate the model.
    This is repeated till every subset was used once as validation set.

    Args:
        training_dataset (list<image:torch.tensor, class:str, item_idx:int>): List of the items of a FloodplainLandscapeElements, that are used for training
        dataset (FloodplainLandscapeElements): Dataset containing the landscape_elements from the fis_byhpass_complete, fish_bypass_0202 and knobloch project
        class_names (list of strings): List of classes of the landscape_elements, which should be detected by the network
        learning_params (FixedFeatureExtractor): Parameters (model, loss_function, optimizer, scheduler) of the CNN GoogLeNet set to execute transfer learning
        k_folds (int): Number of folds
        num_epochs (int): Number of epochs that will be run through during training
        batch_size (int): Number of landscape_elements after which the weights are updated
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard

    Returns:
        average_acc (int): Average of the accuracy values of the validation dataset in the last epoch over all folds
    '''
    acc_per_fold = [None]*k_folds

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(training_dataset)):
        #train_ids und val_ids enthalten nicht notwendigerweise gleich viele landscape elements jeder Klasse
        #falls gleiches Verhältnis gewünscht: StratifiedKFold
        print(f'Fold {fold+1}/{k_folds}')
        print('-'*10)
        acc_per_fold[fold]=0

        # Sample elements randomly from a given list of ids, no replacement.
        torch.manual_seed(42)
        subsampler = {'train': torch.utils.data.SubsetRandomSampler(train_ids),
                    'val': torch.utils.data.SubsetRandomSampler(val_ids)}
        #dataset_sizes = {x: len(subsampler[x]) for x in ['train', 'val']}

        # Define data loaders for training and validation data in this fold
        dataloaders = {x: DataLoader(training_dataset, batch_size=batch_size, sampler=subsampler[x]) for x in ['train', 'val']} #shuffle=True war noch eingetragen, mit subsampler aber raus. num_workers=4 wäre für paralleles programmieren
        
        # check number of landscape elements in each class in train and val data loader
        #count_large_wood_train, count_no_large_wood_train, count_large_wood_val, count_no_large_wood_val = helpers_CNN_uniform.get_number_landscape_elements_each_class(dataloaders)
        #print('count_large_wood_train', count_large_wood_train) #=161
        #print(helpers_CNN_uniform.get_number_landscape_elements(dataloaders['train'], 'large wood'))
        #print(helpers_CNN_uniform.get_number_landscape_elements(dataloaders['val'], 'large wood'))
        # print('count_no_large_wood_train', count_no_large_wood_train) =186
        # print('count_large_wood_val', count_large_wood_val) =186
        # print('count_no_large_wood_val', count_no_large_wood_val) =161 sum=694 passt
        
        # reset weights
        learning_params = FixedFeatureExtractor(learning_params.learning_rate, learning_params.step_size, learning_params.gamma, class_names) #scheduler verändert diese Werte nicht
        # Run the training loop for defined number of epochs
        print('Starting training and validation')
        start_training = time.time()
        for epoch in range(num_epochs):
            model = train_model(dataloaders['train'], class_names, learning_params, epoch, writer, fold=fold)
            test_model(dataloaders['val'], class_names, learning_params, epoch, writer, 'validation', acc_per_fold=acc_per_fold)
        print("Accuracy per fold:", acc_per_fold)
        
        time_training = time.time() - start_training
        print(f'Training and validation for fold {fold+1} complete in {time_training // 60:.0f}m {time_training % 60:.0f}s')
        print('-'*30)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('-'*30)
    sum = 0.0
    for i, acc in enumerate(acc_per_fold):
        print(f'Fold {i+1}: {acc*100.0} %')
        sum += acc*100
    average_acc = sum/len(acc_per_fold)
    print(f'Average: {average_acc} %')
    writer.add_text('Average accuracy over all folds: ', f'{average_acc} %',0)
    return average_acc

def get_dataloaders(class_names, batch_size, split='separated'):
    '''Returns dataloaders for 'train'-, 'val'- and 'test'-phase with the inserted batch_size.
    Depending on split, the data for train/val and test or either spatially split or spatially mixed.

    Args:
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network
        batch_size (int): Number of landscape elements after which the weights are updated
        split (either 'mixed' or 'separated', default='separated'): Determines whether the trainings-/validationset and the testset are spatial separated or not.
            If 'separated', data from knobloch is not involved in the training but provides the whole testset.
            If 'mixed', data from knobloch is involved in the training and the testset is (also) a mixture of data from fish_bypass and knobloch.

    Returns:
        dataloaders (dict of torch.utils.data.DataLoaders): Includes Dataloaders for 'train', 'val' and 'test' phase
    '''
    mixed_dataset = get_mixed_dataset(class_names)
    dataset_indices = list(range(len(mixed_dataset)))
    if split=='mixed':
        test_ids = np.linspace(0,len(dataset_indices), num=212, endpoint=False).astype(int).tolist()
        #test_ids = np.arange(0,len(dataset_indices),int(1/0.25)).tolist()[8:220]
    elif split=='separated':
        test_ids = []
        for i in dataset_indices:
            if 'Knobloch' in mixed_dataset.get_info(i)['raster'].GetDescription():
                test_ids.append(i)
    else:
        raise Exception('split has to be either "mixed" or "separated"')
    dataset_indices_left = [x for x in dataset_indices if x not in test_ids]
    val_ids = [dataset_indices_left[i] for i in np.arange(0,len(dataset_indices_left),int(1/0.2)).tolist()]
    train_ids = [x for x in dataset_indices_left if x not in val_ids]
    # Check in which dataset the images from the slides are
    # ref_slides=[38,89,93,133,446,477,503,694]
    # for idx in train_ids:
    #     if mixed_dataset.get_info(idx)["ref"] in ref_slides and 'Fischpass' in mixed_dataset.get_info(idx)["raster"].GetDescription():
    #         print(f'{mixed_dataset.get_info(idx)["ref"]} is in training')
    # for idx in test_ids:
    #     if mixed_dataset.get_info(idx)["ref"] in ref_slides and 'Fischpass' in mixed_dataset.get_info(idx)["raster"].GetDescription():
    #         print(f'{mixed_dataset.get_info(idx)["ref"]} is in testing')
    # for idx in val_ids:
    #     if mixed_dataset.get_info(idx)["ref"] in ref_slides and 'Fischpass' in mixed_dataset.get_info(idx)["raster"].GetDescription():
    #         print(f'{mixed_dataset.get_info(idx)["ref"]} is in validation')
    torch.manual_seed(42)
    subsampler = {'train': torch.utils.data.SubsetRandomSampler(train_ids), #268 large_wood, 288 everything else
                    'val': torch.utils.data.SubsetRandomSampler(val_ids), #68 large_wood, 72 everything else
                    'test': torch.utils.data.SubsetRandomSampler(test_ids)} #112 large_wood, 88 everything else #shuffles
    dataloaders = {x: DataLoader(mixed_dataset, batch_size=batch_size, sampler=subsampler[x]) for x in ['train', 'val', 'test']}
    return dataloaders

def get_dataloaders_no_test(class_names, batch_size):
    '''Returns dataloaders for 'train'- and 'val'- and phase with the inserted batch_size.
    The data is spatially mixed (fish_bypass and knobloch).

    Args:
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network
        batch_size (int): Number of landscape elements after which the weights are updated

    Returns:
        dataloaders (dict of torch.utils.data.DataLoaders): Includes Dataloaders for 'train' and 'val' phase
    '''
    mixed_dataset = get_mixed_dataset(class_names)
    dataset_indices = list(range(len(mixed_dataset)))
    val_ids = [dataset_indices[i] for i in np.arange(0,len(dataset_indices),int(1/0.2)).tolist()]
    train_ids = [x for x in dataset_indices if x not in val_ids]
    torch.manual_seed(42)
    subsampler = {'train': torch.utils.data.SubsetRandomSampler(train_ids),
                    'val': torch.utils.data.SubsetRandomSampler(val_ids)}
    dataloaders = {x: DataLoader(mixed_dataset, batch_size=batch_size, sampler=subsampler[x]) for x in ['train', 'val']}
    return dataloaders

def get_two_dataloaders_examples_in_test(class_names, batch_size):
    '''Returns dataloaders for 'train'- and 'val'- and phase with the inserted batch_size.
    The data is spatially mixed (fish_bypass and knobloch). The images from the slides are assigned to the validation dataset.

    Args:
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network
        batch_size (int): Number of landscape elements after which the weights are updated

    Returns:
        dataloaders (dict of torch.utils.data.DataLoaders): Includes Dataloaders for 'train' and 'val' phase
    '''
    mixed_dataset = get_mixed_dataset(class_names)
    dataset_indices = list(range(len(mixed_dataset)))
    val_ids = [dataset_indices[i] for i in np.arange(0,len(dataset_indices),int(1/0.2)).tolist()]
    ref_slides=[38,89,93,133,446,477,503,694]
    idx_slides=[17,87,91,131,545,577,601,793]
    for idx in idx_slides:
        if idx not in val_ids:
            val_ids[val_ids.index(round(idx/5)*5)]=idx
    train_ids = [x for x in dataset_indices if x not in val_ids]
    torch.manual_seed(42)
    subsampler = {'train': torch.utils.data.SubsetRandomSampler(train_ids),
                    'val': torch.utils.data.SubsetRandomSampler(val_ids)}
    dataloaders = {x: DataLoader(mixed_dataset, batch_size=batch_size, sampler=subsampler[x]) for x in ['train', 'val']}
    return dataloaders

def get_mixed_dataset(class_names):
    '''Merges the rasters and vectors from the projects fish_bypass_complete, fish_bypass_0202 and knobloch to one FloodplainLandscapeElements,
    which contains all detected landscape elements of the types listed in class_names.

    Args:
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network

    Returns:
        mixed_dataset (FloodplainLandscapeElements): Dataset containing the landscape elements from the fis_byhpass_complete, fish_bypass_0202 and knobloch project
    '''
    rasters_fish_bypass = [paths.raster1_fish_bypass]
    vectors_fish_bypass = [paths.vector1_fish_bypass]
    fish_bypass_complete = GISProject(rasters_fish_bypass, vectors_fish_bypass)
    rasters_fish_bypass_0202 = [paths.raster1_fish_bypass_0202]
    vectors_fish_bypass_0202 = [paths.vector1_fish_bypass_0202]
    fish_bypass_0202 = GISProject(rasters_fish_bypass_0202, vectors_fish_bypass_0202)
    rasters_knobloch = [paths.raster1_knobloch, paths.raster2_knobloch]
    vectors_knobloch = [paths.vector2_knobloch, paths.vector3_knobloch]
    knobloch = GISProject(rasters_knobloch, vectors_knobloch)
    mixed_project = FloodplainProject([fish_bypass_complete, fish_bypass_0202, knobloch]) #könnte noch um 0209 erweitert werden

    mixed_images_large_wood = FloodplainLandscapeElements(mixed_project, class_names[0], transform=transforms.ToTensor())
    length_large_wood=len(mixed_images_large_wood)
    mixed_dataset = FloodplainLandscapeElements(mixed_project, class_names, image_shape=(224,224), transform=transforms.ToTensor())

    #mean, std = mixed_dataset.get_mean_and_std()
    #print(mean,std)
    torch.manual_seed(42)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mixed_dataset.transform=data_transforms #danach ist mean=[0,0,0] und std=[1,1,1] (Funktion aus class rausholen, statt images dataloader und tensor verwenden)
    #print(len(mixed_dataset))
    return mixed_dataset

class FixedFeatureExtractor():
    '''Sets all parameters of the CNN GoogLeNet to execute transfer learning
    (freeze the weights for all layers except the final fully connected one, only those are updated during the training).

    Attributes:
        learning_rate (float): step size of the gradient descent algorithm
        step_size (int): Number of epochs after which the learning rate is changed by the factor gamma
        gamma (float): Factor by which the the learning rate is reduced after every step_size epochs
        model (torchvision.models): Set as googlenet.GoogLeNet with freezed parameters, except for the fc layer (params.requires_grad=False, only model.fc.parameters handed over to the optimizer)
        loss_function (torch.nn.modules.loss): Set as cross entropy loss function
        optimizer (torch.optim): Set as adam
        scheduler (torch.optim): Set as lr_scheduler.StepLR
        use_auxiliary (bool, default=False): If true, two auxiliary classifiers (whole branch) of googlenet are trained as well. Not wise in transfer learning.
    '''

    def __init__(self, learning_rate, step_size, gamma, class_names, use_auxiliary=False):
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.use_auxiliary = use_auxiliary
        if self.use_auxiliary:
            model = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
        else:
            model = googlenet(weights=GoogLeNet_Weights.DEFAULT) #or weights=GoogLeNet_Weights.IMAGENET1K_V1, gleich da googlenet nur eine Version von Gewichten hat. stimmt wohl mit (pretrained=True) überein
        # params.required_grad=True:
        # Every operation executed on params is saved in the autograd graph. During tensor.backward() pytorch passes through this graph in reversed direction to compute the gradients.
        # params.required_grad=False:
        # Freeze the params which should not be trained. For convenience: Freeze all and enable last layer during training
        for param in model.parameters():
            param.requires_grad = False
        if use_auxiliary:
            #aux1:
            #leave layers as initialized, just set requires_grad=True
            torch.manual_seed(42)
            for param in model.aux1.parameters():
                param.requires_grad = True
            #change number of out_features of the last layer
            in_ftrs_model_aux1_fc2 = model.aux1.fc2.in_features
            torch.manual_seed(42)
            model.aux1.fc2 = nn.Linear(in_ftrs_model_aux1_fc2, len(class_names))
            torch.manual_seed(42)
            #aux2:
            #leave layers as initialized, just set requires_grad=True
            for param in model.aux2.parameters():
                param.requires_grad = True
            #change number of out_features of the last layer
            in_ftrs_model_aux2_fc2 = model.aux1.fc2.in_features
            torch.manual_seed(42)
            model.aux2.fc2 = nn.Linear(in_ftrs_model_aux2_fc2, len(class_names))
            torch.manual_seed(42)
        #change number of features of the last layer (outside auxiliary):
        # Get the number of inputs to the fully connected layer
        in_ftrs_model_fc = model.fc.in_features
        # Change the number of outputs to the number of classes we have (fc is Linear, so no other change)
        torch.manual_seed(42)
        model.fc = nn.Linear(in_ftrs_model_fc, len(class_names)) #wird random intialisiert, nur gleich wenn direkt davor nochmal toch.manual_seed(42) gesetzt wird.
        torch.manual_seed(42)
        model = model.to(device)
        self.model = model
        torch.manual_seed(42)
        loss_function = nn.CrossEntropyLoss() #criterion
        self.loss_function = loss_function
        if use_auxiliary:
            # parameters of the auxiliary heads and of the final layer are being optimized
            #folgendes aktualisiert nur Gewichte von fc. Works only on SGD?
            #optimizer = optim.Adam([{'params': model.aux1.parameters(), 'params': model.aux2.parameters(),'params': model.fc.parameters()}],lr=learning_rate)
            optimizer = optim.Adam(list(model.aux1.parameters())+list(model.aux2.parameters())+list(model.fc.parameters()), lr=learning_rate)
        else:
            # only parameters of final layer are being optimized
            optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
        self.optimizer = optimizer
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        self.scheduler = exp_lr_scheduler

#falls noch finetuning machen:
# if transfer_learning_scenario == 'fixed feature extractor':
#     for param in model.parameters():
#         param.requires_grad = False
# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model.fc.in_features
# # Here the size of each output data point is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model.fc = nn.Linear(num_ftrs, 2)
# model = model.to(device)
# loss_function = nn.CrossEntropyLoss() #criterion
# if transfer_learning_scenario == 'finetuning':
#     # all parameters are being optimized
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# elif transfer_learning_scenario == 'fixed feature extractor':
#     # only parameters of final layer are being optimized
#     optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)