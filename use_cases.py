from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
import helpers_CNN
import helpers_testing
import numpy as np
import time

'''Use cases of the Bachelor's Thesis
"Geospatial Mapping of River-Floodplain Landscape Elements
from Aerial Imagery using Pretrained Convolutional Neural Networks", callable in main.py'''

def initialize_writer(phases):
    '''Initializes a tensorboard writer with a layout consistent with the phases.

    Args:
        phases (list<str>): List of the phases in the use case, for example ['train', validation', 'test1', 'test2']

    Returns:
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard
    '''
    from torch.utils.tensorboard import SummaryWriter
    layout = {
        "Learning curves": {
            "Loss": ["Multiline", [f"Loss/{x}" for x in phases]],
            "Accuracy": ["Multiline", [f"Accuracy/{x}" for x in phases]],
        },
    }
    writer = SummaryWriter()
    writer.add_custom_scalars(layout)
    return writer

def finetune_hyperparameters(class_names, training_dataset):
    '''Finds the best hyperparameters among a pre-selection by applying 5-fold cross-validations.

    Args:
        class_names (list<str>): Classes that the model learns to identify
        training_dataset (list<image:torch.tensor, class:str, item_idx:int>): List of the items of a FloodplainLandscapeElements, that are used for training

    Returns:
        best_learning_rate (int): identified learning rate, confirmed by the validation
        best_batch_size (int): identified batch size, confirmed by the validation
        best_step_size (int): identified step size, confirmed by the validation
        best_num_epochs (int): identified number of epochs, confirmed by the validation
    '''
    # Executing several 5-fold-cross-validations to find the best hyperparameters
    k_folds=5
    gamma = 0.1
    max_av_acc_val = 0
    opts_learning_rate = [0.1, 0.01, 0.001, 0.0001]
    for learning_rate in opts_learning_rate:
        step_size=30
        transfer_learning_params = helpers_CNN.FixedFeatureExtractor(learning_rate, step_size, gamma, class_names)
        num_epochs=30
        batch_size=64
        writer = initialize_writer(['train', 'validation'])
        print_hyperparameters = '5-fold cv: ' + 'num_epochs: ' + str(num_epochs) + ', learning_rate: ' + str(learning_rate) + ', batch size: ' + str(batch_size) + ', step_size: ' + str(step_size) + ', gamma: ' + str(gamma)
        writer.add_text('Hyperparameters:', print_hyperparameters)
        av_acc_val = helpers_CNN.k_fold_cross_val(training_dataset, class_names, transfer_learning_params, k_folds, num_epochs, batch_size, writer)
        if av_acc_val>max_av_acc_val:
            best_learning_rate = learning_rate
            max_av_acc_val=av_acc_val
        writer.close()
    best_batch_size=64 #damit Verglich auch funktioniert, falls beste batch size schon die vorausgewählte war
    opts_batch_size = [32,128] #64 wurde ja schon untersucht
    for batch_size in opts_batch_size:
        step_size=30
        transfer_learning_params = helpers_CNN.FixedFeatureExtractor(best_learning_rate, step_size, gamma, class_names)
        num_epochs=30
        writer = initialize_writer(['train', 'validation'])
        print_hyperparameters = '5-fold cv: trainingsdata: fish_bypass_complete and fish_bypass_0202' + ', num_epochs: ' + str(num_epochs) + ', learning_rate: ' + str(best_learning_rate) + ', batch size: ' + str(batch_size) + ', step_size: ' + str(step_size) + ', gamma: ' + str(gamma)
        writer.add_text('Hyperparameters:', print_hyperparameters)
        av_acc_val = helpers_CNN.k_fold_cross_val(training_dataset, class_names, transfer_learning_params, k_folds, num_epochs, batch_size, writer)
        if av_acc_val>max_av_acc_val:
            best_batch_size = batch_size
            max_av_acc_val=av_acc_val
        writer.close()
    best_step_size=30
    opts_step_size = [10,20] #30 wurde ja schon untersucht
    for step_size in opts_step_size:
        transfer_learning_params = helpers_CNN.FixedFeatureExtractor(best_learning_rate, step_size, gamma, class_names)
        num_epochs=30
        writer = initialize_writer(['train', 'validation'])
        print_hyperparameters = '5-fold cv: trainingsdata: fish_bypass_complete and fish_bypass_0202' + ', num_epochs: ' + str(num_epochs) + ', learning_rate: ' + str(best_learning_rate) + ', batch size: ' + str(best_batch_size) + ', step_size: ' + str(step_size) + ', gamma: ' + str(gamma)
        writer.add_text('Hyperparameters:', print_hyperparameters)
        av_acc_val = helpers_CNN.k_fold_cross_val(training_dataset, class_names, transfer_learning_params, k_folds, num_epochs, best_batch_size, writer)
        if av_acc_val>max_av_acc_val:
            best_step_size = step_size
            max_av_acc_val=av_acc_val
        writer.close()
    best_num_epochs=30
    opts_num_epochs = [20, 50] #30 wurde ja schon untersucht
    for num_epochs in opts_num_epochs:
        transfer_learning_params = helpers_CNN.FixedFeatureExtractor(best_learning_rate, best_step_size, gamma, class_names)
        writer = initialize_writer(['train', 'validation'])
        print_hyperparameters = '5-fold cv: trainingsdata: fish_bypass_complete and fish_bypass_0202' + ', num_epochs: ' + str(num_epochs) + ', learning_rate: ' + str(best_learning_rate) + ', batch size: ' + str(best_batch_size) + ', step_size: ' + str(best_step_size) + ', gamma: ' + str(gamma)
        writer.add_text('Hyperparameters:', print_hyperparameters)
        av_acc_val = helpers_CNN.k_fold_cross_val(training_dataset, class_names, transfer_learning_params, k_folds, num_epochs, best_batch_size, writer)
        if av_acc_val>max_av_acc_val:
            best_num_epochs = num_epochs
            max_av_acc_val=av_acc_val
        writer.close()
    # Train model with identified hyperparameters on training and validation data
    #torch.manual_seed(42)
    #trainloader = DataLoader(training_dataset, batch_size=best_batch_size, shuffle=True)
    #best_learning_params = helpers_CNN.FixedFeatureExtractor(best_learning_rate, best_step_size, gamma, class_names)
    #writer = initialize_writer(['train', 'validation'])
    #print_hyperparameters = 'train model with best hyperparameters after cv: num_epochs: ' + str(best_num_epochs) + ', learning_rate: ' + str(best_learning_rate) + ', batch size: ' + str(best_batch_size) + ', step_size: ' + str(best_step_size) + ', gamma: ' + str(gamma)
    #writer.add_text('Hyperparameters:', print_hyperparameters)
    #for epoch in range(best_num_epochs):
    ##    torch.manual_seed(42)
    #    model = helpers_CNN.train_model(trainloader, class_names, best_learning_params, epoch, writer)
    #writer.close()
    ## Saving the model
    ## save_path = f'./model-fold-{fold}.pth'
    ## torch.save(model.state_dict(), save_path)
    return best_learning_rate, best_batch_size, best_step_size, best_num_epochs
    
def split_dataset_train_two_tests(dataset, split='separated'):
    '''Divides the whole dataset into one train and two test dataset by assigning alternating indices.
    Since the data is sequentially ordered into classes, the alternation provides approximately balanced datasets.

    Args:
        dataset (FloodplainLandscapeElements): Dataset containing the landscape elements from the fish_bypass_complete, fish_bypass_0202 and knobloch project
        split (either 'mixed' or 'separated', default='separated'): Determines whether the trainings- and the testsets are spatial separated or not.
            If 'separated', data from knobloch is not involved in the training but provides the second testset.
            If 'mixed', data from knobloch is involved in the training and both testsets are (also) a mixture of data from fish_bypass and knobloch.

    Returns:
        train_ids (list<int>): Item indices of the dataset, that should be used for training (contains 500 data points)
        test1_ids (list<int>): Item indices of the dataset, that should be used for the first testing (contains 204 data points)
        test2_ids (list<int>): Item indices of the dataset, that should be used for the second testing (contains 212 data points)
    '''
    dataset_indices = list(range(len(dataset)))
    if split=='mixed':
        test2_ids = np.linspace(0,len(dataset_indices), num=212, endpoint=False).astype(int).tolist()
    elif split=='separated':
        test2_ids = []
        for i in dataset_indices:
            if 'Knobloch' in dataset.get_info(i)['raster'].GetDescription():
                test2_ids.append(i)
    else:
        raise Exception('split has to be either "mixed" or "separated"')
    dataset_indices_left = [x for x in dataset_indices if x not in test2_ids]
    test1_ids = [dataset_indices_left[i] for i in np.linspace(0,len(dataset_indices_left),num=204,endpoint=False).astype(int).tolist()]
    train_ids = [x for x in dataset_indices_left if x not in test1_ids]
    return train_ids, test1_ids, test2_ids

def cv_separated_training():
    '''Executes several 5-fold cross-validation with different hyperparameters on spatially separated dataset (from fish bypass (Inn)).

    Args:
        None
    
    Returns:
        None
    '''
    writer = initialize_writer(['train', 'test1', 'test2'])
    writer.add_text('Use case:', '5-fold cross-validation on spatially separated datasets (including data from fish bypass (Inn))')
    # Set class_names and distribution of collected data in training, test1 and test2 dataset
    class_names = ['large wood', 'no large wood']
    writer.add_text('class names:', str(class_names))
    mixed_dataset = helpers_CNN.get_mixed_dataset(class_names)
    train_ids, test1_ids, test2_ids = split_dataset_train_two_tests(mixed_dataset, split='separated')
    training_dataset = []
    for train_idx in train_ids:
        training_dataset.append(mixed_dataset[train_idx]) #247 large wood, 253 no large wood
    # Identify hyperparameters
    best_learning_rate, best_batch_size, best_step_size, best_num_epochs = finetune_hyperparameters(class_names, training_dataset)
    writer.add_text('Identified hyperparameters:', f'learning rate = {best_learning_rate}, batch size = {best_batch_size}, step size = {best_step_size}, number of epochs = {best_num_epochs} and gamma = {0.1}')
    writer.close()

def test_separated_training():
    '''Trains model on data from fish bypass (Inn) and tests model on data from Knoblochsaue (Rhine).

    Args:
        None
    
    Returns:
        None
    '''
    writer = initialize_writer(['train', 'test1', 'test2'])
    writer.add_text('Use case:', 'Test separated training: Training and both test data derive from spatially separated datasets. Training: Fish bypass, Test1: Fish bypass, Test2: Knoblochsaue')
    # Set class_names and distribution of collected data in training, test1 and test2 dataset
    class_names = ['large wood', 'no large wood']
    writer.add_text('class names:', str(class_names))
    mixed_dataset = helpers_CNN.get_mixed_dataset(class_names)
    train_ids, test1_ids, test2_ids = split_dataset_train_two_tests(mixed_dataset, split='separated')
    training_dataset = []
    for train_idx in train_ids:
        training_dataset.append(mixed_dataset[train_idx]) #247 large wood, 253 no large wood
    # Hyperparameters identified through
    #finetune_hyperparameters(class_names, training_dataset)
    learning_rate = 0.001
    step_size = 30
    gamma = 0.1
    num_epochs = 30
    batch_size = 32
    transfer_learning_params = helpers_CNN.FixedFeatureExtractor(learning_rate, step_size, gamma, class_names)
    print_hyperparameters = 'trainingsdata: fish_bypass_alles and fish_bypass_0202' + ', num_epochs: ' + str(num_epochs) + ', learning_rate: ' + str(learning_rate) + ', batch size: ' + str(batch_size) + ', step_size: ' + str(step_size) + ', gamma: ' + str(gamma)
    writer.add_text('Hyperparameters:', print_hyperparameters)
    # Determine trainloader (includes data from fish_bypass)
    trainloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    # Determine two testloaders (test1 contains data from fish_bypass, test2 contains data from knoblochsaue)
    torch.manual_seed(42)
    subsampler = {'test1': torch.utils.data.SubsetRandomSampler(test1_ids),
                    'test2': torch.utils.data.SubsetRandomSampler(test2_ids)}
    testloaders = {x: DataLoader(mixed_dataset, sampler=subsampler[x], shuffle=False) for x in ['test1', 'test2']}
    # Run through all epochs, measure loss and accuracy for train and both test sets.
    model = get_model_with_learning_curves(trainloader, testloaders, class_names, transfer_learning_params, num_epochs, 2, writer)
    # Make further evaluation on the final model (best/worst images, confusion matrix, ...)
    print('Starting final testing')
    start_testing = time.time()
    torch.manual_seed(42)
    helpers_testing.test_final_model(testloaders['test1'], mixed_dataset, class_names, model, writer, 'test1')
    torch.manual_seed(42)
    helpers_testing.test_final_model(testloaders['test2'], mixed_dataset, class_names, model, writer, 'test2')
    time_testing = time.time() - start_testing
    print(f'Final testing process has finished in {time_testing // 60:.0f}m {time_testing % 60:.0f}s')
    writer.close()

def split_dataset_train_one_test(dataset):
    ''''Divides the whole dataset into one train and one test dataset by assigning alternating indices.
    Since the data is sequentially ordered into classes, the alternation provides approximately balanced datasets.
    Train and test dataset both contain spatially mixed data.
    The test dataset contains 1/5 of the data, the train dataset the other 4/5 of the data.
    
    Args:
        dataset (FloodplainLandscapeElements): Dataset containing the landscape elements from the fis_byhpass_complete, fish_bypass_0202 and knobloch project

    Returns:
        train_dataset (list): Items of the dataset that should be used for training, stored in a list<tensor, str, int>
        test_dataset (list): Items of the dataset that should be used for testing, stored in a list<tensor, str, int>
    '''
    dataset_indices = list(range(len(dataset)))
    test_ids = [dataset_indices[i] for i in np.arange(0,len(dataset_indices),int(1/0.25)).tolist()]
    train_ids = [x for x in dataset_indices if x not in test_ids]
    train_dataset = []
    for train_idx in train_ids:
        train_dataset.append(dataset[train_idx])
    test_dataset = []
    for test_idx in test_ids:
        test_dataset.append(dataset[test_idx])
    return train_dataset, test_dataset

def cv_mixed_training():
    '''Executes several 5-fold cross-validation with different hyperparameters on spatially mixed dataset (from fish bypass (Inn) and Knoblochsaue (Rhine)).

    Args:
        None

    Returns:
        None
    '''
    writer = initialize_writer(['train', 'test1', 'test2'])
    writer.add_text('Use case:', '5-fold cross-validation on spatially mixed datasets (including data from fish bypass (Inn) and Knobochsaue (Rhine))')
    # Set class_names and distribution of collected data in training and test dataset
    class_names = ['large wood', 'no large wood']
    writer.add_text('class names:', str(class_names))
    mixed_dataset = helpers_CNN.get_mixed_dataset(class_names)
    train_ids, test1_ids, test2_ids = split_dataset_train_two_tests(mixed_dataset, split='mixed') #test1 enthält 47 data points von Knoblochsaue und 157 data points von fish bypass
    training_dataset = []
    for train_idx in train_ids:
        training_dataset.append(mixed_dataset[train_idx]) #244 large wood, 256 no large wood
    # Identify hyperparameters
    best_learning_rate, best_batch_size, best_step_size, best_num_epochs = finetune_hyperparameters(class_names, training_dataset)
    writer.add_text('Identified hyperparameters:', f'learning rate = {best_learning_rate}, batch size = {best_batch_size}, step size = {best_step_size}, number of epochs = {best_num_epochs} and gamma = {0.1}')
    writer.close()

def test_mixed_training():
    '''Trains and tests model on spatially mixed dataset (from fish bypass (Inn) and Knoblochsaue (Rhine)).

    Args:
        None

    Returns:
        None
    '''
    writer = initialize_writer(['train', 'test1', 'test2'])
    writer.add_text('Use case:', 'Test mixed training: Training and both test data derive from spatially mixed datasets (including data from Fish bypass and Knobochsaue)')
    # Set class_names and distribution of collected data in training and test dataset
    class_names = ['large wood', 'no large wood']
    writer.add_text('class names:', str(class_names))
    mixed_dataset = helpers_CNN.get_mixed_dataset(class_names)
    train_ids, test1_ids, test2_ids = split_dataset_train_two_tests(mixed_dataset, split='mixed') #test1 enthält 47 data points von Knoblochsaue und 157 data points von fish bypass
    training_dataset = []
    for train_idx in train_ids:
        training_dataset.append(mixed_dataset[train_idx]) #244 large wood, 256 no large wood
    # Hyperparameters identified through
    #finetune_hyperparameters(class_names, training_dataset)
    learning_rate = 0.001
    step_size = 30
    gamma = 0.1
    num_epochs = 30
    batch_size = 64
    transfer_learning_params = helpers_CNN.FixedFeatureExtractor(learning_rate, step_size, gamma, class_names)
    print_hyperparameters = 'trainingsdata: fish_bypass_alles, fish_bypass_0202 and knoblochsaue' + ', num_epochs: ' + str(num_epochs) + ', learning_rate: ' + str(learning_rate) + ', batch size: ' + str(batch_size) + ', step_size: ' + str(step_size) + ', gamma: ' + str(gamma)
    writer.add_text('Hyperparameters:', print_hyperparameters)
    #av_acc_val = helpers_CNN.k_fold_cross_val(training_dataset, class_names, transfer_learning_params, 5, num_epochs, batch_size, writer)
    # Determine trainloader (includes spatially mixed data)
    trainloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    # Determine two testloaders (test1 and test2 contain data from both study sites)
    torch.manual_seed(42)
    subsampler = {'test1': torch.utils.data.SubsetRandomSampler(test1_ids),
                    'test2': torch.utils.data.SubsetRandomSampler(test2_ids)}
    testloaders = {x: DataLoader(mixed_dataset, sampler=subsampler[x], shuffle=False) for x in ['test1', 'test2']}
    # Run through all epochs, measure loss and accuracy for train and both test sets.
    model = get_model_with_learning_curves(trainloader, testloaders, class_names, transfer_learning_params, num_epochs, 2, writer)
    # Make further evaluation on the final model (best/worst images, confusion matrix, ...)
    torch.manual_seed(42)
    helpers_testing.test_final_model(testloaders['test1'], mixed_dataset, class_names, model, writer, 'test1')
    torch.manual_seed(42)
    helpers_testing.test_final_model(testloaders['test2'], mixed_dataset, class_names, model, writer, 'test2')
    writer.close()

def test_influence_number_of_classes(class_names):
    '''Trains a model to identify the inserted classes.
    Training with 687 data points and testing with 229 data points, both from mixed dataset.

    Args:
        class_names (list<str>): Classes that the model learns to identify

    Returns:
        None
    '''
    writer = initialize_writer(['train', 'test'])
    writer.add_text('class names:', str(class_names))
    mixed_dataset = helpers_CNN.get_mixed_dataset(class_names)
    training_dataset, test_dataset = split_dataset_train_one_test(mixed_dataset)
    # Hyperparameters identified through finetune_hyperparameters(['large wood', 'everything else'], training_dataset):
    learning_rate = 0.001
    step_size = 30
    gamma = 0.1
    num_epochs = 30
    batch_size = 64
    transfer_learning_params = helpers_CNN.FixedFeatureExtractor(learning_rate, step_size, gamma, class_names)
    #av_acc_val = helpers_CNN.k_fold_cross_val(training_dataset, class_names, transfer_learning_params, 5, num_epochs, batch_size, writer)
    print_hyperparameters = 'trainingsdata: fish_bypass_alles, fish_bypass_0202 and knoblochsaue' + ', num_epochs: ' + str(num_epochs) + ', learning_rate: ' + str(learning_rate) + ', batch size: ' + str(batch_size) + ', step_size: ' + str(step_size) + ', gamma: ' + str(gamma)
    writer.add_text('Hyperparameters:', print_hyperparameters)
    # Determine trainloader (includes spatially mixed data)
    torch.manual_seed(42)
    trainloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True) #durch shuffle is sampler set to RandomSampler
    # Determine testloader (includes spatially mixed data)
    torch.manual_seed(42)
    testloader = DataLoader(test_dataset) #thus shuffle=False and sampler=SequentialSampler
    # Run through all epochs, measure loss and accuracy for train and test sets.
    model = get_model_with_learning_curves(trainloader, testloader, class_names, transfer_learning_params, num_epochs, 1, writer)
    # Make further evaluation on the final model (best/worst images, confusion matrix, ...)
    torch.manual_seed(42)
    helpers_testing.test_final_model(testloader, mixed_dataset, class_names, model, writer, 'test')
    writer.close()

def test_influence_number_of_data_points(number):
    '''Trains the model with the inserted number of data points from the mixed dataset and tests model on 229 data points.

    Args:
        number (int): number of data points used in training

    Returns:
        None
    '''
    writer = initialize_writer(['train', 'test'])
    class_names = ['large wood', 'no large wood']
    writer.add_text('class names:', str(class_names))
    writer.add_text('number of data points in training:', str(number))
    mixed_dataset = helpers_CNN.get_mixed_dataset(class_names)
    training_dataset, test_dataset = split_dataset_train_one_test(mixed_dataset)
    training_dataset_limited = [training_dataset[i] for i in np.linspace(0,len(training_dataset), num=number, endpoint=False).astype(int).tolist()]
    # Hyperparameters identified through finetune_hyperparameters(class_names, training_dataset):
    learning_rate = 0.001
    step_size = 30
    gamma = 0.1
    num_epochs = 30
    batch_size = 64
    transfer_learning_params = helpers_CNN.FixedFeatureExtractor(learning_rate, step_size, gamma, class_names)
    #av_acc_val = helpers_CNN.k_fold_cross_val(training_dataset, class_names, transfer_learning_params, 5, num_epochs, batch_size, writer)
    print_hyperparameters = 'trainingsdata: fish_bypass_alles, fish_bypass_0202 and knoblochsaue' + ', num_epochs: ' + str(num_epochs) + ', learning_rate: ' + str(learning_rate) + ', batch size: ' + str(batch_size) + ', step_size: ' + str(step_size) + ', gamma: ' + str(gamma)
    writer.add_text('Hyperparameters:', print_hyperparameters)
    # Determine testloader (includes spatially mixed data)
    torch.manual_seed(42)
    testloader = DataLoader(test_dataset) #thus shuffle=False and sampler=SequentialSampler
    if number==0:
        model = transfer_learning_params.model
        for epoch in range(num_epochs):
            torch.manual_seed(42)
            helpers_CNN.test_model(testloader, class_names, transfer_learning_params, epoch, writer, 'test')
    else:
        # Determine trainloader (includes spatially mixed data)
        torch.manual_seed(42)
        trainloader = DataLoader(training_dataset_limited, batch_size=batch_size, shuffle=True) #durch shuffle is sampler set to RandomSampler
        # Run through all epochs, measure loss and accuracy for train and test sets.
        model = get_model_with_learning_curves(trainloader, testloader, class_names, transfer_learning_params, num_epochs, 1, writer)
    # Make further evaluation on the final model (best/worst images, confusion matrix, ...)
    torch.manual_seed(42)
    helpers_testing.test_final_model(testloader, mixed_dataset, class_names, model, writer, 'test')
    writer.close()

def get_model_with_learning_curves(trainloader, testloaders, class_names, learning_params, num_epochs, number_of_tests, writer):
    '''Trains model with the inserted parameters and tests model on the inserted amount of test datasets.

    Args:
        trainloader
        testloaders (dict<torch.utils.data.DataLoader> or torch.utils.data.DataLoader): Contains batches of data points for each test dataset. Keys are 'test1', 'test2', usw.
        class_names (list<str>): Classes that the model learns to identify
        learning_params (FixedFeatureExtractor): Parameters (model, loss_function, optimizer, scheduler) of the CNN GoogLeNet set to execute transfer learning
        num_epochs (int): Number of epochs that will be run through during training
        number_of_tests (int): Number of test datasets listed in testloaders
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard

    Returns:
        model (torchvision.models): GoogLeNet with adjusted parameters in the fc layer
    '''
    # Run through all epochs, measure loss and accuracy for train and all test sets.
    print('Starting training and test along the way')
    start_training = time.time()
    model=learning_params.model
    if number_of_tests==1:
        for epoch in range(num_epochs):
            torch.manual_seed(42)
            model = helpers_CNN.train_model(trainloader, class_names, learning_params, epoch, writer)
            torch.manual_seed(42)
            helpers_CNN.test_model(testloaders, class_names, learning_params, epoch, writer, 'test')
    elif number_of_tests==2:
        for epoch in range(num_epochs):
            torch.manual_seed(42)
            model = helpers_CNN.train_model(trainloader, class_names, learning_params, epoch, writer)
            torch.manual_seed(42)
            helpers_CNN.test_model(testloaders['test1'], class_names, learning_params, epoch, writer, 'test1')
            torch.manual_seed(42)
            helpers_CNN.test_model(testloaders['test2'], class_names, learning_params, epoch, writer, 'test2')
    else:
        raise Exception('Number of tests can be either 1 or 2, nothing else.')
    time_training = time.time() - start_training
    print(f'Training process has finished in {time_training // 60:.0f}m {time_training % 60:.0f}s')
    return model

def save_model_le_2classes_mixed_687():
    '''Trains the model with 3/4 of the data points from the mixed dataset on the classes "large wood" and "no large wood" and saves the developed model.

    Args:
        None

    Returns:
        None
    '''
    writer = initialize_writer(['train'])
    class_names = ['large wood', 'no large wood']
    mixed_dataset = helpers_CNN.get_mixed_dataset(class_names)
    training_dataset, test_dataset = split_dataset_train_one_test(mixed_dataset)
    # Hyperparameters identified through finetune_hyperparameters(class_names, training_dataset):
    learning_rate = 0.001
    step_size = 30
    gamma = 0.1
    num_epochs = 30
    batch_size = 64
    transfer_learning_params = helpers_CNN.FixedFeatureExtractor(learning_rate, step_size, gamma, class_names)
    # Determine trainloader (includes spatially mixed data)
    torch.manual_seed(42)
    trainloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True) #durch shuffle is sampler set to RandomSampler
    # Run through all epochs, measure loss and accuracy for train set.
    for epoch in range(num_epochs):
        torch.manual_seed(42)
        model = helpers_CNN.train_model(trainloader, class_names, transfer_learning_params, epoch, writer)
    # Saving the model
    save_path = f'./model_le_2classes_mixed_687.pth'
    torch.save(model.state_dict(), save_path)

def save_model_le_7classes_mixed_687():
    '''Trains the model with 3/4 of the data points from the mixed dataset on the classes 'large wood', 'sand', 'wet', 'shotrock', 'coarse-gravel', 'grass', 'everything else' and saves the developed model.

    Args:
        None

    Returns:
        None
    '''
    writer = initialize_writer(['train'])
    class_names = ['large wood', 'sand', 'wet', 'shotrock', 'coarse-gravel', 'grass', 'everything else']
    mixed_dataset = helpers_CNN.get_mixed_dataset(class_names)
    training_dataset, test_dataset = split_dataset_train_one_test(mixed_dataset)
    # Hyperparameters identified through finetune_hyperparameters(class_names, training_dataset):
    learning_rate = 0.001
    step_size = 30
    gamma = 0.1
    num_epochs = 30
    batch_size = 64
    transfer_learning_params = helpers_CNN.FixedFeatureExtractor(learning_rate, step_size, gamma, class_names)
    # Determine trainloader (includes spatially mixed data)
    torch.manual_seed(42)
    trainloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True) #durch shuffle is sampler set to RandomSampler
    # Run through all epochs, measure loss and accuracy for train set.
    for epoch in range(num_epochs):
        torch.manual_seed(42)
        model = helpers_CNN.train_model(trainloader, class_names, transfer_learning_params, epoch, writer)
    # Saving the model
    save_path = f'./model_le_7classes_mixed_687.pth'
    torch.save(model.state_dict(), save_path)