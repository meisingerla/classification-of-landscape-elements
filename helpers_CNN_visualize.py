import torch
import matplotlib.pyplot as plt
import helpers_CNN

'''Methods to visualize the data that passed the network'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow_own_norm(inp, dataset):
    '''Transforms a normalized image back to its original colours, when dataset's mean and std were used for normalization.

    Args:
        inp (torch.Tensor of shape (3, h, w) h und w andersrum???): Normalized image
    
    Returns:
        inp (numpy.array of shape (h, w, 3)): Image with the original colours
    '''
    inp = inp.permute((1,2,0))
    mean, std = dataset.get_mean_and_std()
    inp = std * inp.cpu().numpy() + mean
    return inp

def imshow(inp):
    '''Transforms a normalized image back to its original colours, when ImageNet's mean and std were used for normalization.

    Args:
        inp (torch.Tensor of shape (3, h, w) h und w andersrum???): Normalized image
    
    Returns:
        inp (numpy.array of shape (h, w, 3)): Image with the original colours
    '''
    inp = inp.permute((1,2,0))
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    inp = std * inp.cpu().numpy() + mean
    return inp

def get_highest_probs_each_batch(highest_probs_each_batch, highest_probs_item_idcs_each_batch, highest_probs_inputs_each_batch, highest_probs_preds_each_batch, selected_indices, probs, preds, inputs, item_idcs, class_name=None, dataset=None):
    '''Finds the probability values, items, images and predictions with the highest three probabilities among the selected indices.
    If selected_indices are the correct_indices, we will get the top three predicted landscape elements (highest probability in the rightly predicted class).
    If selected_indices are the incorrect_indices, we will get the three worst predicted landscape elements (highest probability in the misleadingly predicted class).
    If class_name is not None, only those landscape elements with the class_name as target are considered.

    Args:
        highest_probs_each_batch (torch.Tensor): Contains the highest three probability values of each batch already sampled
        highest_probs_item_idcs_each_batch (torch.Tensor): Contains the item_idcs (FlooplainLandscapeElements[item_idcs]) of the landscape elements with the highest three probabilities in each batch already sampled
        highest_probs_inputs_each_batch (list of torch.Tensor of shape (3,224,224)): Contains the rgb images of the landscape elements with the highest three probabilities in each batch already sampled
        highest_probs_preds_each_batch (torch.Tensor): Contains the predicted class of the landscape elements with the highest three probabilities in each batch already sampled
        selected_indices (torch.Tensor): Indices relating to the data in the batch (!= item_idcs of featue), which are either predicted correct or incorrect (depending on the variable passed as selected_indices)
        probs (torch.Tensor of len batch_size): Probability for the predicted class of each landscape element in the current batch
        preds (torch.Tensor): Predicted class of each landscape element in the current batch
        inputs (torch.tensor of shape(batch_size, 3, 224, 224)): Image of each landscape element in the current batch
        item_idcs (torch.Tensor of len batch_size): Index of each landscape element (FloodplainLandscapeElements[item_idcs]) in the current batch
        class_name (str, default = None): Possible class of the data, to specify which landscape elements are analyzed
        dataset (FloodplainLandscapeElements, default=None): Dataset containing the landscape elements intended for training the network. Only necessary if class_name is specified
        
    Returns:
        highest_probs_each_batch (torch.Tensor): Contains the highest three probability values of each batch already sampled and the current batch
        highest_probs_item_idcs_each_batch (torch.Tensor): Contains the item_idcs (FlooplainLandscapeElements[item_idcs]) of the landscape elements with the highest three probabilities in each batch already sampled and the current batch
        highest_probs_inputs_each_batch (list of torch.Tensor of shape (3,224,224)): Contains the rgb images of the landscape elements with the highest three probabilities in each batch already sampled and the current batch
        highest_probs_preds_each_batch (torch.Tensor): Contains the predicted class of the landscape elements with the highest three probabilities in each batch already sampled and the current batch
    '''
    selected_item_idcs = item_idcs[selected_indices.cpu()]
    #print('selected item idcs', selected_item_idcs)
    if class_name:
        selected_indices = selected_indices[[dataset[selected_item_idx][1]==class_name for selected_item_idx in selected_item_idcs]]
    selected_probs = torch.index_select(probs, 0, selected_indices)
    three_highest_probs = selected_probs.sort()[0][-3:]
    highest_probs_each_batch = torch.cat((highest_probs_each_batch, three_highest_probs))
    three_highest_probs_indices = selected_probs.sort()[1][-3:]
    three_highest_selected_indices = selected_indices[three_highest_probs_indices.cpu()]
    three_highest_probs_item_idcs = item_idcs[three_highest_selected_indices.cpu()]
    highest_probs_item_idcs_each_batch = torch.cat((highest_probs_item_idcs_each_batch, three_highest_probs_item_idcs.to(device)))
    #print('idcs of correct items', three_best_item_idcs)
    if three_highest_probs_item_idcs.shape!=three_highest_probs_item_idcs.unique().shape:
        raise Exception('Image is more than once in this list.')
    highest_probs_inputs_each_batch.extend([inputs[item] for item in three_highest_selected_indices])
    three_highest_probs_preds = preds[three_highest_selected_indices.cpu()]
    highest_probs_preds_each_batch = torch.cat((highest_probs_preds_each_batch, three_highest_probs_preds.to(device)))
    return highest_probs_each_batch, highest_probs_item_idcs_each_batch, highest_probs_inputs_each_batch, highest_probs_preds_each_batch

def show_three_striking_probs(highest_probs_each_batch, highest_probs_item_idcs_each_batch, highest_probs_inputs_each_batch, highest_probs_preds_each_batch, type_of_striking, writer, dataset, class_names, phase):
    '''Prints the three best and three worst predicted landscape elements in tensorboard.
    If less than 3 landscape elements fulfill the conditions, the number of landscape elements that fulfill the condition (0, 1 or 2) are printed

    Args:
        highest_probs_each_batch (torch.Tensor): Contains the highest three probability values of each batch (len: ceil(size_val_set/batch_size)*3)
        highest_probs_item_idcs_each_batch (torch.Tensor): Contains the item_idcs (FlooplainLandscapeElements[item_idcs]) of the landscape elements with the highest three probabilities in each batch
        highest_probs_inputs_each_batch (list of torch.Tensor of shape (3,224,224)): Contains the rgb images of the landscape elements with the highest three probabilities in each batch
        highest_probs_preds_each_batch (torch.Tensor): Contains the predicted class of the landscape elements with the highest three probabilities in each batch
        type_of_striking (str): Transmits classification to writer, either 'best' or 'worst'
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard (transmits striking images)
        dataset (FloodplainLandscapeElements): Dataset containing the landscape elements intended for training the network
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network
        phase (str): Defines for the writer if the test is applied on 'validation' or 'test1', 'test2', ... data 

    Returns: None
    '''
    probs = highest_probs_each_batch.sort()[0][-3:]
    indices = highest_probs_each_batch.sort()[1][-3:]
    item_idcs = highest_probs_item_idcs_each_batch[indices.cpu()]
    inputs = [highest_probs_inputs_each_batch[x] for x in indices.cpu()]
    preds_numbers = highest_probs_preds_each_batch[indices.cpu()]
    preds = [helpers_CNN.number_to_name(class_names, preds_number) for preds_number in preds_numbers]
    class_names = [dataset.get_info(item_idx)["class"] for item_idx in item_idcs]
    if type_of_striking[:4]=='best':
        if preds!=class_names:
            raise Exception('Preds and class names should be the same if type of striking is "best"')
    elif type_of_striking[:5]=='worst':
        for i in range(len(preds)):
            if preds[i]==class_names[i]:
                raise Exception('Preds and class names should not be the same if type of striking is "worst"')
    #writer.add_images('three ' + type_of_striking, torch.stack(inputs), 0, dataformats='NCHW')
    if len(probs)==0:
        fig = plt.figure()
        plt.title('No image fulfills the conditions', fontsize=20)
        plt.axis('off')
    else:
        number_subplots = min(len(probs),3)
        fig = plt.figure()
        plt.subplots_adjust(wspace=0.8)
        for i in range(number_subplots):
            plt.subplot(1,number_subplots, i+1, title=f'prob: {probs[i]:.4f} \n ref: {dataset.get_info(item_idcs[i])["ref"]} \n class: {class_names[i]} \n pred: {preds[i]}')
            plt.imshow((imshow(inputs[i])*255).astype('uint8'))
            plt.axis('off')
    writer.add_figure('three ' + type_of_striking + ' ' + phase, fig, 0)