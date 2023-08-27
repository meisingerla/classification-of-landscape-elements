from __future__ import print_function, division
import torch
import paths
import helpers_CNN
import helpers_CNN_visualize
from torch.utils.data import DataLoader 
from classes_data_preprocessing import GISProject
from classes_data_preprocessing import FloodplainProject
from classes_data_preprocessing import FloodplainLandscapeElements
from torchvision import transforms
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

'''Methods for testing an already trained model.'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    '''Generates a GoogLeNet model with adjusted parameters of the fc layer saved in modul_name under the path ../models/.

    Args:
        model_name (str): File name of the trained GoogLeNet model

    Returns:
        model (torchvision.models): GoogLeNet with adjusted parameters in the fc layer
    '''
    model_path = '../models/' + model_name
    model = googlenet(weights=GoogLeNet_Weights.DEFAULT) #or weights=GoogLeNet_Weights.IMAGENET1K_V1, gleich da googlenet nur eine Version von Gewichten hat.
    for param in model.parameters():
        param.requires_grad = False
    # Get the number of inputs to the fully connected layer
    num_ftrs = model.fc.in_features
    # Change the number of outputs to the number of classes we have (fc is Linear, so no other change)
    model.fc = nn.Linear(num_ftrs, len(torch.load(model_path, map_location=torch.device(device))['fc.bias']))
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model

def get_pred_certain_refs(refs, dataset, item_idcs, targets, predictions):
    for j,item_idx in enumerate(item_idcs):
        if dataset.get_info(item_idx)["ref"] in refs and 'Fischpass' in dataset.get_info(item_idx)["raster"].GetDescription():
            print(dataset.get_info(item_idx)["ref"])
            print(targets[j])
            print(predictions[j])
            print()

def test_final_model(testloader, dataset, class_names, model, writer, phase, acc_per_fold=None):
    '''Testing a trained model.
    Thus evaluation of specific landscape elements, e.g. best/worst predicitions.

    Args:
        testloader (torch.utils.data.DataLoader): Contains batches of the landscape elements which are used to test the model
        class_names (list of strings): List of classes of the landscape elements, which should be detected by the network
        model (torchvision.models): Trained model
        writer (torch.utils.tensorboard.writer.SummaryWriter): Transmitter to tensorboard (transmits loss and accuracy)
        phase (str): Defines for the writer if the test is applied on 'validation' or 'test1', 'test2', ... data 
        acc_per_fold (list, default=None): Accuracy of the final model of each fold of cross validation, None for hold_out method

    Returns:
        None
    '''
    correct_count = 0
    correct_count_per_class = {class_name: 0 for class_name in class_names}
    total_count_per_class = {class_name: 0 for class_name in class_names}
    best_probs_each_batch = torch.empty(0).to(device)
    best_item_idcs_each_batch = torch.empty(0, dtype=int).to(device)
    best_inputs_each_batch = []
    best_preds_each_batch = torch.empty(0, dtype=int).to(device)
    worst_probs_each_batch = torch.empty(0).to(device)
    worst_item_idcs_each_batch = torch.empty(0, dtype=int).to(device)
    worst_inputs_each_batch = []
    worst_preds_each_batch = torch.empty(0, dtype=int).to(device)
    best_probs_each_batch_large_wood = torch.empty(0).to(device)
    best_item_idcs_each_batch_large_wood = torch.empty(0, dtype=int).to(device)
    best_inputs_each_batch_large_wood = []
    best_preds_each_batch_large_wood = torch.empty(0, dtype=int).to(device)
    worst_probs_each_batch_large_wood = torch.empty(0).to(device)
    worst_item_idcs_each_batch_large_wood = torch.empty(0, dtype=int).to(device)
    worst_inputs_each_batch_large_wood = []
    worst_preds_each_batch_large_wood = torch.empty(0, dtype=int).to(device)
    all_targets = torch.empty(0, dtype=int).to(device)
    all_preds = torch.empty(0, dtype=int).to(device)
    # Set the mode to evaluation (e.g. dropout is deactivated)
    model.eval()
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
            targets = helpers_CNN.names_to_numbers(class_names, targets_str)
            targets = targets.to(device)
            all_targets = torch.cat((all_targets, targets))

            # Generate outputs
            outputs = model(inputs)
            outputs_normed = nn.functional.softmax(outputs.data, dim=1)
            writer.add_text(f'Probability of image being in class "{class_names[0]}":', str(outputs_normed[:,0]), i)
            writer.add_text(f'Probability of image being in class "{class_names[1]}":', str(outputs_normed[:,1]), i)
            
            # Set total and correct
            probs, predictions = torch.max(outputs_normed, 1) #outputs und outputs.data ist eig gleich, siehe Erklärung oben
            all_preds = torch.cat((all_preds, predictions))

            # Accuracy statistics
            correct_indices = (targets==predictions).nonzero(as_tuple=True)[0]
            incorrect_indices = (targets!=predictions).nonzero(as_tuple=True)[0]
            correct_count += len(correct_indices) #correct += (predicted == targets).sum().item()
            for target, prediction in zip(targets, predictions):
                if target == prediction:
                    correct_count_per_class[helpers_CNN.number_to_name(class_names, target)] += 1
                total_count_per_class[helpers_CNN.number_to_name(class_names, target)] +=1
            if sum(correct_count_per_class.values())!=correct_count:
                raise Exception('sum of correct counts per class differs from the number of correct counts.')
            # Remember best and worst probability with belonging item indices and images of each batch
            best_probs_each_batch, best_item_idcs_each_batch, best_inputs_each_batch, best_preds_each_batch = helpers_CNN_visualize.get_highest_probs_each_batch(best_probs_each_batch, best_item_idcs_each_batch, best_inputs_each_batch, best_preds_each_batch, correct_indices, probs, predictions, inputs, item_idcs)
            worst_probs_each_batch, worst_item_idcs_each_batch, worst_inputs_each_batch, worst_preds_each_batch = helpers_CNN_visualize.get_highest_probs_each_batch(worst_probs_each_batch, worst_item_idcs_each_batch, worst_inputs_each_batch, worst_preds_each_batch, incorrect_indices, probs, predictions, inputs, item_idcs)
            best_probs_each_batch_large_wood, best_item_idcs_each_batch_large_wood, best_inputs_each_batch_large_wood, best_preds_each_batch_large_wood = helpers_CNN_visualize.get_highest_probs_each_batch(best_probs_each_batch_large_wood, best_item_idcs_each_batch_large_wood, best_inputs_each_batch_large_wood, best_preds_each_batch_large_wood, correct_indices, probs, predictions, inputs, item_idcs, class_name='large wood', dataset=dataset)
            worst_probs_each_batch_large_wood, worst_item_idcs_each_batch_large_wood, worst_inputs_each_batch_large_wood, worst_preds_each_batch_large_wood = helpers_CNN_visualize.get_highest_probs_each_batch(worst_probs_each_batch_large_wood, worst_item_idcs_each_batch_large_wood, worst_inputs_each_batch_large_wood, worst_preds_each_batch_large_wood, incorrect_indices, probs, predictions, inputs, item_idcs, class_name='large wood', dataset=dataset)

            #ref_slides=[38,89,93,133,446,477,503,694]
            #get_pred_certain_refs(ref_slides, dataset, item_idcs, targets, predictions)
            #small_std=[343,344,345,346,347,348]
            #get_pred_certain_refs(small_std, dataset, item_idcs, targets, predictions)
            # idx_slides=[17,87,91,131,545,577,601,793]
            # for idx in idx_slides:
            #     if idx in item_idcs:
            #         j=item_idcs.index(idx)
            #         print(dataset.get_info(idx)["ref"])
            #         print(targets_str[j])
            #         print(targets[j])
            #         print(predictions[j])
            #         print(outputs_normed[j])
            #         print()

        # Print accuracy
        try:
            length_testloader = len(testloader.sampler.indices)
        except AttributeError as e:
            length_testloader = len(testloader.sampler)
        acc = correct_count / length_testloader
        if acc_per_fold is not None:
            writer.add_text("Accuracy on " + phase + " data in fold" + str(len(acc_per_fold)) + ":", str(acc), 0)
        else:
            writer.add_text("Accuracy on " + phase + " data:", str(acc), 0)
        if acc_per_fold is not None:
            acc_per_fold.append(acc)
        for i, (class_name, count) in enumerate(correct_count_per_class.items()):
            if total_count_per_class[class_name]!=0:
                acc_per_class = 100 * float(count) / total_count_per_class[class_name]
            elif total_count_per_class[class_name]==0 and count==0:
                acc_per_class = 100
            else:
                acc_per_class = None

            if acc_per_fold is not None:
                try:
                    writer.add_text(f'Accuracy per classes in fold {len(acc_per_fold)}', f'{class_name}: {acc_per_class:.1f} % ({total_count_per_class[class_name]} landscape elements at hand)', i)
                except TypeError as e:
                    writer.add_text(f'Accuracy per classes in fold {len(acc_per_fold)}', f'{class_name}: None ({total_count_per_class[class_name]} landscape elements at hand)', i)
            else:
                try:
                    writer.add_text(f'Accuracy per classes in {phase}', f'{class_name}: {acc_per_class:.1f} % ({total_count_per_class[class_name]} landscape elements at hand)', i)
                except TypeError as e:
                    writer.add_text(f'Accuracy per classes in {phase}', f'{class_name}: None ({total_count_per_class[class_name]} landscape elements at hand)', i)
        # Print images of three best and three worst predictions
        helpers_CNN_visualize.show_three_striking_probs(best_probs_each_batch, best_item_idcs_each_batch, best_inputs_each_batch, best_preds_each_batch, 'best', writer, dataset, class_names, phase)
        helpers_CNN_visualize.show_three_striking_probs(worst_probs_each_batch, worst_item_idcs_each_batch, worst_inputs_each_batch, worst_preds_each_batch, 'worst', writer, dataset, class_names, phase)
        helpers_CNN_visualize.show_three_striking_probs(best_probs_each_batch_large_wood, best_item_idcs_each_batch_large_wood, best_inputs_each_batch_large_wood, best_preds_each_batch_large_wood, 'best large wood', writer, dataset, class_names, phase)
        helpers_CNN_visualize.show_three_striking_probs(worst_probs_each_batch_large_wood, worst_item_idcs_each_batch_large_wood, worst_inputs_each_batch_large_wood, worst_preds_each_batch_large_wood, 'worst large wood', writer, dataset, class_names, phase)
        # Metrics
        conf_matrix = confusion_matrix(all_targets.cpu().numpy(), all_preds.cpu().numpy())
        f1_classes = f1_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average=None)
        f1_micro = f1_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='micro')
        f1_weighted = f1_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='weighted')
        precision_classes = precision_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average=None)
        precision_micro = precision_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='micro')
        precision_weighted = precision_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='weighted')
        recall_classes = recall_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average=None)
        recall_micro = recall_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='micro')
        recall_weighted = recall_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='weighted')
        #writer.add_text('Metrics', f'Confusion matrix:  \n {conf_matrix[0,:]}  \n {conf_matrix[1,:]}', 0)
        writer.add_text('Metrics', f'Confusion matrix:  \n {conf_matrix}', 0)
        writer.add_text('Metrics', f'F1_micro score: {f1_micro}', 1)
        writer.add_text('Metrics', f'F1_weighted score: {f1_weighted}', 2)
        writer.add_text('Metrics', f'Precision_micro score: {precision_micro}', 4)
        writer.add_text('Metrics', f'Precision_weighted score: {precision_weighted}', 5)
        writer.add_text('Metrics', f'Recall_micro score: {recall_micro}', 6)
        writer.add_text('Metrics', f'Recall_weighted score: {recall_weighted}', 7)
        writer.add_text('Metrics', f'F1 score: {f1_classes}', 8)
        writer.add_text('Metrics', f'Precision score: {precision_classes}', 9)
        writer.add_text('Metrics', f'Recall score: {recall_classes}', 10)
