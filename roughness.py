from __future__ import print_function, division
import helpers_CNN
import helpers_testing
import roughness_sites
import torch
from torch.utils.data import DataLoader

'''Assumption: Model is more likely to spot large wood, if the landscape environment is rough/uneven.
Check assumption by comparing roughness of right/wrong classified images of large wood/no large wood.
Roughness is measured by standard deviation.'''
'''Approch: Classify images by their class (large wood or no large wood) und whether they are predicted correct or not.
On each image set compute mean and standard deviation of each rgb channel.'''
'''Result: Images, which are predicted as large wood have a higher standard deviation and smaller mean than images predicted as no large wood.
Images, which are falsely predicted as large wood also have a high std and small mean.
-> Model tends to classify dark and rough images as large wood.
'''

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = ['large wood', 'no large wood']

# load model
#model_name = 'model-Jun07_08-23-56_iws-juck.pth' #für 3 classes
#model_name = 'model-May24_22-19-21_iws-juck.pth'
model_name = 'model-Jun27_21-51-50_iws-juck.pth' #2 classes, mixed trainingset
#model_path = '../models/' + model_name
model = helpers_testing.load_model(model_name)

# load data
fish_bypass_dataset = roughness_sites.get_dataset_fish_bypass(class_names)
#knobloch_dataset = roughness_sites.get_dataset_knoblochsaue(class_names, balanced=False)
#mixed_dataset = helpers_CNN.get_mixed_dataset(class_names)
#training_dataset.transform=transforms.ToTensor() so werden alle Bilder falsch klassifiziert

testloader = DataLoader(fish_bypass_dataset)
#testloader = DataLoader(knobloch_dataset)
#testloader = DataLoader(mixed_dataset)

# Set the mode to evaluation (e.g. dropout is deactivated)
model.eval()
with torch.no_grad():
    possibilities = ['correct large_wood', 'incorrect large_wood', 'correct no_large_wood', 'incorrect no_large_wood']
    rgb_sum = {x: torch.Tensor([0.0, 0.0, 0.0]).to(device) for x in possibilities}
    rgb_sum_sq = {x: torch.Tensor([0.0, 0.0, 0.0]).to(device) for x in possibilities}
    count = {x: 0 for x in possibilities}
    # Iterate over the test data and generate predictions
    for i, data in enumerate(testloader):
        # Get inputs
        inputs, targets_str, item_idcs = data
        inputs = inputs.to(device)
        targets = helpers_CNN.names_to_numbers(class_names, targets_str)
        targets = targets.to(device)

        # Generate outputs
        outputs = model(inputs)

        # Set total and correct
        probs, predicted = torch.max(outputs.data, 1) #outputs und outputs.data ist eig gleich, siehe Erklärung oben
        
        # Accuracy statistics
        large_wood_tensor = torch.full((targets.shape[0],), class_names.index('large wood')).to(device)
        large_wood_or_not = {'large_wood': (targets==large_wood_tensor),
                           'no_large_wood': (targets!=large_wood_tensor)}
        correct_pred_or_not = {'correct': (targets==predicted),
                          'incorrect': (targets!=predicted)}
        
        for correct_pred_or_not_str, correct_pred_or_not_bool in correct_pred_or_not.items():
            for large_wood_or_not_str, large_wood_or_not_bool in large_wood_or_not.items():
                indices = torch.logical_and(correct_pred_or_not_bool, large_wood_or_not_bool).nonzero(as_tuple=True)[0]
                key = correct_pred_or_not_str + ' ' + large_wood_or_not_str
                for index in indices:
                    image = inputs[index] #image ist normiert, daher nicht in 0 bis 1
                    rgb_sum[key] += image.sum(axis=[1,2])
                    rgb_sum_sq[key] += (image**2).sum(axis=[1,2])
                    count[key] += 1
    total_mean = {x: torch.Tensor([0.0, 0.0, 0.0]).to(device) for x in possibilities}
    total_var = {x: torch.Tensor([0.0, 0.0, 0.0]).to(device) for x in possibilities}
    total_std = {x: torch.Tensor([0.0, 0.0, 0.0]).to(device) for x in possibilities}
    for key in possibilities:
        pixels = count[key]*224*224
        total_mean[key] = rgb_sum[key] / pixels
        print(key, ': mean =', total_mean[key])
        total_var[key]  = (rgb_sum_sq[key] / pixels) - (total_mean[key] ** 2)
        total_std[key]  = torch.sqrt(total_var[key])
        print(key, ': std =', total_std[key])

    # Has large wood really a higher std?
    pixels_large_wood = (count['correct large_wood']+count['incorrect large_wood'])*224*224
    mean_large_wood = (rgb_sum['correct large_wood']+rgb_sum['incorrect large_wood'])/pixels_large_wood
    print('mean_large_wood', mean_large_wood)
    var_large_wood = (rgb_sum_sq['correct large_wood']+rgb_sum_sq['incorrect large_wood'])/pixels_large_wood - (mean_large_wood**2)
    std_large_wood = torch.sqrt(var_large_wood)
    print('std_large_wood', std_large_wood)

    # Kontrolle der no_large_wood Daten
    pixels_no_large_wood = (count['correct no_large_wood']+count['incorrect no_large_wood'])*224*224
    mean_no_large_wood = (rgb_sum['correct no_large_wood']+rgb_sum['incorrect no_large_wood'])/pixels_no_large_wood
    print('mean_no_large_wood', mean_no_large_wood)
    var_no_large_wood = (rgb_sum_sq['correct no_large_wood']+rgb_sum_sq['incorrect no_large_wood'])/pixels_no_large_wood - (mean_no_large_wood**2)
    std_no_large_wood = torch.sqrt(var_no_large_wood)
    print('std_no_large_wood', std_no_large_wood)

    # Test ob Daten normalisiert sind:
    overall_pixels = len(testloader.sampler)*224*224
    overall_mean = sum(rgb_sum.values()) / overall_pixels #=[0,0,0] passt
    overall_var = sum(rgb_sum_sq.values()) / overall_pixels - (overall_mean**2)
    overall_std = torch.sqrt(overall_var) #=[1,1,1] passt
