from __future__ import print_function, division
import helpers_testing
import helpers_CNN
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

'''Program to test model on data from Rhein_Knoblochsaue'''

# Initialize tensorboard writer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

torch.manual_seed(42)

class_names = ['large wood', 'everything else']
writer.add_text('class names', str(class_names))

# Load GoogLeNet model with trained fc layer
model_name = 'model-Jun27_21-51-50_iws-juck.pth'
model = helpers_testing.load_model(model_name)
if model.fc.out_features!=len(class_names):
    raise Exception('Model was trained on an other number of classes.')

writer.add_text('Test with data from knoblochsaue on model', model_name, 0)

# Load test dataset from Knoblochsaue
test_dataset = helpers_CNN.get_mixed_dataset(class_names)
testloader = DataLoader(test_dataset, shuffle=False)

# Evaluation
helpers_testing.test_final_model(testloader, class_names, model, writer, 'test')

writer.close()