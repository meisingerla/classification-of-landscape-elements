from __future__ import print_function, division
import helpers_testing
import helpers_CNN
import torch
import matplotlib.pyplot as plt
import helpers_CNN_visualize
import matplotlib.image as mpimg
import math

'''Program to receive the feature maps of the inputs'''


torch.manual_seed(42)

class_names = ['large wood', 'everything else']

# Load GoogLeNet model with trained fc layer
model_name = 'model-Jun27_21-51-50_iws-juck.pth'
model = helpers_testing.load_model(model_name)
if model.fc.out_features!=len(class_names):
    raise Exception('Model was trained on an other number of classes.')
model.eval()

# Load test dataset from Knoblochsaue
#test_dataset = roughness_sites.get_dataset_knoblochsaue(class_names, balanced=False)
test_dataset = helpers_CNN.get_mixed_dataset(class_names)

# Visualize feature maps
activation={}
def get_activation(name):
    def hook(model,input,output): #auch wenn sie hier nicht gebraucht werden, funktioniert Zeile mit output = model(data) sonst nicht
        activation[name]=output.detach()
    return hook

ref_slides=[38,89,93,133,446,477,503,694]
#idx_slides=[17,87,91,131,545,577,601,793]
# for i in range(len(test_dataset)):
#     if test_dataset.get_info(i)['ref']==133 and 'Fischpass' in test_dataset.get_info(i)["raster"].GetDescription():
#         example=test_dataset[i]
#         print(i) #= 131
example=test_dataset[131]

#conv1: alles was in conv1 steht wurde angewendet. min=0, max=10.2177
#conv1.conv: liefert feature map ohne veränderung (keine activation function). min=-16.1773, max=15.5591
#conv1.bn: gewisse Art von Nomierung, sodass alles >=0 (aber nicht einfach ReLU). min=0, max=10.2177
model.conv1.bn.register_forward_hook(get_activation('conv1')) #conv1.conv? scheint nach Anwendung von ReLU zu sein?
data, _, _ = example #size=[3,225,224]
plt.figure()
plt.imshow(helpers_CNN_visualize.imshow(data,test_dataset))
plt.axis('off')
plt.show()
data.unsqueeze_(0) #size=[1,3,224,224]
output = model(data) #das fügt was zu activation hinzu!
act = activation['conv1'].squeeze()
fig, ax = plt.subplots(nrows=math.ceil(math.sqrt(act.size(0))),ncols=math.ceil(math.sqrt(act.size(0))))
idx=0
for row in ax:
    for col in row:
        if idx<act.size(0):
            col.imshow(act[idx])
        col.axis('off')
        idx+=1
plt.show()

for i,feature_map in enumerate(act):
    # plt.figure()
    # plt.imshow(feature_map)
    # plt.axis('off')
    # plt.show()
    save_path = '../feature_maps_conv1bn/' + str(i) + '.png'
    mpimg.imsave(save_path, feature_map)

#alternativ:
#conv1: min=0, also ReLU wurde angewendet, conv1 ganz durchgespielt. max=20.4869
#conv1.conv: min=-35.8430, max=34.3988
#conv1.bn geht gar nicht
#es kann nicht beliebiges Layer abgefragt werden, müsste alles durchlaufen. Auch möglich zu programmieren, aber obige Lösung funktioniert ja
# layer = model.conv1
# image,_,_ = example
# feature_map = layer(image)
# feature_map.shape
# fig, ax = plt.subplots(nrows=8,ncols=8)
# idx=0
# for row in ax:
#     for col in row:
#         col.imshow(act[idx])
#         col.axis('off')
#         idx+=1
# plt.show()