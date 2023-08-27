import matplotlib.image as mpimg
from classes_data_preprocessing import GISProject
from classes_data_preprocessing import FloodplainLandscapeElements
import paths
#import helpers_testing

'''Saves the images containing a landscape element'''

rasters_fishbypass =  [paths.raster1_fish_bypass]
vectors_fish_bypass = [paths.vector1_fish_bypass]

fish_bypass_complete = GISProject(rasters_fishbypass, vectors_fish_bypass)
fish_bypass_complete_dataset = FloodplainLandscapeElements(fish_bypass_complete, 'no large wood')

rasters_knobloch = [paths.raster2_knobloch, paths.raster1_knobloch]
vectors_knobloch = [paths.vector2_knobloch, paths.vector3_knobloch]

knobloch_project = GISProject(rasters_knobloch, vectors_knobloch)
knobloch_images = FloodplainLandscapeElements(knobloch_project, ['no large wood'])
# for i in range(1):
#     plt.figure()
#     plt.imshow(fish_bypass_complete_dataset[i][0])
#     save_path='../images_large_wood/' + str(5) + '.jpg'
#     mpimg.imsave(save_path, fish_bypass_complete_dataset[i][0])
#     print(type(fish_bypass_complete_dataset[i][0]))
#     pp.pprint(fish_bypass_complete_dataset.get_info(i))
#     plt.show()

# images of landscape elements:
# for landscape_element in fish_bypass_complete_dataset:
#     image = landscape_element[0]
#     idx = landscape_element[2]
#     ref = fish_bypass_complete_dataset.get_info(idx)['ref']
#     class_name = fish_bypass_complete_dataset.get_info(idx)['class']
#     label = fish_bypass_complete_dataset.get_info(idx)['label']
#     save_path = '../images_' + class_name.replace(' ', '_') + '/' + str(ref) + '_' + label.replace(' ', '_') + '.jpg'
#     mpimg.imsave(save_path, image)
#     print()

refs = []
class_names = []
for landscape_element in knobloch_images:
    image = landscape_element[0]
    idx = landscape_element[2]
    ref = knobloch_images.get_info(idx)['ref']
    class_name = knobloch_images.get_info(idx)['class']
    label = knobloch_images.get_info(idx)['label']
    if ref in refs:
        save_path = '../knobloch_images_' + class_name.replace(' ', '_') + '/' + str(ref) + '_' + str(refs.count(ref)) + '_' + label.replace(' ', '_') + '.jpg'
    else:
        save_path = '../knobloch_images_' + class_name.replace(' ', '_') + '/' + str(ref) + '_' + label.replace(' ', '_') + '.jpg'
    mpimg.imsave(save_path, image)
    refs.append(ref)
    class_names.append(class_name)

# Normalized image of landscape element with ref=133
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
# class_names = ['large_wood', 'no large_wood']
# training_dataset = roughness_sites.get_dataset_fish_bypass(class_names)
# for landscape_element in training_dataset:
#     idx = landscape_element[2]
#     ref = training_dataset.get_info(idx)['ref']
#     if ref==133:
#         image = landscape_element[0]
#         writer.add_image('133 normalized', image, 0)
#         class_name = fish_bypass_complete_dataset.get_info(idx)['class']
#         #save_path = '../images_' + class_name.replace(' ', '_') + '/' + str(ref) + 'normalized.jpg'
#         #mpimg.imsave(save_path, image)
 