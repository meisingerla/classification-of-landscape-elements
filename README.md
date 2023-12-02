# Code for the classification of river-floodplains landscape elements
Algorithmic implementation of the Bachelor's Thesis "Geospatial Mapping of River-Floodplain Landscape Elements from Aerial Imagery using Pretrained Convolutional Neural Networks".

The algorithm trains the top layer of the CNN GoogLeNet with labeled aerial images from a river-floodplain (e.g., Knoblochsaue Rhine) to classify a pictured landscape element into one of the categories large wood, sand, wet, shotrock, coarse-gravel, grass, fine-gravel, riprap, rock, mound, shadow, tree, manhole cover round, manhole cover angular, plant and water-plant. The model can be applied to aerial images of other river-floodplains (e.g., fish bypass Inn) to investigate the landscape elements occurring there.

Results can be reproduced by following the instructions in main.py<br>
All other files are either used in the above files or for visualization purposes.

Used folder structure for data intentionally not uploaded in github:<br>
Sister folder named '**Luftbilder-qgis**' containing the folders:
- 'Rhein_Knoblochsaue'
- 'Fischpass_4962_Frauenstein_alles'
- 'Fischpass_4962_Frauenstein_0202'

which contain the respectively vectors and rasters.<br>
Path directories can be customised in paths.py.

For necessary installations, see requirements.txt.