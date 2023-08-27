import use_cases
import roughness_sites

'''Algorithmic implementation of the Bachelor's Thesis
"Geospatial Mapping of River-Floodplain Landscape Elements
from Aerial Imagery using Pretrained Convolutional Neural Networks".

For instructions on which files to include and how, see README.md.
For necessary installations, see requirements.txt.

The options of use_case are (with corresponding result section):
'cv separated training dataset' (4.1 Hyperparameter tuning)
'separated training dataset'    (4.2 Model efficiency for training on one study site)
'cv mixed training dataset'     (4.3 Model efficiency for training on both study sites)
'mixed training dataset'        (4.3 Model efficiency for training on both study sites)
'vary number of data points'    (4.4 Varying number of data points)
'vary number of classes 2'      (4.5 Varying number of no large wood classes)
'vary number of classes 3'      (4.5 Varying number of no large wood classes)
'vary number of classes 4'      (4.5 Varying number of no large wood classes)
'vary number of classes 5'      (4.5 Varying number of no large wood classes)
'vary number of classes 6'      (4.5 Varying number of no large wood classes)
'vary number of classes 7'      (4.5 Varying number of no large wood classes)
'roughness fish bypass'         (4.6 Examination of the landscape environment of large wood)
'roughness knoblochsaue'        (4.6 Examination of the landscape environment of large wood)

The results are recorded on TensorToard. TensorBoard can be called in VSCode as follows:
1. Enter the key combination Strg + shift + P
2. Select "Python: Launch TensorBoard" in the appearing window
3. When TensorBoard was not used in VSCode before: Follow Python recommendations
(something like enter "conda install -c conda-forge --name bachelor tensorboard -y" and "pip install torch-tb-profiler" in Terminal)
4. If Tensorboard does not load: enter "lsof -i:6006" in Terminal, this shows PID and user,
then enter "kill <PID>" in TensorBoard (if you are the corresponding user) and repeat step 1 and 2.
5. TensorBoard appears in VSCode window or can be called in the browser via "http:\\<Local Address>\#" (<Local Address> is listed under Ports in VSCode).
'''

use_case = 'separated training dataset'

match use_case:
    case 'cv separated training dataset':
        use_cases.cv_separated_training()
    case 'separated training dataset':
        use_cases.test_separated_training()
    case 'cv mixed training dataset':
        use_cases.cv_mixed_training()
    case 'mixed training dataset':
        use_cases.test_mixed_training()
    case 'vary number of data points':
        use_cases.test_influence_number_of_data_points(0)
        use_cases.test_influence_number_of_data_points(10)
        use_cases.test_influence_number_of_data_points(50)
        use_cases.test_influence_number_of_data_points(100) #49 no large wood, 51 large wood
        use_cases.test_influence_number_of_data_points(300) #153 no large wood, 147 large wood
        use_cases.test_influence_number_of_data_points(687)
    case 'vary number of classes 2':
        use_cases.test_influence_number_of_classes(['large wood', 'everything else'])
    case 'vary number of classes 3':
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'wet', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'shotrock', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'coarse-gravel', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'grass', 'everything else'])
    case 'vary number of classes 4':
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'wet', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'shotrock', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'coarse-gravel', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'wet', 'shotrock', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'wet', 'coarse-gravel', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'wet', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'shotrock', 'coarse-gravel', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'shotrock', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'coarse-gravel', 'grass', 'everything else'])
    case 'vary number of classes 5':
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'wet', 'shotrock', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'wet', 'coarse-gravel', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'wet', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'wet', 'shotrock', 'coarse-gravel', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'wet', 'shotrock', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'wet', 'coarse-gravel', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'shotrock', 'coarse-gravel', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'shotrock', 'coarse-gravel', 'sand', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'shotrock', 'grass', 'sand', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'coarse-gravel', 'grass', 'sand', 'everything else'])
    case 'vary number of classes 6':
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'wet', 'shotrock', 'coarse-gravel', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'wet', 'shotrock', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'wet', 'coarse-gravel', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'shotrock', 'coarse-gravel', 'grass', 'everything else'])
        use_cases.test_influence_number_of_classes(['large wood', 'wet', 'shotrock', 'coarse-gravel', 'grass', 'everything else'])
    case 'vary number of classes 7':
        use_cases.test_influence_number_of_classes(['large wood', 'sand', 'wet', 'shotrock', 'coarse-gravel', 'grass', 'everything else'])
    case 'roughness fish bypass':
        roughness_sites.evaluate_roughness_lw('fish bypass')
    case 'roughness knoblochsaue':
        roughness_sites.evaluate_roughness_lw('knoblochsaue')
    case _:
        raise Exception('use_case does not match')

