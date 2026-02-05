#%%
import sys
sys.path.append('./scripts')
from DataYatesV1 import enable_autoreload, get_free_device
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

#enable_autoreload()
device = get_free_device()

from models.config_loader import load_dataset_configs

dataset_configs_path = "/home/declan/VisionCore/experiments/dataset_configs/multi_basic_120_long_rowley.yaml"
    
dataset_configs = load_dataset_configs(dataset_configs_path)

print(dataset_configs)

#%%
from models.data import prepare_data
data = {}
for i in range(len(dataset_configs)):
    dataset_configs[i]['types'] = ['gaborium']
    session_name = dataset_configs[i]['session']
    data[session_name] = prepare_data(dataset_configs[i], strict=False)

#%%

samples_per_neuron = {}
for session_name in data.keys():
    print(f"Dataset: {session_name}")
    unit_samples = data[session_name][0].dsets[0]['dfs'].sum(axis=0)
    samples_per_neuron[session_name] = unit_samples


# Plot a histogram of the number of valid samples per neuron for each dataset
import numpy as np
max_samples = max([max(samples) for samples in samples_per_neuron.values()])
bins = np.linspace(0, max_samples, 50)
plt.figure(figsize=(10, 6))
for session_name, unit_samples in samples_per_neuron.items():
    plt.hist(unit_samples, bins=bins, alpha=0.7, label=session_name)
plt.xlabel('Number of Valid Samples per Neuron')
plt.ylabel('Number of Neurons')
plt.title('Distribution of Valid Samples per Unit in Gaborium Datasets')
plt.legend()
plt.show()

#%%
# print number of units for each dataset
for session_name, unit_samples in samples_per_neuron.items():
    num_units = len(unit_samples)
    print(f"Dataset: {session_name}, Number of Units: {num_units}")
#%%
print(len([560, 564, 588, 591, 595, 596, 597, 600, 617, 622, 623, 624, 625, 640, 645, 648, 650, 652, 658, 661, 663, 671, 678, 684, 685, 686, 691, 695, 696, 702, 703, 704, 707, 714, 715, 718, 719, 722, 725, 727, 730, 732, 734, 736, 737, 740, 741, 744, 745, 746, 747, 749, 750, 751, 752, 757, 759, 761, 763, 764, 765, 766, 767, 771, 773, 775, 779, 783, 784, 786, 788, 789, 792, 797, 798, 800, 801, 802, 805, 806, 807, 810, 815, 816, 818, 820, 823, 825, 828, 831, 835, 840, 841, 844, 846, 847, 848, 849, 850, 852, 853, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 867, 869, 870, 871, 873, 874, 875, 878, 880, 882, 883, 884, 886, 887, 888, 889, 891, 894, 896, 901, 902, 906, 908, 909, 910, 911, 912, 913, 914, 915, 917, 920, 923, 924, 925, 929, 931, 933, 935, 936, 939, 940, 943, 946, 950, 959, 963]))
