import numpy as np
import os
from functools import lru_cache

import os
import numpy as np
import sys
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from scipy.optimize import curve_fit
import torch

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import RSSLocDataset
from CELF_POWDERdata import *
from tirem_helper import load_tirem_alt_params, load_tirem_prop_map
from models import TiremMLP
from locconfig import LocConfig

import numpy as np
import tirem_params
import torch
import pickle
import argparse
import os
from localization_bspline2_justTest import DLLocalization
from locconfig import LocConfig
from dataset2 import RSSLocDataset
from models_bspline2 import CoMLoss, SlicedEarthMoversDistance
from attacker import batch_wrapper, get_all_attack_preds_without_grad
from synthetic_augmentations2 import SyntheticAugmentor

frs_dsm_corners = np.array([
    [1447.88347094,  492.35956159],
    [3847.88347094, 2712.35956159]
])

device_lookup_dict = {
    'cbrssdr1-bes-comp': 'Behavioral',
    'bookstore-nuc': 'Bookstore',
    'cbrssdr1-browning-comp': 'Browning',
    #'': 'Dentistry',
    'ebc-nuc': 'EBC',
    'cnode-ebc-dd-b210': 'EBC-DD',
    'cbrssdr1-fm-comp': 'Friendship',
    'garage-nuc': 'Garage',
    'cnode-guesthouse-dd-b210': 'GH-DD',
    'guesthouse-nuc': 'Guesthouse',
    'cbrssdr1-honors-comp': 'Honors',
    'cbrssdr1-hospital-comp': 'Hospital',
    'humanities-nuc': 'Humanities',
    'law73-nuc': 'Law73',
    'madsen-nuc': 'Madsen',
    'cnode-mario-dd-b210': 'Mario-DD',
    'cbrssdr1-smt-comp': 'Medical_Tower',
    'moran-nuc': 'Moran',
    'cnode-moran-dd-b210': 'Moran-DD',
    'sagepoint-nuc': 'Sagepoint',
    'cbrssdr1-ustar-comp': 'USTAR',
    'cnode-ustar-dd-b210': 'USTAR-DD',
    'cnode-wasatch-dd-b210': 'Wasatch-DD',
    'web-nuc': 'WEB',
}

param_keys = [
    '_diff_parameter',
    '_diff_param_obstacle_with_max_h_over_r',
    '_diffraction_angles',
    '_elevation_angles_NLOS',
    '_elevation_angles_tx_rx',
    '_LOS_NLOS',
    '_number_knife_edges',
    '_number_obstacles',
    '_shadowing_angles',
]

should_train = True #making training true
should_load_model = True #making load model true
restart_optimizer = False #don't restart optimizer
one_tx = True #one transmitter only

triangulation_enable = True #making load model true

# Specify params
max_num_epochs = 1000 #number of epochs in training process
# max_num_epochs = 10
include_elevation_map = True #include elevation map in training

batch_size = 32 if should_train else 32 #batch size 64

num_training_repeats = 1 #traning repeat one time

device = torch.device('cuda') #using gpu
all_results = {} #dictionary to store all results

def get_rx_data_by_tx_location(rldataset, source_key=None, combine_sensors=True, use_db_for_combination=True, required_limit=100):
    # rldataset = self.rldataset
    if source_key is None and hasattr(rldataset, 'train_key'):
        source_key = rldataset.train_key
    source_data = rldataset.data[source_key]
    # print(source_key)
    data = {}
    train_x, train_y = source_data.ordered_dataloader.dataset.tensors[:2]
    train_x, train_y = train_x.cpu().numpy(), train_y.cpu().numpy()
    # print("train_x shape = ",train_x[0], " train_y shape = ",train_y[0])
    coords = train_y[:,0,1:] #it just keeps the coordinates of the one transmitter
    for key in rldataset.location_index_dict:
        name = rldataset.location_index_dict[key]
        if isinstance(name, str):
            if 'bus' in name: continue
            if combine_sensors:
                if 'nuc' in name and 'b210' in name:
                    name = name[:-6]
                elif 'cellsdr' in name:
                    name = name.replace('cell','cbrs')
        #If bus, then dont even go further
        rss = train_x[:,key,0] #just take received rss for this particular key 
        valid_inds = rss != 0 #take the valid and invalid indices
        if valid_inds.sum() < required_limit or valid_inds.sum() == 0: continue
        sensor_loc = train_x[valid_inds][0,key,1:3] #Taking the location of this particular sensor
        sensor_type = train_x[valid_inds][0,key,4] #Taking sensor cateogry
        rss = rss[valid_inds]# taking just the valid rss values
        if combine_sensors and use_db_for_combination: #here actually converting back all the normalized rss value to their true value
            min_rss, max_rss = rldataset.get_min_max_rss_from_key(key)
            rss = rss * (max_rss - min_rss) + min_rss
        few_coords = coords[valid_inds] #taking the coordinates of the respective samples

        if name not in data:
            data[name] = [few_coords, rss, sensor_loc, [key], sensor_type]
        else:
            data[name][3].append(key)
            data[name] = [
                np.concatenate((data[name][0], few_coords)),
                np.concatenate((data[name][1], rss)),
                sensor_loc,
                data[name][3],
                sensor_type
                ]
        #all_trainsmitter coords, all_rss, the receiver location, the receiver key, the sensor category
    for name in data:
        few_coords, rss, sensor_loc, keys, sensor_type = data[name]
        #Each position can have two keys such as madsen-nuc2-b210 has 38 and madsen-nuc1-b210 has 2
        sorted_inds = np.lexsort(few_coords.T)
        few_coords = few_coords[sorted_inds]
        rss = rss[sorted_inds]
        if combine_sensors and use_db_for_combination:
            min_rss = min(rldataset.get_min_max_rss_from_key(key)[0] for key in keys)
            max_rss = min(rldataset.get_min_max_rss_from_key(key)[1] for key in keys)
            rss = (rss - min_rss) / (max_rss - min_rss)

        #So, in the previous loop, they gathered all the sensors of same location
        #Now, renormalizing it with min rss and max rss of both
        # print(abcd)
        row_mask = np.append([True], np.any(np.diff(few_coords,axis=0),1))
        #It is taking difference between consecutive rows
        #so that can take transmitter of unique indices
        #adding true in front to make $n-1$ samples to $n$ samples
        data[name][0] = few_coords[row_mask]
        data[name][1] = rss[row_mask]
        #just adding trimmed version for one position just once
        #so, data[name] will basically have transmitter coords, all rss, receiver position, receiver key and sensor category
    return data

def build_dictionary(rldataset, name, sensor_loc, keys, sensor_type):
    dataset_index = rldataset.params.dataset_index
    # name = 'madsen-nuc'
    corners = tuple(rldataset.corners.flatten())
    corners = np.array(corners).reshape(2,2)

    file_suffix = '_tirem_rssi_462.7.npz'
    npz_name = os.path.join(
        '/scratch/general/vast/u1472438/frs',
        device_lookup_dict[name]+file_suffix
    )
    downsample_ratio = 2
    bounds = np.zeros(2)
    bounds = bounds.astype(int)
    """print("Corners = ", corners, " corner 1-0 = ", corners[1] - corners[0])
        [[1452.88347094  497.35956159]
    [3852.88347094 2697.35956159]]  corner 1-0 =  [2400. 2200.]
    """
    size = (corners[1] - corners[0]).astype(int)*downsample_ratio + 10
    rssi_file = np.load(npz_name, allow_pickle=True)
    #['arr_0' = 'tirem_rssi, 'arr_1'="distances", 'arr_2'='parameters']
    # print("Tirem RSSI shape = ",rssi_file['arr_0'].shape) (4441, 4801)
    # print("Distances shape = ",rssi_file['arr_1'].shape) (4441, 4801)
    """print("Parameters shape = ",rssi_file['arr_2']) #This is the object of tirem parameters
    BS ENDPOINT NAME: Madsen
    BS_IS_TX: 0
    TX HEIGHT: 1.5
    RX HEIGHT: 1.5
    BS LONGITUDE: -111.84167
    BS LATITUDE: 40.76895
    BS X: 3021
    BS Y: 855
    FREQUENCY: 462.7
    POLARIZATION: 462.7
    GENERATE FEATURES: False
    MAP TYPE: corrected
    MAP FILEDIR: corrected_dsm.tif
    GAIN: 0
    FIRST CALL: 0
    EXTENSION: 0
    REFRACTIVITY: 450
    CONDUCTIVITY:50
    PERMITTIVITY: 81
    HUMIDITY: 80
    SIDE LENGTH: 0.5
    SAMPLING INTERVAL: 0.5          
    """
    # print("Distances = ",rssi_file['arr_1'][:5,:5]) distances from particular base station to all the positions in the grid
    tirem_prop_map_rssi = rssi_file['arr_0'][bounds[1]:bounds[1]+size[1]:1,bounds[0]:bounds[0]+size[0]:1]
    """print("RSSI Propagation Map shape = ", tirem_prop_map_rssi.shape) (2205, 2401)
    Here we took a half version of the true map"""
    # print("Meter Scale = ", rldataset.params.meter_scale)Meter Scale =  25
    # print("Map shape = ",rssi_file['arr_0'].shape," Size = ", size, "Downsample Ratio = ",downsample_ratio, " tirem_prop_map_rssi shape = ",tirem_prop_map_rssi.shape)
    # Map shape =  (4441, 4801)  Size =  [4810 4410] Downsample Ratio =  2  tirem_prop_map_rssi shape =  (4410, 4801)

    max_x = 97  # Maximum value for the 0th coordinate
    max_y = 89  # Maximum value for the 1st coordinate
    interval = 0.5  # Step size

    # Generate coordinate values
    x_values = np.arange(0, max_x + interval, interval)  # Including max_x
    y_values = np.arange(0, max_y + interval, interval)  # Including max_y

    # Create a meshgrid of coordinates
    X, Y = np.meshgrid(x_values, y_values, indexing='ij')

    # Stack into a (N, 2) shaped array
    few_coords = np.column_stack((X.ravel(), Y.ravel()))

    #few_coords is the coordinates in the pixels
    #rss is the normalized rss value
    """print("Few Coords = ", few_coords[:5])
    [[63.010017 17.135702]
    [63.069717 17.149967]
    [63.06989  17.150541]
    [63.06945  17.15059 ]
    [63.07976  17.165813]]
    """

    tirem_coords = np.rint(few_coords*rldataset.params.meter_scale*downsample_ratio).astype(int)
    """print("Tirem Coords = ", tirem_coords[:5])
    [[1575  428]
    [1577  429]
    [1577  429]
    [1577  429]
    [1577  429]]"""
    # print("Tirem Coordinates shape = ", tirem_coords.shape) (3022, 2)

    # print("Few Coords : first = ", np.max(few_coords[:,0]), " and second = ", np.max(few_coords[:,1])) first =  91.54554  and second =  86.31162
    #Image height: 89 97
    # print("Tirem Coords = ", tirem_coords[:15])
    # print(rldataset.Samples.rectangle_width, rldataset.Samples.rectangle_height)


    shape = tirem_prop_map_rssi.shape
    # print("Tirem Prop map shape = ", shape)  (2205, 2401)

    tirem_db = tirem_prop_map_rssi[np.minimum(tirem_coords[:,1], shape[0]-1), np.minimum(tirem_coords[:,0], shape[1]-1)]
    # print(np.minimum(tirem_coords[:,1], shape[0]-1)) [ 428  429  429 ... 2110 2142 2158]
    # print(np.minimum(tirem_coords[:,0], shape[1]-1)) [1575 1577 1577 ... 1117 1581 1568]
    # print("Tirem DB shape = ", tirem_db.shape) (3022,)
    """print("tirem db = ",tirem_db[:10]) [-87.79444885 -88.89899445 -88.89899445 -88.89899445 -88.89899445
    -85.96736908 -81.84765625 -86.17285156 -87.5594635  -84.60083008]"""
    
    min_rss, max_rss = rldataset.get_min_max_rss_from_key(keys[0])
    tirem_normalized = (tirem_db - min_rss) / (max_rss - min_rss)
    """print(tirem_normalized[:10])[0.07918188 0.06704402 0.06704402 0.06704402 0.06704402 0.09925968
    0.14453125 0.09700163 0.08176414 0.11427659]"""
    
    flatten_corners = tuple(rldataset.corners.flatten())

    corners = np.array(flatten_corners).reshape(2,2)

    folder = '/scratch/general/vast/u1472438/frs'
    file_suffix = '_462.7.npy'
    downsample_ratio = 2
    name_suffix = ''
    bounds = np.zeros(2)

    bounds = bounds.astype(int)
    device_name = device_lookup_dict[name] if name in device_lookup_dict else name
    size = (corners[1] - corners[0]).astype(int)*downsample_ratio + 10

    npz_file = os.path.join(
        folder,
        device_name+name_suffix+'_tirem_rssi'+file_suffix[:-1]+'z'
    )
    f = np.log(np.load(npz_file)['arr_1'][bounds[1]:bounds[1]+size[1]:1,bounds[0]:bounds[0]+size[0]:1] + 1) #(2205, 2401)
    zero_coords = np.argwhere(np.load(npz_file)['arr_1'][bounds[1]:bounds[1]+size[1]:1,bounds[0]:bounds[0]+size[0]:1] == 0)
    zero_coords = zero_coords[:, [1, 0]]
    print(name," and coordinate = ",zero_coords/(rldataset.params.meter_scale*downsample_ratio))
    #The +1 in f is avoiding log(0) case.
    # print("Original: ", np.load(npz_file)['arr_1'][bounds[1]:bounds[1]+size[1]:downsample_ratio,bounds[0]:bounds[0]+size[0]:downsample_ratio][:15])
    # print("Processed: ", f[:15])
    # print("Map shape = ",rssi_file['arr_1'].shape," Size = ", size, "Downsample Ratio = ",downsample_ratio, " f shape = ",f.shape)
    # Map shape =  (4441, 4801)  Size =  [4810 4410] Downsample Ratio =  2  f shape =  (4410, 4801)
    param_results = np.zeros((f.shape[0], f.shape[1], 14), dtype=np.float32)
    param_results[:,:,0] = f #Storing the distances
    current_ind = 1

    # print("Distance: min value = ", np.min(param_results[:,:,0]), " max value = ",np.max(param_results[:,:,0]))
    for param in param_keys:
        npy_file = os.path.join(
            folder,
            device_name+name_suffix+param+file_suffix
        )
        # print(npy_file)
        f = np.load(npy_file)[bounds[1]:bounds[1]+size[1]:1,bounds[0]:bounds[0]+size[0]:1]
        # print("Map shape = ",np.load(npy_file).shape," Size = ", size, "Downsample Ratio = ",downsample_ratio, " f shape = ",f.shape)
        # Map shape =  (4441, 4801, 2)  Size =  [4810 4410] Downsample Ratio =  2  f shape =  (4410, 4801, 2)
        if len(f.shape) == 2:
            # print("At index ", current_ind)
            param_results[:,:,current_ind] = f
            # print(param," :", " min value = ", np.min(param_results[:,:,current_ind]), " max value = ",np.max(param_results[:,:,current_ind]))
            current_ind += 1
        else:
            #Two Layers
            #_diff_parameter_
            #_diffraction_angles_
            #_elevation_angles_NLOS_
            #_shadowing_angles_
            param_results[:,:,current_ind:current_ind + f.shape[-1]] = f
            # print("At Index = ", current_ind," - ", current_ind+f.shape[-1])
            # print(param,"1 :", " min value = ", np.min(param_results[:,:,current_ind]), " max value = ",np.max(param_results[:,:,current_ind]))
            # print(param,"2 :", " min value = ", np.min(param_results[:,:,current_ind+1]), " max value = ",np.max(param_results[:,:,current_ind+1]))
            current_ind += f.shape[-1]   
        """The serial of storage
        'distance' min value =  0.0  max value =  7.7548738
        '_diff_parameter',2, min value =  0.0  max value =  13.822624, min value =  0.0  max value =  102.83729
        '_diff_param_obstacle_with_max_h_over_r',min value =  0.0  max value =  128.30615
        '_diffraction_angles',2, min value =  0.0  max value =  147.24057, min value =  0.0  max value =  147.24057
        '_elevation_angles_NLOS',2, min value =  0.0  max value =  160.96265, min value =  0.0  max value =  94.47321
        '_elevation_angles_tx_rx', min value =  0.0  max value =  166.07703
        '_LOS_NLOS',min value =  0.0  max value =  1.0
        '_number_knife_edges',min value =  0.0  max value =  18.0
        '_number_obstacles',min value =  0.0  max value =  44.0
        '_shadowing_angles',2, min value =  -98.96937  max value =  111.57743, min value =  0.0  max value =  77.59788    
        """
    # print("Diff parameter - 1 = ",param_results[:10,:10,1])
    # print("Diff parameter - 2 = ",param_results[:10,:10,2])
    scale_factors = np.array([
        8,20,180,180,
        180,180,180,180,
        180, 1, 20, 20,
        180,180
    ]).astype(float).reshape(1,1,-1)
    param_results[:] /= scale_factors

    tirem_params = param_results
    tirem_coords = np.rint(few_coords*rldataset.params.meter_scale*downsample_ratio).astype(int)
    shape = tirem_params.shape
    tirem_features = tirem_params[np.minimum(tirem_coords[:,1], shape[0]-1), np.minimum(tirem_coords[:,0], shape[1]-1)]
    # print("Tirem Features = ",tirem_features.shape, " tirem normalized = ",tirem_normalized.shape)
    tirem_features = np.concatenate((tirem_features, tirem_normalized[:,None]), axis=1)
    save_npz_file = f'/uufs/chpc.utah.edu/common/home/u1472438/dl-image-localization-main/tirem_features2/tirem_all_features_DS{rldataset.params.dataset_index}_{name}.npz'
    np.savez(save_npz_file, rss_db=tirem_db, features=tirem_features, coords=few_coords)

def get_tirem_features(rldataset, name, few_coords, keys):
    few_coords = few_coords.astype(np.float32)
    if os.path.exists(f'/uufs/chpc.utah.edu/common/home/u1472438/dl-image-localization-main/tirem_features2/tirem_all_features_DS{rldataset.params.dataset_index}_{name}.npz'):
        features_dict = np.load(f'/uufs/chpc.utah.edu/common/home/u1472438/dl-image-localization-main/tirem_features2/tirem_all_features_DS{rldataset.params.dataset_index}_{name}.npz')
        rss_db = features_dict['rss_db']
        library_coords = features_dict['coords'].astype(np.float32)
        db_dict = {coord.tobytes():rss_db[i] for i, coord in enumerate(library_coords)}
        tirem_db = []

        for coord in few_coords:
            key = coord.tobytes()
            if key in db_dict:
                tirem_db.append(db_dict[key]) 
            else:
                distances = np.linalg.norm(library_coords - coord, axis=1)
                closest_idx = np.argmin(distances)
                closest_coord = library_coords[closest_idx]
                closest_key = closest_coord.tobytes()

                if closest_key in db_dict:
                    # print(f"Using closest library coordinate: {closest_coord}")
                    tirem_db.append(db_dict[closest_key])  # Use closest coordinate's value
                else:
                    # print(f"Closest coordinate {closest_coord} also missing in db_dict, setting NaN")
                    tirem_db.append(np.nan)  # Use NaN if no valid replacement is found
        
        tirem_db = np.array(tirem_db)

        min_rss, max_rss = rldataset.get_min_max_rss_from_key(keys[0])
        tirem_normalized = (tirem_db - min_rss) / (max_rss - min_rss)

        features = features_dict['features']
        features_dict = {coord.tobytes():features[i] for i, coord in enumerate(library_coords)}
        tirem_features = []

        for coord in few_coords:
            key = coord.tobytes()
            if key in features_dict:
                tirem_features.append(features_dict[key])  # Use existing features
            else:
                distances = np.linalg.norm(library_coords - coord, axis=1)
                closest_idx = np.argmin(distances)
                closest_coord = library_coords[closest_idx]
                closest_key = closest_coord.tobytes()

                if closest_key in features_dict:
                    # print(f"For missing {coord}, using closest library coordinate: {closest_coord}")
                    tirem_features.append(features_dict[closest_key])  # Use closest features
                else:
                    tirem_features.append(np.full(features.shape[1], np.nan))  # Fill with NaNs

        tirem_features = np.array(tirem_features)
    input_features = torch.tensor(tirem_features, dtype=torch.float32, device=rldataset.params.device)
    tirem_tensor = torch.tensor(tirem_normalized, dtype=torch.float32, device=rldataset.params.device)

    return input_features, tirem_tensor, tirem_db


def main():
    global dataset_index #global variable of dataset index
    global meter_scale #global variable on how many meters in each pixel
    cmd_line_params = [] #here is the list for all command line parameters
    parser = argparse.ArgumentParser() #it will parse all the arguments given in the command line
    parser.add_argument("--param_selector", type=int, default=-1, help='Index of pair for selecting params') #parsing parameter selector value
    parser.add_argument("--random_ind", type=int, default=-1, help='Random Int for selecting set of params') #parsing random ind value
    args = parser.parse_args() #args taking all the arguments. 

    for random_state in range(0,num_training_repeats): #how many times repeat the training
        # for di in [6,7,8]:#which dataset to train on
        for di in [6]:
            # for split in ['random', 'grid2', 'grid5', 'grid10']: #the split in dataset
            for split in ['random']:
                # for loss_func in [ CoMLoss(), torch.nn.MSELoss(), SlicedEarthMoversDistance(num_projections=100, scaling=0.01, p=1)]:
                for loss_func in [SlicedEarthMoversDistance(num_projections=100, scaling=0.01, p=1)]: #which parameter to use
                    cmd_line_params.append([di, split, random_state, loss_func]) #adding 6, 'random', 0 and loss_func to the list
    
    if args.param_selector > -1:
        param_list = [cmd_line_params[args.param_selector] ]
    elif args.random_ind > -1:
        param_list = [param for param in cmd_line_params if param[3] == args.random_ind]
    else:
        param_list = cmd_line_params #if and elif will select specific combination whereas this will move forward with whatever it has. 
    
    for ind, param_set in enumerate(param_list):
        dataset_index, split, random_state, loss_func  = param_set
        print(param_set)
        dict_params = {
            "dataset": dataset_index,
            "data_split": split,
            "batch_size":batch_size,
            "random_state":random_state,
            "include_elevation_map":include_elevation_map,
            "one_tx": one_tx,
            "use_triangulation": triangulation_enable,
        } #making a dictionary of parameters to store dataset, split, batch_size, random_state and elevation map
        params = LocConfig(**dict_params) #pass to LocConfig, for initializing the configuration
        #This will have an object. Also a string if needed
        loss_label = 'com' if isinstance(loss_func, CoMLoss) else 'mse' if isinstance(loss_func, torch.nn.MSELoss) else 'emd' if isinstance(loss_func, SlicedEarthMoversDistance) else 'unknown'
        #'emd' as we just using the Earth Mover Distance
        param_string = f"{params}_{loss_label}"
        #parameter string_emd in our case
        PATH = 'models_justTest/%s__model.pt' % param_string
        #save in models folder with param_string__model.pt extension
        model_ending = 'train_val.'
        #model ending contains train_val
        global all_results
        #declaring all_results variable for no reason at all
        pickle_filename = 'results_bspline/augment_%s.pkl' % param_string
        #another pickle file for storing results
        model_filename = PATH.replace('model.', 'model_' + model_ending)
        #path replace with new model name
        if not os.path.isdir(os.path.dirname(PATH)):
            os.makedirs(os.path.dirname(PATH), exist_ok = True)
        #if not any such path exists, than create one path

        rldataset = RSSLocDataset(params)
        train_data_key = rldataset.train_key
        test_data_key1 = rldataset.test_keys[0]
        test_data_key2 = rldataset.test_keys[1]
        rldataset.print_dataset_stats()
        #it will print all the stats

        rldataset.make_elevation_tensors(meter_scale=5)
        print("Dataset elevation map = ", rldataset.elevation_tensors.shape) #torch.Size([1, 440, 480])
        print("Dataset Building map = ", rldataset.building_tensors.shape) #torch.Size([1, 440, 480])


        train_data = get_rx_data_by_tx_location(rldataset, rldataset.train_key, combine_sensors=True, required_limit=100)
        test_datasets = [get_rx_data_by_tx_location(rldataset, test_key, combine_sensors=True, required_limit=0) for test_key in ([rldataset.train_key] + rldataset.test_keys)]
        train_data=train_data,
        val_data=test_datasets[2]
        test_data=test_datasets[1]

        """print(train_data)
        These Train data are like dictionary.
        where key is each receiver station.
        values are: 1. numpy array with coordinates in pixels for all the transmitters
        2. normalized rss
        3. coordinate of the receiver
        4. keys of receivers nodes under this name
        5. category of reciver """
        tirem_correction_models = {}
        features, tirem_preds, labels, eval_errors = {}, {}, {}, {}
        for d in features, tirem_preds, labels, eval_errors:
            for key, option in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
                d[key] = {a:[] for a in range(2,5)}     
        # print(features, tirem_preds, labels, eval_errors)  
        # all_station_names = [train_data.keys()]
        all_device_names = list(device_lookup_dict.keys())
        
        for k in range(0,len(all_device_names)):
            name = all_device_names[k]
            # print("Name = ", name)
            if not os.path.exists(f'/uufs/chpc.utah.edu/common/home/u1472438/dl-image-localization-main/tirem_features3/tirem_all_features_DS{rldataset.params.dataset_index}_{name}.npz'):
                few_coords, rss, sensor_loc, keys, sensor_type = train_data[0][name]
                build_dictionary(rldataset=rldataset, name=name, sensor_loc=sensor_loc,keys=keys, sensor_type=sensor_type)
        train_data = train_data[0]
        for i, name in enumerate(train_data):
            for dataset, set_name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
                if dataset is None or name not in dataset:
                    continue
                few_coords, rss, sensor_loc, keys, sensor_type = dataset[name]
                if set_name == 'train' and hasattr(rldataset, 'sensors_to_remove') and keys[0] in rldataset.sensors_to_remove:
                    continue
                sensor_type = int(sensor_type)
                input_features, tirem_tensor, tirem_db = get_tirem_features(rldataset=rldataset, name=name, few_coords=few_coords, keys=keys)
                # for i in range(0,rss.shape[0]):
                #     print("True value = ", rss[i], " In database = ",tirem_tensor[i])
                # print(abcd)
                rss_tensor = torch.tensor(rss, dtype=torch.float32, device=rldataset.params.device) 
                features[set_name][sensor_type].append(input_features)
                tirem_preds[set_name][sensor_type].append(tirem_tensor)
                labels[set_name][sensor_type].append(rss_tensor)
        
        for set_name in features:
            for sensor_type in features[set_name]:
                features[set_name][sensor_type] = torch.concatenate(features[set_name][sensor_type])
                tirem_preds[set_name][sensor_type] = torch.concatenate(tirem_preds[set_name][sensor_type])
                labels[set_name][sensor_type] = torch.concatenate(labels[set_name][sensor_type])

        for sensor_type in features['train']:
            model = TiremMLP([features[set_name][sensor_type].shape[-1],200,100])
            model_file = f'models_tirem_nn/{rldataset.params}_{sensor_type}{"" if (not hasattr(rldataset, "sensors_to_remove") or len(rldataset.sensors_to_remove) == 0) else "muted_sensors"}.pt'
            backup_file = model_file.replace('unet__', 'unet2__')
            if os.path.exists(model_file) or os.path.exists(backup_file):
                try:
                    ckpt = torch.load(model_file)
                except:
                    ckpt = torch.load(backup_file)
                model.load_state_dict(ckpt)
                print('Loaded TIREM correction model for', sensor_type )
            else:
                mse_loss = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
                tv_err = 1e11
                num_epochs=80000
                for n in range(num_epochs):
                    y_pred = model(features['train'][sensor_type], tirem_preds['train'][sensor_type])
                    loss = mse_loss(labels['train'][sensor_type], y_pred.squeeze())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if n % 100 == 0:
                        for key in eval_errors:
                            preds = model(features[key][sensor_type], tirem_preds[key][sensor_type])
                            errors = labels[key][sensor_type] - preds.squeeze()
                            mean_err = abs(errors).mean().item()
                            eval_errors[key][sensor_type].append(mean_err)
                            if key == 'val' and mean_err < tv_err:
                                tv_err = mean_err
                                best_model = copy.deepcopy(model)
                            print(f"{n}/{num_epochs} and the mean_err = {mean_err}", end='\r')           
                print('Generated TIREM correction model for', sensor_type )
                torch.save(model.state_dict(), model_file)
                model = best_model
            tirem_correction_models[sensor_type] = model

        for sensor_type in features['test']:
            test_features = features['test'][sensor_type]
            test_tirem = tirem_preds['test'][sensor_type]
            true_labels = labels['test'][sensor_type]
            preds = tirem_correction_models[sensor_type](test_features, test_tirem )
            print("For Sensor type = ",sensor_type)
            for i in range(preds.shape[0]):
                print("True = ", true_labels[i].item()," and Tirem Preds = ",preds[i].item())


        


        # #tirem_normalized becomes the 15th column of tirem_features
        # input_features = torch.tensor(tirem_features, dtype=torch.float32, device=rldataset.params.device)
        # tirem_tensor = torch.tensor(tirem_normalized, dtype=torch.float32, device=rldataset.params.device) 
        # #so, input_features has normalized 15 features
        # #tirem_tensor has normalized rss value
        # #tirem_db has true rss values
        # true_rss_tensor = torch.tensor(rss, dtype=torch.float32, device=rldataset.params.device) 
        # print("Few coordinates min value = ",np.min(few_coords), " and Max value = ",np.max(few_coords)) min value =  4.963854  and Max value =  91.54554


























if __name__ == '__main__':
    main()
