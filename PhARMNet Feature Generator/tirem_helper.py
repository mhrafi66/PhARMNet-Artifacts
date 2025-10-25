import numpy as np
import os
from functools import lru_cache

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

@lru_cache(maxsize=2)
def load_tirem_prop_map(dataset_index, name, corners):
    print("Corners passed = ", corners)
    corners = np.array(corners).reshape(2,2)
    print("Corners Reshaped = ", corners)
    if dataset_index == 6:
        file_suffix = '_tirem_rssi_462.7.npz'
        npz_name = os.path.join(
            'tirem_prop/frs',
            device_lookup_dict[name]+file_suffix
        )
        downsample_ratio = 2
        bounds = np.zeros(2)
    elif dataset_index == 7:
        file_suffix = '_tirem_rssi_868.0.npz'
        npz_name = os.path.join(
            'tirem_prop/antwerp',
            name+file_suffix
        )
        downsample_ratio = 1
        bounds = np.zeros(2)
    elif dataset_index == 8:
        file_suffix = '_tirem_rssi_3534.0.npz'
        npz_name = os.path.join(
            'tirem_prop/cbrs',
            device_lookup_dict[name]+'-TX'+file_suffix
        )
        downsample_ratio = 2
        bounds = (downsample_ratio*(corners[0] - frs_dsm_corners[0])).round()
    bounds = bounds.astype(int)
    print("Bounds = ", bounds)
    size = (corners[1] - corners[0]).astype(int)*downsample_ratio + 10
    print("Size = ", size)
    print("NPZ_name = ", npz_name)
    # print("First Index = ", bounds[1]:bounds[1]+size[1]:downsample_ratio)
    # print("Last Index = ", bounds[0]:bounds[0]+size[0]:downsample_ratio)
    return np.load(npz_name)['arr_0'][bounds[1]:bounds[1]+size[1]:downsample_ratio,bounds[0]:bounds[0]+size[0]:downsample_ratio]

@lru_cache(maxsize=2)
def load_tirem_alt_params(dataset_index, name, corners):
    corners = np.array(corners).reshape(2,2)
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
    if dataset_index == 6:
        folder = 'tirem_prop/frs'
        file_suffix = '_462.7.npy'
        downsample_ratio = 2
        name_suffix = ''
        bounds = np.zeros(2)
    elif dataset_index == 7:
        folder = 'tirem_prop/antwerp'
        file_suffix = '_868.0.npy'
        downsample_ratio = 1
        name_suffix = ''
        bounds = np.zeros(2)
    elif dataset_index == 8:
        folder = 'tirem_prop/cbrs'
        file_suffix = '_3534.0.npy'
        downsample_ratio = 2
        name_suffix = '-TX'
        bounds = (downsample_ratio*(corners[0] - frs_dsm_corners[0])).round()
        
    bounds = bounds.astype(int)
    device_name = device_lookup_dict[name] if name in device_lookup_dict else name
    size = (corners[1] - corners[0]).astype(int)*downsample_ratio + 10
    try:
        npz_file = os.path.join(
            folder,
            device_name+name_suffix+'_tirem_rssi'+file_suffix[:-1]+'z'
        )
        f = np.log(np.load(npz_file)['arr_1'][bounds[1]:bounds[1]+size[1]:downsample_ratio,bounds[0]:bounds[0]+size[0]:downsample_ratio] + 1)
        param_results = np.zeros((f.shape[0], f.shape[1], 14), dtype=np.float32)
        param_results[:,:,0] = f
        current_ind = 1
        for param in param_keys:
            npy_file = os.path.join(
                folder,
                device_name+name_suffix+param+file_suffix
            )
            f = np.load(npy_file)[bounds[1]:bounds[1]+size[1]:downsample_ratio,bounds[0]:bounds[0]+size[0]:downsample_ratio]
            if len(f.shape) == 2:
                param_results[:,:,current_ind] = f
                current_ind += 1
            else:
                param_results[:,:,current_ind:current_ind + f.shape[-1]] = f
                current_ind += f.shape[-1]
        scale_factors = np.array([
            8,20,180,180,
            180,180,180,180,
            180, 1, 20, 20,
            180,180
        ]).astype(float).reshape(1,1,-1)
        param_results[:] /= scale_factors
        return param_results
    except:
        print(f'No params in {device_name}')