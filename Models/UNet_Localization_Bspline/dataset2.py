import coordinates
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import torch
from typing import List
from scipy.spatial.distance import cdist
from cv2 import getAffineTransform, warpAffine
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn.model_selection import train_test_split
from collections import defaultdict
from IPython import embed

import os
import csv
import numpy as np

from locconfig import LocConfig
#This is actually for bspline
data_files = {
        1:['datasets/data1/data1.txt'],
        2:['datasets/data2/data1.txt'],
        3:['datasets/data3/data1.txt', 'datasets/data3/data2.txt'],
        4:['datasets/data4/data1.txt', 'datasets/data4/data2.txt'],
        5:['datasets/data5/data1.txt', 'datasets/data5/data2.txt'],
        6:['datasets/frs_data/separated_data/all_data/no_tx.json', 'datasets/frs_data/separated_data/all_data/single_tx.json', 'datasets/frs_data/separated_data/all_data/two_tx.json'],
        7:['datasets/data_antwerp/lorawan_antwerp_2019_dataset.json.txt'],
        8:['datasets/slc_cbrs_data/slc_prop_measurement/data/data.json'],
}
#it has link to all the datafiles


class RSSLocDataset():
    class Samples():
        def __init__(self, rldataset, rx_vecs=None, tx_vecs=None, filter_boundaries=[], filter_rx=False, tx_metadata=None, no_shift=False):
            """
            rx_vecs: Iterable of Rx rss and locations
            tx_vecs: Iterable of Tx locations
            filter_boundaries: list of np.arrays, coordinates with which to filter rx_vecs and/or tx_vecs. Must be in same coordinate system as rx_vecs and tx_vecs
            """
            #comes with array of rx_vecs, tx_vecs and tx_metadata
            assert(rx_vecs is not None and tx_vecs is not None)
            #asserts that it has to have both rx and tx vecs
            self.made_tensors = False
            #made_tensors = False
            self.rldataset = rldataset
            #store the object
            self.origin = np.zeros(2)
            #initialize origin to be all zeros
            meter_scale = self.rldataset.params.meter_scale
            #take the whatever predefined meter_scale
            if not no_shift:
                self.origin = np.array([self.rldataset.min_x, self.rldataset.min_y]) - self.rldataset.buffer*meter_scale
                #origin coordinates should be one pixel below min_x, min_y so that the min values are marked as 1

            # print("self.origin = ", self.origin)
            # print("meter_scale = ", meter_scale)
            # print("self.rldataset.buffer = ", self.rldataset.buffer)
            # print(" self.rldataset.max_x = ", self.rldataset.max_x, " self.rldataset.min_x = ", self.rldataset.min_x)
            # print(" self.rldataset.max_y = ", self.rldataset.max_y, " self.rldataset.min_y = ", self.rldataset.min_y)
            self.rectangle_width = self.rldataset.max_x - self.rldataset.min_x + 2*self.rldataset.buffer*meter_scale
            self.rectangle_height = self.rldataset.max_y - self.rldataset.min_y + 2*self.rldataset.buffer*meter_scale

            # print("self.rectangle_width = ", self.rectangle_width)
            # print("self.rectangle_height = ", self.rectangle_height)
            #take height and weidth two more coordinates so that padded two positions left in the boundary
            self.rectangle_width = round(self.rectangle_width / (meter_scale*2)) * meter_scale*2
            self.rectangle_height = round(self.rectangle_height / (meter_scale*2)) * meter_scale*2
            #actually converting the value to the nearest multiple of meter_scale*2
            # print("self.rectangle_width = ", self.rectangle_width)
            # print("self.rectangle_height = ", self.rectangle_height)

            # print(abcd)

            
            if len(filter_boundaries) > 0:
            #we kept it empty
            #it for filtering the dataset in a specific boundary
                for i, filter_boundary in enumerate(filter_boundaries):
                    if not isinstance(filter_boundary, Polygon):
                        filter_boundaries[i] = Polygon(filter_boundary)
                new_tx_vecs = []
                new_rx_vecs = []
                new_tx_metadata = []
                for sample_ind, (tx_vec, rx_vec) in enumerate(zip(tx_vecs, rx_vecs)):
                    if filter_rx:
                        tx_locs, rx_tups = rldataset.filter_bounds(filter_boundaries, tx_coords=tx_vec, rx_coords=rx_vec)
                        if len(tx_locs) == 0:
                            continue
                        new_rx_vecs.append(rx_tups)
                    else:
                        tx_locs = rldataset.filter_bounds(filter_boundaries, tx_coords=tx_vec)
                        if len(tx_locs) == 0:
                            continue
                        new_rx_vecs.append(rx_vec)
                    new_tx_vecs.append(tx_locs)
                    if len(tx_metadata) > 0:
                        metadata = tx_metadata[sample_ind]
                        new_tx_metadata.append(metadata)
                tx_vecs = new_tx_vecs
                rx_vecs = new_rx_vecs
                tx_metadata = new_tx_metadata
            
            self.tx_vecs = np.array(tx_vecs, dtype=object)
            self.rx_vecs = np.array(rx_vecs, dtype=object)
            self.tx_metadata = np.array(tx_metadata, dtype=object) if tx_metadata is not None else np.array([])

            self.max_num_tx = max([len(vec) for vec in self.tx_vecs])
            self.max_num_rx = self.rldataset.max_num_rx
            # print(len(tx_vecs))
            # print(len(tx_vecs[0]))
            # print(tx_vecs[0])
            # self.tx_vecs = np.array(tx_vecs)
            # self.tx_vecs = [np.array(tx_vec) if len(tx_vec) > 0 else np.empty((0, 2)) for tx_vec in tx_vecs]

            # max_num_tx = max(len(vec) for vec in tx_vecs)
            # padded_tx_vecs = np.zeros((len(tx_vecs), max_num_tx, 2))

            # for i, tx_vec in enumerate(tx_vecs):
            #     tx_array = np.array(tx_vec)
            #     if len(tx_array) > 0:
            #         padded_tx_vecs[i, :len(tx_array)] = tx_array

            # self.tx_vecs = padded_tx_vecs  # Now a 3D NumPy array

            # self.tx_vecs = np.array(self.tx_vecs, dtype=object)
            # self.rx_vecs = np.array(rx_vecs, dtype=object)
            # # self.tx_metadata = np.array(tx_metadata) if tx_metadata is not None else np.array([])
            # # self.tx_metadata = [np.array(metadata) if metadata is not None and len(metadata) > 0 else np.empty((0,)) for metadata in tx_metadata]
            
            # max_length = max(len(meta) if hasattr(meta, '__len__') else 0 for meta in tx_metadata)
            
            # padded_tx_metadata = np.zeros((len(tx_metadata), max_length), dtype=object)  # Replace 'desired_dtype' with the appropriate one
            # for i, meta in enumerate(tx_metadata):
            #     if meta is not None and hasattr(meta, '__len__'):
            #         padded_tx_metadata[i, :len(meta)] = meta
            # self.tx_metadata = padded_tx_metadata

            # self.tx_metadata = np.array(self.tx_metadata, dtype=object)

            # self.max_num_tx = max([len(vec) for vec in self.tx_vecs])
            # print(self.tx_vecs.shape)
            # print(self.max_num_tx)
            # self.max_num_rx = self.rldataset.max_num_rx
            # #how many transmitters and receivers are there
            # max_num_tx = max(len(tx) for tx in tx_vecs)
            # dummy_value = [np.nan, np.nan]  # Define the dummy value
            # print("Here is padding happening")
            # padded_tx_vecs = [
            #     list(tx) + [dummy_value] * (max_num_tx - len(tx))
            #     for tx in tx_vecs
            # ]
            # self.tx_vecs = np.array(padded_tx_vecs)
            # # self.tx_vecs = np.array(tx_vecs)
            # self.rx_vecs = np.array(rx_vecs, dtype=object)
            # print("In samples, shape of rx_vecs = ", self.rx_vecs.shape)
            # print("In samples, shape of tx_vecs = ", self.tx_vecs.shape)
            # # self.tx_metadata = np.array(tx_metadata) if tx_metadata is not None else np.array([])
            # max_length = max(len(meta) if hasattr(meta, '__len__') else 0 for meta in tx_metadata)
            
            # padded_tx_metadata = np.zeros((len(tx_metadata), max_length), dtype=object)  # Replace 'desired_dtype' with the appropriate one
            # for i, meta in enumerate(tx_metadata):
            #     if meta is not None and hasattr(meta, '__len__'):
            #         padded_tx_metadata[i, :len(meta)] = meta
            # self.tx_metadata = padded_tx_metadata

            # self.tx_metadata = np.array(self.tx_metadata, dtype=object)

            # self.max_num_tx = max([len(vec) for vec in self.tx_vecs])
            # self.max_num_rx = self.rldataset.max_num_rx
        
        def make_tensors(self):
            self.made_tensors = True



            # Appending the tx and rx distance to yvecs
            if self.tx_vecs[0].shape[-1] < 3:
                if len(self.tx_vecs.shape) == 3:
                    tmp_shape = list(self.tx_vecs.shape)
                    tmp_shape[-1] = tmp_shape[-1] + 2
                    tmp_tx_vecs = np.zeros(tmp_shape)
                    #Prepares an expanded temporary array to store TX vectors with 2 extra columns: one for distance to other TXs, one for distance to RXs.
                else:
                    tmp_tx_vecs = self.tx_vecs
                    #If shape is not 3D, keep as-is (could be malformed or unexpected input).
                for i, (rx_vec, tx_vec) in enumerate(zip(self.rx_vecs, self.tx_vecs)):
                    if len(tx_vec) == 0: continue
                    tx_distances = []
                    rx_distances = []
                    #For each TX in this sample, initialize lists to collect distances.
                    for tx in tx_vec:
                        min_tx_distance = np.ma.masked_equal(np.linalg.norm( (tx[0:2] - tx_vec[:,0:2]).astype(float), axis=1), 0, copy=False).min()
                        #Computes minimum distance from current TX to all other TXs in the same sample, excluding itself (masking zero distance).
                        min_rx_distance = np.linalg.norm( (tx[0:2] - rx_vec[:,1:3]).astype(float), axis=1).min()
                        #Computes minimum distance from this TX to all RXs.
                        tx_distances.append(min_tx_distance)
                        rx_distances.append(min_rx_distance)

                    tx_distances = np.array(tx_distances).reshape(-1,1)
                    rx_distances = np.array(rx_distances).reshape(-1,1)
                    tmp_tx_vecs[i] = np.hstack((self.tx_vecs[i], tx_distances, rx_distances))
                    #Append distances as new features (columns) to the current TX vector array
                self.tx_vecs = tmp_tx_vecs
            else:
                for i, (rx_vec, tx_vec) in enumerate(zip(self.rx_vecs, self.tx_vecs)):
                    if len(tx_vec) == 0: continue
                    rx_distances = []
                    for tx in tx_vec:
                        min_rx_distance = np.linalg.norm((tx[0:2] - rx_vec[:,1:3]).astype(float), axis=1).min()
                        rx_distances.append(min_rx_distance)
                    rx_distances = np.array(rx_distances)
                    self.tx_vecs[i][:,-1] = rx_distances

            num_params = 2
            rx_vecs_arr = np.zeros((len(self.rx_vecs),self.max_num_rx+(10 if self.rldataset.params.adv_train else 0), num_params+3))
            #Dimensions: [samples, max RXs (+ extra for adv training), features]
            #Features per RX: 2 params (e.g., x, y) + 3 additional (likely: RSS, sensor ID, type).
            #IMPORTANT: Why 10 more dimensions for the adversarial training

            for i, rx_vec in enumerate(self.rx_vecs):
                if len(rx_vec):
                    #consistent positioning and scaling
                    rx_vec = rx_vec / np.array([1,self.rldataset.params.meter_scale, self.rldataset.params.meter_scale,1,1])
                    rx_vecs_arr[i,rx_vec[:,3].astype(int)] = rx_vec
                    self.rx_vecs[i][:,1:3] -= self.origin
            rx_vecs_arr[:,:,1:3] -= self.origin/self.rldataset.params.meter_scale # Adjust coordinates to the bottom left corner of rectangle as origin
            rx_vecs_arr[rx_vecs_arr < 0] = 0
            rx_vecs_tensor = torch.Tensor(rx_vecs_arr).to(self.rldataset.params.device)

            tx_vecs_arr = np.zeros((len(self.tx_vecs),self.max_num_tx, num_params+1))
            for i, tx_vec in enumerate(self.tx_vecs):
                if len(tx_vec):
                    tx_vecs_arr[i,:len(tx_vec),0] = 1
                    tx_vecs_arr[i,:len(tx_vec),1:num_params+1] = (np.array(tx_vec)[:,:num_params] - self.origin) / np.array([self.rldataset.params.meter_scale, self.rldataset.params.meter_scale])
                    self.tx_vecs[i][:,0:2] -= self.origin
            tx_vecs_tensor = torch.Tensor(tx_vecs_arr).to(self.rldataset.params.device)
            dataset = torch.utils.data.TensorDataset(rx_vecs_tensor, tx_vecs_tensor)
            pin_memory = self.rldataset.params.device != torch.device('cuda') 
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.rldataset.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.rldataset.params.batch_size, shuffle=False, pin_memory=pin_memory)


    def __init__(self, params: LocConfig, sensors_to_remove: List[int]=[]):
        """
        params: a LocConfig object, see config.py for parameters
        sensors_to_remove: List[int] input indicies to remove from training data, for experiments on adding new devices.
        """
        #from the training file, we landed here with config object with all the configuration values
        #and no sensors to remove
        self.params = params
        #keeping the object in a file
        self.data = {}
        #data dictionary is empty
        self.buffer = 1
        #buffer size is equal to 1
        self.data_files = data_files[self.params.dataset_index]
        #take dataset locations into data_files variable
        self.load_data()
        # print(self.data[None].__dict__)
        # print(abcd)
        #calling load_data from here
        #now jump to load_data() function
        #after this line, there is a dictionary "data" with key "None" stoing the tx,rx and metadata values
        self.elevation_tensors = None
        self.building_tensors = None
        #we have the map but dunno why the tensors are None

        # print("Before Making dataset the data = ", vars(self.data[None]))
        # 'tx_vecs': array([[[3100.5790935840923, 1734.1401453958824]]
        self.make_datasets(
            split=self.params.data_split,
            make_val=self.params.make_val,
            should_augment=self.params.should_augment,
            convert_all_inputs=True,
            sensors_to_remove=sensors_to_remove
        )
        #it will come up with more keys in the dictionary respective to train and test sets


    def make_elevation_tensors(self, meter_scale=None):
        meter_scale = self.params.meter_scale if meter_scale is None else meter_scale
        corners = self.corners.copy()
        if len(corners) == 4:
            height = int(np.linalg.norm(corners[0] - corners[1]).round())
            width = int(np.linalg.norm(corners[1] - corners[2]).round())
        else:
            height = corners[1,1] - corners[0,1]
            width = corners[1,0] - corners[0,0]
            corners = np.array([corners[0], [corners[0,0], corners[1,1]], corners[1] ])
            #store three corners points

        transform = np.array(self.elevation_map.transform).reshape(3,3)
        transform[:2,2] -= self.elevation_map.origin
        #take elevation map and deduct origin to get transform array
        #This part takes the transformation matrix (self.elevation_map.transform), 
        # reshapes it into a 3x3 matrix, and then subtracts the origin (from the geotiff file) from the translation portion of the transformation matrix.
        building_tensor = None
        inv_transform = np.linalg.inv(transform)
        #This creates the inverse of the transformation matrix, which will be used to convert the corner coordinates into image pixel coordinates for further processing.
        if len(corners) == 4:
            padded_coordinates = np.hstack((corners, np.ones((4,1)) )).T
            img_coords = (inv_transform @ padded_coordinates)[:2].T

            img_to_rect_transform = getAffineTransform(img_coords[:3].astype(np.float32), np.float32([[0,0],[0,height], [width,height]]))
            warped_img = warpAffine(self.elevation_map.read(1), img_to_rect_transform, (width, height) )
            if self.building_map is not None:
                building_img = warpAffine(self.building_map.read(1), img_to_rect_transform, (width, height) )
            img = warped_img
            downsample_rate = round(meter_scale) #Rounding since resolution is a messy float
        else:
            a,b = corners.min(axis=0), corners.max(axis=0)
            #It identifies the minimum (a) and maximum (b) coordinates from the corners.
            a_ind = (inv_transform @ np.array([a[0], a[1], 1]) ).round().astype(int)
            b_ind = (inv_transform @ np.array([b[0], b[1], 1]) ).round().astype(int)
            #These corner points are transformed into pixel coordinates (a_ind, b_ind) using the inverse transformation matrix.
            sub_img = self.elevation_map.read(1)[b_ind[1]:a_ind[1]+1, a_ind[0]:b_ind[0]+1]
            #sub_img = elevation_map.read(1)[a_ind[0]:b_ind[0]+1, b_ind[1]:a_ind[1]+1]
            sub_img = np.flipud(sub_img)
            if self.building_map is not None:
                building_img = self.building_map.read(1)[b_ind[1]:a_ind[1]+1, a_ind[0]:b_ind[0]+1]
                building_img = np.flipud(building_img)
                #The image is flipped vertically because the image's coordinate system (top-left origin) is 
                # different from the geographic coordinate system (bottom-left origin). Flipping ensures the 
                # data is aligned correctly with the geographic coordinates.
                #it actually meaning upside down
            img = sub_img
            #It extracts a subregion (sub_img) of the elevation map between these two points and flips the image vertically (np.flipud) to ensure proper orientation.
            downsample_rate = round(meter_scale / self.elevation_map.res[0]) #Rounding since resolution is a messy float
            # Could also use avg_pool2d, not sure that it makes much of a difference.
            #The function calculates the downsampling rate by dividing the meter_scale 
            # by the resolution of the elevation map (self.elevation_map.res[0]), ensuring the image is downsampled to fit the desired scale.
        elevation_tensor = torch.nn.functional.max_pool2d(torch.tensor(img.copy()).unsqueeze(0), downsample_rate) 
        #The elevation image (img) is converted into a PyTorch tensor and downsampled using max_pool2d. 
        # This ensures that the elevation data is reduced in size based on the downsample_rate.
        elevation_tensor = (elevation_tensor - elevation_tensor.min()) / 300
        #The tensor is normalized by subtracting the minimum value and dividing by 300 
        # (a constant likely used to scale elevation values into a manageable range).
        if self.building_map is not None:
            building_tensor = torch.nn.functional.max_pool2d(torch.tensor(building_img.copy()).unsqueeze(0), downsample_rate) 
        self.elevation_tensors = elevation_tensor
        self.building_tensors = building_tensor
        #it is just creatign elevation and building tensors


    def separate_dataset(self, separation_method, excluded_metadata={'stationary':'stationary', 'transport':'inside'} ,data_key_prefix='', grid_size=5, train_split=0.8, source_key=None, keys=['train', 'test'], random_state=None):
        inds = self.filter_inds_by_metadata(excluded_metadata, source_key=source_key)
        #Came here with train or test set
        if separation_method == 'grid':
            if self.params.one_tx:
                if self.params.only_print_propagation_estimation:
                    (train_inds, train_grid_inds), (test_inds, test_grid_inds) = self._get_random_grid_old_only_sides(inds, grid_size=grid_size, train_split=0.05, randomly_add_mixed=False, source_key=source_key, random_state=random_state, edge='right')
                else:
                    (train_inds, train_grid_inds), (test_inds, test_grid_inds) = self._get_random_grid_old(inds, grid_size=grid_size, train_split=train_split, randomly_add_mixed=False, source_key=source_key, random_state=random_state)
            else:
                (train_inds, train_grid_inds), (test_inds, test_grid_inds) = self._get_random_grid(inds, grid_size=grid_size, train_split=train_split, randomly_add_mixed=False, source_key=source_key, random_state=random_state)
            print("In separate Dataset")
            print("Train_grids = ", train_grid_inds)
            print("Test Grids = ", test_grid_inds)
            print("One Tx Var = ", self.params.one_tx)
            separation_method = separation_method + str(grid_size)
        elif separation_method == 'walking':
            train_inds, test_inds = self._metadata_filter(inds, search_key='transport', search_value='walking', source_key=source_key)
        elif separation_method == 'driving':
            train_inds, test_inds = self._metadata_filter(inds, search_key='transport', search_value='driving', source_key=source_key)
        elif separation_method == 'station':
            train_inds, test_inds = self._metadata_filter(inds, search_key='stationary', source_key=source_key)
        elif separation_method == 'mobile':
            test_inds, train_inds = self._metadata_filter(inds, search_key='stationary', source_key=source_key)
        else: 
            raise NotImplementedError
        rx_vecs = self.data[source_key].rx_vecs
        tx_vecs = self.data[source_key].tx_vecs

        if len(data_key_prefix) == 0:
            train_key = separation_method + '_' + keys[0] 
            test_key = separation_method + '_' + keys[1]
        else:
            train_key = data_key_prefix+ separation_method + '_' + keys[0]
            test_key = data_key_prefix+ separation_method + '_' + keys[1]

        #Train Key =  0.2testsizegrid5_train
        #Test Key =  0.2testsizegrid5_test
        x_vecs_train, x_vecs_test = rx_vecs[train_inds], rx_vecs[test_inds]
        label_train, label_test = tx_vecs[train_inds], tx_vecs[test_inds]
        metadata = self.data[source_key].tx_metadata

        if len(metadata) > 0:
            metadata_train, metadata_test = metadata[train_inds], metadata[test_inds]
            self.data[train_key] = self.Samples(self, x_vecs_train, label_train, tx_metadata=metadata_train, no_shift=True)
            self.data[test_key] = self.Samples(self, x_vecs_test, label_test, tx_metadata=metadata_test, no_shift=True)
        else:
            self.data[train_key] = self.Samples(self, x_vecs_train, label_train, no_shift=True)
            self.data[test_key] = self.Samples(self, x_vecs_test, label_test, no_shift=True)
        if 'grid' in separation_method:
            self.data[train_key].grid_inds = train_grid_inds
            self.data[test_key].grid_inds = test_grid_inds
        return train_key, test_key


    def filter_inds_by_metadata(self, excluded_metadata={'stationary':'stationary', 'transport':'inside'}, source_key=None):
        inds = np.arange(len(self.data[source_key].tx_vecs))
        for key in excluded_metadata:
            value = excluded_metadata[key]
            _, inds = self._metadata_filter(inds, search_key=key, search_value=value)
        return inds


    def _metadata_filter(self, inds, search_key, source_key=None, search_value=None):
        """
        Split the dataset by metadata terms:
        stationary: an int index, where samples with the same index are taken at the same location
        transport: 'driving' or 'walking'
        radio: 'Audiovox', 'TXA', or 'TXB'
        power: 0.5 or 1 (Audiovox or Baofeng)
        """
        inds_with_term = []
        inds_without_term = []
        tx_metadata = self.data[source_key].tx_metadata
        if len(tx_metadata) == 0 or self.params.dataset_index != 6:
            return inds_with_term, inds
        for ind in inds:
            if search_key == 'stationary':
                if 'stationary' in tx_metadata[ind][0]:
                    ### Should we have a function here to combine stationary samples into one huge sample?
                    inds_with_term.append(ind)
                else:
                    inds_without_term.append(ind)
            else:
                if tx_metadata[ind][0][search_key] == search_value:
                    inds_with_term.append(ind)
                else:
                    inds_without_term.append(ind)
        inds_with_term, inds_without_term = np.array(inds_with_term, dtype=int), np.array(inds_without_term, dtype=int)
        return inds_with_term, inds_without_term

    def _select_grids_for_test(self, grid_to_one_tx, grid_to_two_tx,
                            total_one_tx, total_two_tx,
                            one_tx_target_pct=0.2, two_tx_target_pct=0.2,
                            one_tx_tol=0.05, two_tx_tol=0.01,
                            max_iter=500):

        one_tx_min = int(total_one_tx * (one_tx_target_pct - one_tx_tol))
        one_tx_max = int(total_one_tx * (one_tx_target_pct + one_tx_tol))
        two_tx_min = int(total_two_tx * (two_tx_target_pct - two_tx_tol))
        two_tx_max = int(total_two_tx * (two_tx_target_pct + two_tx_tol))

        all_grids = list(set(grid_to_one_tx) | set(grid_to_two_tx))
        two_tx_grids = [g for g in all_grids if len(grid_to_two_tx[g]) > 0]
        two_tx_grids.sort(key=lambda g: len(grid_to_two_tx[g]))

        selected_grids = set()
        selected_one_tx = set()
        selected_two_tx = set()

        for g in two_tx_grids:
            selected_grids.add(g)
            selected_one_tx.update(grid_to_one_tx.get(g, set()))
            selected_two_tx.update(grid_to_two_tx.get(g, set()))
            if len(selected_two_tx) >= two_tx_min:
                break

        for _ in range(max_iter):
            one_count = len(selected_one_tx)
            two_count = len(selected_two_tx)

            if one_tx_min <= one_count <= one_tx_max and two_tx_min <= two_count <= two_tx_max:
                break

            if one_count > one_tx_max:
                drop_cands = sorted(
                    list(selected_grids),
                    key=lambda g: len(grid_to_one_tx[g]), reverse=True
                )[:5]
                drop_cands.sort(key=lambda g: len(grid_to_two_tx[g]))
                g = drop_cands[0]
                selected_grids.remove(g)
            elif one_count < one_tx_min:
                remain = [g for g in all_grids if g not in selected_grids and len(grid_to_one_tx[g]) > 0]
                remain.sort(key=lambda g: len(grid_to_two_tx[g]))
                if not remain:
                    break
                g = remain[0]
                selected_grids.add(g)

            selected_one_tx = set.union(*(grid_to_one_tx.get(gr, set()) for gr in selected_grids))
            selected_two_tx = set.union(*(grid_to_two_tx.get(gr, set()) for gr in selected_grids))

        return selected_grids, selected_one_tx, selected_two_tx


    def _get_random_grid(self, inds, grid_size=5, train_split=0.8,
                        randomly_add_mixed=True, x_grid_size=None, y_grid_size=None,
                        source_key=None, random_state=None):

        source = self.data[source_key]

        if x_grid_size is None:
            x_grid_size = grid_size
        if y_grid_size is None:
            y_grid_size = grid_size

        all_tx_coords = np.vstack([
            tx[:, :2] if isinstance(tx, np.ndarray) else np.array([]).reshape(0, 2)
            for tx in source.tx_vecs if tx is not None and len(tx)
        ])
        min_bounds = all_tx_coords.min(axis=0) - 1
        max_bounds = all_tx_coords.max(axis=0) + 1

        x_bins = np.linspace(min_bounds[0], max_bounds[0], x_grid_size + 1)
        y_bins = np.linspace(min_bounds[1], max_bounds[1], y_grid_size + 1)
        self.x_grid_lines = x_bins
        self.y_grid_lines = y_bins

        sample_to_grids = defaultdict(set)
        grid_to_one_tx = defaultdict(set)
        grid_to_two_tx = defaultdict(set)
        one_tx_inds = []
        two_tx_inds = []

        for idx in inds:
            tx_vec = source.tx_vecs[idx]
            if tx_vec is None or len(tx_vec) == 0:
                continue
            is_two_tx = (len(tx_vec) == 2)
            for tx in tx_vec:
                x_ind = np.clip(np.digitize(tx[0], x_bins) - 1, 0, x_grid_size - 1)
                y_ind = np.clip(np.digitize(tx[1], y_bins) - 1, 0, y_grid_size - 1)
                grid = np.ravel_multi_index((x_ind, y_ind), (x_grid_size, y_grid_size))
                sample_to_grids[idx].add(grid)
                if is_two_tx:
                    grid_to_two_tx[grid].add(idx)
                else:
                    grid_to_one_tx[grid].add(idx)
            if is_two_tx:
                two_tx_inds.append(idx)
            else:
                one_tx_inds.append(idx)

        selected_grids, selected_one_tx, selected_two_tx = self._select_grids_for_test(
            grid_to_one_tx, grid_to_two_tx,
            total_one_tx=len(one_tx_inds),
            total_two_tx=len(two_tx_inds),
            one_tx_target_pct=(1 - train_split),
            two_tx_target_pct=(1 - train_split),
            one_tx_tol=0.05,
            two_tx_tol=0.03,
            max_iter=500
        )

        train_inds, test_inds = [], []
        for idx in inds:
            if sample_to_grids[idx] & selected_grids:
                test_inds.append(idx)
            else:
                train_inds.append(idx)

        print(f"[INFO] 2-TX test samples: {len(selected_two_tx)} / {len(two_tx_inds)} = {100 * len(selected_two_tx)/len(two_tx_inds):.2f}%")
        print(f"[INFO] 1-TX test samples: {len(selected_one_tx)} / {len(one_tx_inds)} = {100 * len(selected_one_tx)/len(one_tx_inds):.2f}%")
        print(f"[INFO] Selected test grids : {selected_grids}")

        print("Selected two tx = ", selected_two_tx)
        print("Total two tx = ", two_tx_inds)

        return (
            train_inds,
            list(set(range(x_grid_size * y_grid_size)) - selected_grids)
        ), (
            test_inds,
            list(selected_grids)
        )


        

    
    def _get_random_grid_old(self, inds, grid_size=5, train_split=0.5, randomly_add_mixed=True, x_grid_size=None, y_grid_size=None, source_key=None, random_state=None):
        source = self.data[source_key]
        #Source key =  0.2testsize_train
        if x_grid_size is None:
            x_grid_size = grid_size
        if y_grid_size is None:
            y_grid_size = grid_size
        #X and Y Gird Size =  10   10

        # print(source.tx_vecs[:100])
        # print(abcd)

        min_bounds = source.tx_vecs.min(axis=0)[0,:2] - 1
        max_bounds = source.tx_vecs.max(axis=0)[0,:2] + 1

        #Min bounds =  [123.09634137 427.3925714 ]  Max Bounds =  [2289.6385351  2168.81462927]
        x_bins = np.linspace(min_bounds[0], max_bounds[0], x_grid_size+1)
        y_bins = np.linspace(min_bounds[1], max_bounds[1], y_grid_size+1)
        self.x_grid_lines = x_bins
        self.y_grid_lines = y_bins
        #X Grid Lines =  [ 123.09634137  339.75056074  556.40478011  773.05899949  989.71321886 1206.36743824 1423.02165761 1639.67587698 1856.33009636 2072.98431573 2289.6385351 ]
        #Y Grid Lines =  [ 427.3925714   601.53477719  775.67698298  949.81918876 1123.96139455 1298.10360034 1472.24580612 1646.38801191 1820.5302177  1994.67242348 2168.81462927]
        if random_state is None:
            random_state = self.params.random_state
        np.random.seed(random_state)
        choices = list(range(x_grid_size*y_grid_size))

        #Choices =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        train_grid_flat = list(range(x_grid_size*y_grid_size))

        #Train grid flat =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        if hasattr(source, 'grid_inds'):
            choices = copy.deepcopy(source.grid_inds)
            train_grid_flat = copy.deepcopy(source.grid_inds)
        train_size = round(len(choices) * train_split)
        #print("Train Size = ", train_size) 80
        test_size = len(choices) - train_size
        #print("Test Size = ", test_size) 20
        test_grid_flat = []
        backup_inds = []
        while len(test_grid_flat) < test_size:
            choice = np.random.choice(choices)
            #Choice in the loop =  44
            choices.remove(choice)
            train_grid_flat.remove(choice)
            for to_remove in [choice+1, choice-1, choice+x_grid_size, choice-x_grid_size]:
                # print("To Remove = ", to_remove)
                try:
                    choices.remove(to_remove)
                    backup_inds.append(to_remove)
                    #adding some nearby indices to the backup so that can be used if needed
                except:
                    pass
            test_grid_flat.append( choice )
            if len(choices) == 0:
                ind_need = test_size - len(test_grid_flat)
                new_inds = np.random.choice(backup_inds, size=ind_need).tolist()
                test_grid_flat += new_inds
                for ind in new_inds:
                    train_grid_flat.remove(ind)
                break
        
        #Test Grid Flat =  [44, 51, 74, 81, 86, 9, 25, 53, 31, 6, 46, 49, 22, 1, 97, 68, 79, 60, 18, 95]
        #Backup inds =  [45, 43, 54, 34, 52, 50, 61, 41, 75, 73, 84, 64, 82, 80, 91, 71, 87, 85, 96, 76, 10, 8, 19, 26, 24, 35, 15, 63, 32, 30, 21, 7, 5, 16, 47, 56, 36, 48, 59, 39, 23, 12, 2, 0, 11, 98, 69, 67, 78, 58, 89, 70, 17, 28, 94]


        train_inds = []
        test_inds = []
        for ind in inds:
            #loop one all the indices passed
            if len(source.tx_vecs[ind]) == 0: # If there are no transmitters (0 Tx)
                #if no transmitter in the sample, then add wherever you want to
                if np.random.random() <= train_split:
                    train_inds.append(ind)
                else:
                    test_inds.append(ind)
                continue
            tx_vec = source.tx_vecs[ind] # - source.origin
            #Take the tx_position
            x_inds = np.digitize(tx_vec[:,0], x_bins) - 1
            y_inds = np.digitize(tx_vec[:,1], y_bins) - 1
            #x_inds =  [8] - which bins the x coordinate falls in
            #y_inds = [3] - which bins the y coordinate falls in
            grid_inds = np.ravel_multi_index((x_inds, y_inds), (x_grid_size, y_grid_size))
            #grid_inds =  [83] - final grid number based on indices
            # print("grid_inds = ", grid_inds)
            ## I'm not sure if this is still working, with the two separate x and y grid divisions.
            #then look into which grid falls in which set and so assign the respective index in that set
            if all(x in train_grid_flat for x in grid_inds):
                train_inds.append(ind)
            elif all(x not in train_grid_flat for x in grid_inds):
                test_inds.append(ind)
            elif randomly_add_mixed: # If transmitters occur in both train and test grids, randomly assign them to either side
                if np.random.random() >= train_split:
                    train_inds.append(ind)
                else:
                    test_inds.append(ind)
            else: # If we don't add mixed transmitters randomly, just add them to test
                test_inds.append(ind)

        # print(abcd)
        # print("Train inds = ", train_inds) Train inds =  [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18]
        # print("Train Grid Flat = ", train_grid_flat) Train Grid Flat =  [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24]
        return (train_inds, train_grid_flat), (test_inds, test_grid_flat)

    def _get_random_grid_old_only_sides(self, inds, grid_size=5, train_split=0.5, randomly_add_mixed=True, x_grid_size=None, y_grid_size=None, source_key=None, random_state=None, edge = 'left'):
        source = self.data[source_key]
        #Source key =  0.2testsize_train
        if x_grid_size is None:
            x_grid_size = grid_size
        if y_grid_size is None:
            y_grid_size = grid_size
        #X and Y Gird Size =  10   10

        # print(source.tx_vecs[:100])
        # print(abcd)

        min_bounds = source.tx_vecs.min(axis=0)[0,:2] - 1
        max_bounds = source.tx_vecs.max(axis=0)[0,:2] + 1

        #Min bounds =  [123.09634137 427.3925714 ]  Max Bounds =  [2289.6385351  2168.81462927]
        x_bins = np.linspace(min_bounds[0], max_bounds[0], x_grid_size+1)
        y_bins = np.linspace(min_bounds[1], max_bounds[1], y_grid_size+1)
        self.x_grid_lines = x_bins
        self.y_grid_lines = y_bins
        #X Grid Lines =  [ 123.09634137  339.75056074  556.40478011  773.05899949  989.71321886 1206.36743824 1423.02165761 1639.67587698 1856.33009636 2072.98431573 2289.6385351 ]
        #Y Grid Lines =  [ 427.3925714   601.53477719  775.67698298  949.81918876 1123.96139455 1298.10360034 1472.24580612 1646.38801191 1820.5302177  1994.67242348 2168.81462927]
        if random_state is None:
            random_state = self.params.random_state
        np.random.seed(random_state)
        choices = list(range(x_grid_size*y_grid_size))

        #Choices =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        train_grid_flat = list(range(x_grid_size*y_grid_size))

        #Train grid flat =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        if hasattr(source, 'grid_inds'):
            choices = copy.deepcopy(source.grid_inds)
            train_grid_flat = copy.deepcopy(source.grid_inds)
        train_size = round(len(choices) * train_split)
        #print("Train Size = ", train_size) 80
        test_size = len(choices) - train_size
        #print("Test Size = ", test_size) 20
        # test_grid_flat = []
        # backup_inds = []
        # while len(test_grid_flat) < test_size:
        #     choice = np.random.choice(choices)
        #     #Choice in the loop =  44
        #     choices.remove(choice)
        #     train_grid_flat.remove(choice)
        #     for to_remove in [choice+1, choice-1, choice+x_grid_size, choice-x_grid_size]:
        #         # print("To Remove = ", to_remove)
        #         try:
        #             choices.remove(to_remove)
        #             backup_inds.append(to_remove)
        #             #adding some nearby indices to the backup so that can be used if needed
        #         except:
        #             pass
        #     test_grid_flat.append( choice )
        #     if len(choices) == 0:
        #         ind_need = test_size - len(test_grid_flat)
        #         new_inds = np.random.choice(backup_inds, size=ind_need).tolist()
        #         test_grid_flat += new_inds
        #         for ind in new_inds:
        #             train_grid_flat.remove(ind)
        #         break

        test_split = 1 - train_split
        total_cells = x_grid_size * y_grid_size
        test_cells_needed = int(total_cells * test_split)

        test_grid_flat = []
        train_grid_flat = []
        if edge == 'random':
            all_grids = list(range(total_cells))
            np.random.shuffle(all_grids)
            test_grid_flat = all_grids[:test_cells_needed]
            train_grid_flat = all_grids[test_cells_needed:]

        else:
            # Compute which (x, y) pairs lie on the desired edge
            if edge == 'left':
                edge_columns = list(range(x_grid_size))  # Leftmost columns
                for x in edge_columns:
                    for y in range(y_grid_size):
                        if len(test_grid_flat) < test_cells_needed and x == 0:
                            test_grid_flat.append(y * x_grid_size + x)
                        else:
                            train_grid_flat.append(y * x_grid_size + x)

            elif edge == 'right':
                for x in reversed(range(x_grid_size)):
                    for y in range(y_grid_size):
                        grid_id = y * x_grid_size + x
                        if len(test_grid_flat) < test_cells_needed and x == x_grid_size - 1:
                            test_grid_flat.append(grid_id)
                        else:
                            train_grid_flat.append(grid_id)

            elif edge == 'top':
                for y in range(y_grid_size):
                    for x in range(x_grid_size):
                        grid_id = y * x_grid_size + x
                        if len(test_grid_flat) < test_cells_needed and y == 0:
                            test_grid_flat.append(grid_id)
                        else:
                            train_grid_flat.append(grid_id)

            elif edge == 'bottom':
                for y in reversed(range(y_grid_size)):
                    for x in range(x_grid_size):
                        grid_id = y * x_grid_size + x
                        if len(test_grid_flat) < test_cells_needed and y == y_grid_size - 1:
                            test_grid_flat.append(grid_id)
                        else:
                            train_grid_flat.append(grid_id)
        
        print("train_grid_flat = ", train_grid_flat)
        print("test_grid_flat = ", test_grid_flat)
        
        #Test Grid Flat =  [44, 51, 74, 81, 86, 9, 25, 53, 31, 6, 46, 49, 22, 1, 97, 68, 79, 60, 18, 95]
        #Backup inds =  [45, 43, 54, 34, 52, 50, 61, 41, 75, 73, 84, 64, 82, 80, 91, 71, 87, 85, 96, 76, 10, 8, 19, 26, 24, 35, 15, 63, 32, 30, 21, 7, 5, 16, 47, 56, 36, 48, 59, 39, 23, 12, 2, 0, 11, 98, 69, 67, 78, 58, 89, 70, 17, 28, 94]


        train_inds = []
        test_inds = []
        for ind in inds:
            #loop one all the indices passed
            if len(source.tx_vecs[ind]) == 0: # If there are no transmitters (0 Tx)
                #if no transmitter in the sample, then add wherever you want to
                if np.random.random() <= train_split:
                    train_inds.append(ind)
                else:
                    test_inds.append(ind)
                continue
            tx_vec = source.tx_vecs[ind] # - source.origin
            #Take the tx_position
            x_inds = np.digitize(tx_vec[:,0], x_bins) - 1
            y_inds = np.digitize(tx_vec[:,1], y_bins) - 1
            #x_inds =  [8] - which bins the x coordinate falls in
            #y_inds = [3] - which bins the y coordinate falls in
            grid_inds = np.ravel_multi_index((x_inds, y_inds), (x_grid_size, y_grid_size))
            #grid_inds =  [83] - final grid number based on indices
            # print("grid_inds = ", grid_inds)
            ## I'm not sure if this is still working, with the two separate x and y grid divisions.
            #then look into which grid falls in which set and so assign the respective index in that set
            if all(x in train_grid_flat for x in grid_inds):
                train_inds.append(ind)
            elif all(x not in train_grid_flat for x in grid_inds):
                test_inds.append(ind)
            elif randomly_add_mixed: # If transmitters occur in both train and test grids, randomly assign them to either side
                if np.random.random() >= train_split:
                    train_inds.append(ind)
                else:
                    test_inds.append(ind)
            else: # If we don't add mixed transmitters randomly, just add them to test
                test_inds.append(ind)

        # print(abcd)
        # print("Train inds = ", train_inds) Train inds =  [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18]
        # print("Train Grid Flat = ", train_grid_flat) Train Grid Flat =  [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24]

        return (train_inds, train_grid_flat), (test_inds, test_grid_flat)
        

    def get_data_within_radius(self, point, radius, data_key=None, new_key=None, out_key=None, percent_filtered=1):
        assert (new_key is None and out_key is None) or (new_key is not None and out_key is not None)
        if new_key is None:
            if percent_filtered < 1:
                new_key = '%s_inradius%i_%.1f' % (str(data_key), radius, percent_filtered)
                out_key = '%s_outradius%i_%.1f' % (str(data_key), radius, percent_filtered)
            else:
                new_key = '%s_inradius%i' % (str(data_key), radius)
                out_key = '%s_outradius%i' % (str(data_key), radius)
        assert data_key in self.data
        source = self.data[data_key]
        inds_within_radius = []
        inds_without_radius = []
        for ind, tx_vec in enumerate(source.tx_vecs):
            for tx in tx_vec:
                if np.linalg.norm(tx[:2] - point) <= radius:
                    inds_within_radius.append(ind)
                    break
                else:
                    inds_without_radius.append(ind)
                    break
        if len(source.tx_metadata):
            self.data[new_key] = self.Samples(self, rx_vecs=source.rx_vecs[inds_within_radius], tx_vecs=source.tx_vecs[inds_within_radius], tx_metadata=source.tx_metadata[inds_within_radius], no_shift=True)
            self.data[out_key] = self.Samples(self, rx_vecs=source.rx_vecs[inds_without_radius], tx_vecs=source.tx_vecs[inds_without_radius], tx_metadata=source.tx_metadata[inds_without_radius], no_shift=True)
        else:
            self.data[new_key] = self.Samples(self, rx_vecs=source.rx_vecs[inds_within_radius], tx_vecs=source.tx_vecs[inds_within_radius], no_shift=True)
            self.data[out_key] = self.Samples(self, rx_vecs=source.rx_vecs[inds_without_radius], tx_vecs=source.tx_vecs[inds_without_radius], no_shift=True)
        return new_key, out_key


    def make_datasets(self, split=None, make_val=True, eval_train=False, eval_special=False, train_size=None, should_augment=False, synthetic_only=False, convert_all_inputs=False, sensors_to_remove=[], from_grid_avoid = False):
        #    split= random,make_val=Yes,should_augment=No,convert_all_inputs=True,sensors_to_remove=[empty list]
        params = self.params
        if split==None:
            split = params.data_split
        train_key=None
        test_keys = []
        special_keys = []
        #all are some empty lists
        data_key_prefix = '%.1ftestsize' % params.test_size
        #generating new key with prefix testsize
        if split == 'random' or split == 'random_limited':
            if data_key_prefix + '_train' in self.data.keys(): return '0.2testsize_train', '0.2testsize_test'
            #if key already exists, return the key
            if self.params.dataset_index in [6,8]:
                if 'campus' not in self.data.keys():
                    self.make_filtered_sample_source([coordinates.CAMPUS_POLYGON], 'campus')
                #for our collected dataset, make a key with the data only in campus
                #it will create a new key in dictionary
                #assigning a 'Sample' object with filtered coordinates
                if from_grid_avoid:
                    train_key, test_key = self.separate_random_data(test_size = 0.01, train_size=0.99, data_key_prefix=data_key_prefix, data_source_key='campus', random_state=0) 
                else:
                    train_key, test_key = self.separate_random_data(test_size=params.test_size, train_size=train_size, data_key_prefix=data_key_prefix, data_source_key='campus', random_state=0) 
                # We fix the random state so we always have the same test set, but different validation sets.
            else:
                if from_grid_avoid:
                    train_key, test_key = self.separate_random_data(test_size = 0.01, train_size=0.99, data_key_prefix=data_key_prefix, random_state=0) # We fix the random state so we always have the same test set, but different validation sets.
                else:
                    train_key, test_key = self.separate_random_data(test_size=params.test_size, train_size=train_size, data_key_prefix=data_key_prefix, random_state=0) # We fix the random state so we always have the same test set, but different validation sets.
                #the keys are basically two key in the 'data' dictionary
            if make_val:
                train_key, train_val_key = self.separate_random_data(test_size=params.test_size, train_size=train_size, data_key_prefix=data_key_prefix, data_source_key=train_key, ending_keys=['train', 'train_val'])
                test_keys = [test_key, train_val_key]
                #similarly make validation sets and keep under test_keys
            else:
                test_keys = [test_key]

        elif 'grid' in split:
            if self.params.only_print_propagation_estimation:
                train_random, test_random = self.make_datasets(make_val=False, split='random', from_grid_avoid= True)
            else:
                train_random, test_random = self.make_datasets(make_val=False, split='random')

            print("train_random = ", train_random, " and test_random = ", test_random)

            #it at first calling make_datasets function and asking them for random splitting.
            #this is a recursive loop
            #but at what purpose?
            #Here tx_vecs jumps to 4 values. but why? the rest two values are distance to the other tx and distance to the closest receiver sensor
            # print("Now at grid = ", vars(self.data[train_random]))

            #should_augment is a local variable which is by default false. only activates if you active it locally

            #but why generated train_random and test_random?
            #This is as because, they want this to be in such a way that the test data has data across the region
            #But the train is on grids
            random_state = 0 
            grid_size = int(split.split('grid')[1])
            if grid_size == 3 and self.params.dataset_index == 8 and self.params.use_alt_for_ds8_grid2:
                random_state = 1

            #data_key_prefix =  0.2testsize  train_split =  0.8  train_random =  0.2testsize_train
            if self.params.only_print_propagation_estimation:
                train_key, test_key = self.separate_dataset('grid', grid_size=grid_size, data_key_prefix=data_key_prefix, train_split=0.8, source_key=train_random, random_state=random_state)
                test_keys = [test_key, test_key]

            else:
                train_key, test_key = self.separate_dataset('grid', grid_size=grid_size, data_key_prefix=data_key_prefix, train_split=params.training_size, source_key=train_random, random_state=random_state)
                train_val_key, test_extra_key = self.separate_dataset('grid', grid_size=grid_size, data_key_prefix=data_key_prefix, train_split=params.training_size, source_key=test_random, random_state=random_state, keys=['train_val', 'test_extra'])
                test_keys = [test_key, train_val_key]
                if len(self.data[test_key].rx_vecs) < 100:
                    test_keys.append(test_extra_key)
        elif 'april' in split or 'july' in split or 'nov' in split:
            if 'campus' not in self.data.keys():
                self.make_filtered_sample_source([coordinates.CAMPUS_POLYGON], 'campus')
            source = self.data['campus']
            april_metadata = np.array(['2022-04' in meta[0]['time'] for meta in source.tx_metadata])
            july_metadata = np.array(['2022-07' in meta[0]['time'] for meta in source.tx_metadata])
            nov_metadata = np.array(['2022-11' in meta[0]['time'] for meta in source.tx_metadata])
            april_inds = np.where(april_metadata)
            july_inds = np.where(july_metadata)
            nov_inds = np.where(nov_metadata)
            combined_inds = np.where(april_metadata + july_metadata)
            self.data['april'] = self.Samples(self, rx_vecs=source.rx_vecs[april_inds], tx_vecs=source.tx_vecs[april_inds], tx_metadata=source.tx_metadata[april_inds])
            self.data['july'] = self.Samples(self, rx_vecs=source.rx_vecs[july_inds], tx_vecs=source.tx_vecs[july_inds], tx_metadata=source.tx_metadata[july_inds])
            self.data['nov'] = self.Samples(self, rx_vecs=source.rx_vecs[nov_inds], tx_vecs=source.tx_vecs[nov_inds], tx_metadata=source.tx_metadata[nov_inds])
            if 'combined' in split:
                self.data['combined'] = self.Samples(self, rx_vecs=source.rx_vecs[combined_inds], tx_vecs=source.tx_vecs[combined_inds], tx_metadata=source.tx_metadata[combined_inds])
            if 'april' in split[:5]:
                train_key = 'april'
                test_keys = ['nov'] if 'nov' in split else ['july']
            elif 'july' in split[:4]:
                train_key = 'july'
                test_keys = ['nov'] if 'nov' in split else ['april']
            elif 'nov' in split[:3]:
                train_key = 'nov'
                test_keys = ['april'] if 'april' in split else ['july']

            if 'combined' in split:
                train_key = 'combined'
                test_keys = ['nov', 'april', 'july']

            prefix = train_key
            if 'selftest' in split:
                _, ood_test_key = self.separate_random_data('april', random_state=0, data_key_prefix='april')
                _, ood_test_key = self.separate_random_data(test_keys[0], random_state=0, data_key_prefix=test_keys[0])
                train_key, id_test_key = self.separate_random_data(train_key, random_state=0, data_key_prefix=train_key)
                test_keys = [id_test_key, ood_test_key]
            if make_val:
                train_key, train_val_key = self.separate_random_data(train_key, random_state=params.random_state, data_key_prefix=prefix, ending_keys=['train', 'train_val'])
                test_keys = test_keys + [train_val_key]
            test_key = test_keys[0]
        elif 'driving' in split:
            self.make_filtered_sample_source([coordinates.CAMPUS_POLYGON], 'campus')
            split_keys = ['driving', 'non-driving']
            driving_key, non_driving_key = self.separate_dataset('driving', keys=split_keys, source_key='campus')
            if split == 'driving':
                train_key, test_key = driving_key, non_driving_key
            else:
                train_key, test_key = non_driving_key, driving_key
            if make_val:
                train_key, train_val_key = self.separate_random_data(train_key, random_state=params.random_state, ending_keys=['train', 'train_val'])
                test_keys = [test_key, train_val_key]
            else:
                test_keys = [test_key]
        elif 'biking' in split:
            self.make_filtered_sample_source([coordinates.CAMPUS_POLYGON], 'campus')
            split_keys = ['driving', 'non-driving']
            driving_key, non_driving_key = self.separate_dataset('driving', keys=split_keys, source_key='campus')
            walking_key, biking_key = self.separate_dataset('walking', keys=['walking', 'biking'], source_key=non_driving_key)
            train_key, test_key = biking_key, walking_key
            test_keys = [walking_key, driving_key]
            if make_val:
                train_key, train_val_key = self.separate_random_data(train_key, random_state=params.random_state, ending_keys=['train', 'train_val'])
                test_keys = test_keys + [train_val_key]
        elif 'radius' in split:
            keys = self.make_datasets(make_val=make_val, split='random')
            radius = int(split.split('radius')[1])
            center_point = self.get_center_point()
            train_key, _ = self.get_data_within_radius(center_point, radius, data_key=self.train_key)
            test_keys, train_val_keys = [self.get_data_within_radius(center_point, radius, data_key=key) for key in self.test_keys]
            test_key = test_keys[1]
            test_keys = [test_keys[1], train_val_keys[0]]


        if 'off_campus' == split or eval_special:
            if 'campus' not in self.data.keys():
                self.make_filtered_sample_source([coordinates.campus_polygon], 'campus')
            if '0.2testsize_train' not in self.data.keys():
                self.make_datasets(make_val=False, split='random')
            self.data['off_campus'] = self.make_missing_samples('campus')
            train_key = '0.2testsize_train'
            if eval_special:
                special_keys.append('off_campus')
            else:
                test_key = 'off_campus'
        if 'indoor' == split or eval_special:
            if '0.2testsize_train' not in self.data.keys():
                self.make_datasets(make_val=False, split='random')
            source = self.data[None]
            metadata = np.array([meta[0]['transport'] for meta in source.tx_metadata])
            inds = np.where(metadata == 'inside')
            self.data['indoor'] = self.Samples(self, rx_vecs=source.rx_vecs[inds], tx_vecs=source.tx_vecs[inds], tx_metadata=source.tx_metadata[inds])
            train_key = '0.2testsize_train'
            if eval_special:
                special_keys.append('indoor')
            else:
                test_key = 'indoor'
        if split == '2tx' or eval_special:
            params2 = copy.deepcopy(params)
            params2.make_val = False
            params2.one_tx = False
            params2.data_split = 'random'
            rldataset2 = RSSLocDataset(params2)
            source = rldataset2.data[None]
            tx_lengths = np.array([len(tx_vec) for tx_vec in source.tx_vecs])
            inds = np.where(tx_lengths == 2)
            tx_vecs = np.array(list(source.tx_vecs[inds]))
            self.data['2tx'] = self.Samples(self, rx_vecs=source.rx_vecs[inds], tx_vecs=tx_vecs, tx_metadata=source.tx_metadata[inds])
            if '0.2testsize_train' not in self.data.keys():
                self.make_datasets(make_val=False, split='random')
            train_key = '0.2testsize_train'
            if eval_special:
                special_keys.append('2tx')
            else:
                test_key = '2tx'

        if eval_train:
            test_keys.append(train_key)

        #Till now, y_vecs is fine with two entries only
        

        if len(test_keys) > 0:
            self.set_default_keys(train_key=train_key, test_keys=test_keys + special_keys)
        else:
            self.set_default_keys(train_key=train_key, test_key=test_key)

        #Here y_vecs changes. Something to do with set_default_keys function
        #actually make_tensor() function withing it
        #Two values. the first one is the difference with the other tx in your sample and the other one with the rx
        # print("The data Now = ", vars(self.data[train_key]))
        # print(abcd)

        #should_augment is false. but why?

        if should_augment and params.augmentation is not None:
            print("train_key = ", train_key, " test_keys= ", test_keys)
    
            #add synthetic training data if needed
            # Must indicate if sensors should be removed BEFORE training the augmentor object
            self.sensors_to_remove = sensors_to_remove
            #Synthetic Only =  False  convert all inputs =  True
            if self.params.only_print_propagation_estimation:
                self.add_synthetic_training_data_just_print(synthetic_only=synthetic_only, convert_all_inputs=convert_all_inputs)        
            else:
                self.add_synthetic_training_data(synthetic_only=synthetic_only, convert_all_inputs=convert_all_inputs)
            #self.nonlearning_localization()
        # Remove sensors from train data for experiments on adding new devices at inference time

        # if should_augment and params.tirem_augment_two_tx:
        #     self.add_two_tx_tirem_augmentation()

        if params.introduce_new_randomization:
            self.introduce_new_randomization()
        if sensors_to_remove:
        #mute sensors if needed
            for sensor in sensors_to_remove:
                self.mute_inputs_in_data_key(train_key, sensor)

        return train_key, test_key

    
    def mute_inputs_in_data_key(self, key, sensor_id, mute_synthetic=False):
        dataset = self.data[key].ordered_dataloader.dataset
        if mute_synthetic:
            dataset.tensors[0][:,sensor_id] = 0
        else:
            mute_length = len(self.data[key].rx_vecs)
            dataset.tensors[0][:mute_length,sensor_id] = 0


    def nonlearning_localization(self):
        rx_data_tr = self.data[self.train_key].ordered_dataloader.dataset.tensors[0]
        tx_locs_tr = self.data[self.train_key].ordered_dataloader.dataset.tensors[1][:,:,1:]
        rx_data_tv = self.data[self.test_keys[1]].ordered_dataloader.dataset.tensors[0]
        tx_locs_tv = self.data[self.test_keys[1]].ordered_dataloader.dataset.tensors[1][:,:,1:]
        rx_data_te = self.data[self.test_keys[0]].ordered_dataloader.dataset.tensors[0]
        tx_locs_te = self.data[self.test_keys[0]].ordered_dataloader.dataset.tensors[1][:,:,1:]
        keys = self.prop_keys
        eps = 0.05
        tr_err = np.zeros(300) #len(tx_locs_tr))
        tv_err = np.zeros(300) #len(tx_locs_tv))
        te_err = np.zeros(300) #len(tx_locs_te))
        tr_err1 = [np.zeros(300) for i in range(5)] #len(tx_locs_tr))
        tv_err1 = [np.zeros(300) for i in range(5)] #len(tx_locs_tr))
        te_err1 = [np.zeros(300) for i in range(5)] #len(tx_locs_tr))
        #tr_err1 = np.zeros(300) #len(tx_locs_tr))
        #tv_err1 = np.zeros(300) #len(tx_locs_tv))
        #te_err1 = np.zeros(300) #len(tx_locs_te))
        for rx_set, tx_set, err_set, err_set1 in zip(
            [rx_data_tr, rx_data_tv, rx_data_te],
            [tx_locs_tr, tx_locs_tv, tx_locs_te],
            [tr_err, tv_err, te_err],
            [tr_err1, tv_err1, te_err1],
        ):
            for i in range(300): #len(tx_set)):
                input = rx_set[i,:,0]
                inp = input[keys][input[keys].cpu().numpy() > 0].cpu().numpy()
                tx = tx_set[i,0].cpu().numpy()
                inp_maps = self.prop_maps[input[keys].cpu() > 0].cpu().numpy()
                inp_maps[np.isnan(inp_maps)] = 0
                pred_maps = (inp_maps >= inp[:,None,None] - eps) * (inp_maps <= inp[:,None,None] + eps)
                pred_map = pred_maps.sum(axis=0)
                err = np.linalg.norm(np.fliplr(np.array(np.where(pred_map == pred_map.max())).T) - tx, axis=1).min() * self.params.meter_scale
                err_set[i] = err
                pred_maps = (1 - abs(inp_maps - inp[:,None,None])) * inp[:,None,None]
                pred_map = pred_maps.sum(axis=0)
                err = np.linalg.norm(np.fliplr(np.array(np.where(pred_map == pred_map.max())).T) - tx, axis=1).min() * self.params.meter_scale
                err_set1[1][i] = err
        print(tr_err.mean(), tv_err.mean(), te_err.mean())
        print(tr_err1[1].mean(), tv_err1[1].mean(), te_err1[1].mean())
        #print(np.median(tr_err), np.median(tv_err), np.median(te_err), np.median(tr_err1), np.median(tv_err1), np.median(te_err1))
        embed()
        exit
    
    def add_two_tx_tirem_augmentation(self):
        from tirem_two_transmitters_augmentation import two_transmitter_augmented_data, two_transmitter_augmented_data_new_locations
        if self.params.train_two_tx_tirem_with_ood_samples:
            ood_augmented_two_tx, ood_augmented_two_tx_rx = two_transmitter_augmented_data_new_locations(self)
        # print("OOD Augmented y = ", ood_augmented_two_tx[:5])
        # print("OOD Augmented X = ", ood_augmented_two_tx_rx[:5])

        num_aug_samples = self.params.amount_aug_samples
        
        augmented_two_tx, augmented_two_tx_rx = two_transmitter_augmented_data(self, num_samples=num_aug_samples)
        print("first y shape = ",augmented_two_tx.shape, " first X shape = ", augmented_two_tx_rx.shape)
        # print(abcd)

        key = self.train_key
        train_x, train_y = self.data[key].ordered_dataloader.dataset.tensors[:2]

        print("Test Keys = ", self.test_keys)
        print("Train x shape = ", train_x.shape)
        print("Train y shape = ", train_y.shape, " and two transmitters = ", torch.sum(torch.sum(train_y[:, :, 0], dim=1) == 2).item())

        test_x, test_y = self.data[self.test_keys[0]].ordered_dataloader.dataset.tensors[:2]
        print("Test x shape = ", test_x.shape)
        print("Test y shape = ", test_y.shape, " and two transmitters = ", torch.sum(torch.sum(test_y[:, :, 0], dim=1) == 2).item())

        val_x, val_y = self.data[self.test_keys[1]].ordered_dataloader.dataset.tensors[:2]
        print("Validation x shape = ", val_x.shape)
        print("Validation y shape = ", val_y.shape, " and two transmitters = ", torch.sum(torch.sum(val_y[:, :, 0], dim=1) == 2).item())


        pin_memory = self.params.device != torch.device('cuda') 
        if self.params.tirem_augment_two_tx and self.params.tirem_two_tx_only_synthetic_on_train and not self.params.train_two_tx_tirem_with_ood_samples:
            augmented_two_tx_tensor = torch.tensor(np.array(augmented_two_tx), device=self.params.device)
            augmented_two_tx_rx_tensor = torch.tensor(np.array(augmented_two_tx_rx), device=self.params.device)

            # Split augmented data into 85% train and 15% validation
            split_index = int(0.85 * len(augmented_two_tx_tensor))
            aug_train_x = augmented_two_tx_rx_tensor[:split_index].to(train_x.dtype)
            aug_train_y = augmented_two_tx_tensor[:split_index].to(train_y.dtype)
            aug_val_x = augmented_two_tx_rx_tensor[split_index:].to(val_x.dtype)
            aug_val_y = augmented_two_tx_tensor[split_index:].to(val_y.dtype)

            print("Before Processing:")
            print(f"Train: {train_x.shape}, {train_y.shape}")
            print(f"Test: {test_x.shape}, {test_y.shape}")
            print(f"Validation: {val_x.shape}, {val_y.shape}")

            # Find two transmitter samples
            two_tx_train_mask = torch.sum(train_y[:, :, 0], dim=1) == 2
            two_tx_val_mask = torch.sum(val_y[:, :, 0], dim=1) == 2           

            # Extract two transmitter samples and add to the test set
            test_x = torch.cat([test_x, train_x[two_tx_train_mask], val_x[two_tx_val_mask]], dim=0)
            test_y = torch.cat([test_y, train_y[two_tx_train_mask], val_y[two_tx_val_mask]], dim=0)

            # Remove two transmitter samples from train and validation
            train_x = train_x[~two_tx_train_mask]
            train_y = train_y[~two_tx_train_mask]
            val_x = val_x[~two_tx_val_mask]
            val_y = val_y[~two_tx_val_mask]

            # Add augmented data to train and validation
            train_x = torch.cat([train_x, aug_train_x], dim=0)
            train_y = torch.cat([train_y, aug_train_y], dim=0)
            val_x = torch.cat([val_x, aug_val_x], dim=0)
            val_y = torch.cat([val_y, aug_val_y], dim=0)

            print("After Processing:")
            print(f"Train: {train_x.shape}, {train_y.shape}")
            print(f"Test: {test_x.shape}, {test_y.shape}")
            print(f"Validation: {val_x.shape}, {val_y.shape}")

            # print("Train y shape = ", train_y.shape, " and two transmitters = ", torch.sum(torch.sum(train_y[:, :, 0], dim=1) == 2).item())
            # print("Train data samples: ", train_y[:15])
            # print(abcd)

            dataset = torch.utils.data.TensorDataset(train_x, train_y)
            # key = self.train_key
            self.data[self.train_key+'_augmented'] = copy.copy(self.data[self.train_key])
            self.data[self.train_key+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.train_key+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
            self.train_key_augmented = self.train_key + '_augmented'

            self.test_keys_augmented = [key + '_augmented' for key in self.test_keys]


            dataset = torch.utils.data.TensorDataset(test_x, test_y)
            # key = self.train_key
            self.data[self.test_keys[0]+'_augmented'] = copy.copy(self.data[self.test_keys[0]])
            self.data[self.test_keys[0]+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.test_keys[0]+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
            # self.train_key_augmented = self.train_key + '_augmented'

            dataset = torch.utils.data.TensorDataset(val_x, val_y)
            # key = self.train_key
            self.data[self.test_keys[1]+'_augmented'] = copy.copy(self.data[self.test_keys[1]])
            self.data[self.test_keys[1]+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.test_keys[1]+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
            # self.train_key_augmented = self.train_key + '_augmented'

            print("Test keys for augmented data:", self.test_keys_augmented)

        elif self.params.tirem_augment_two_tx and not self.params.tirem_two_tx_only_synthetic_on_train and self.params.tirem_two_tx_specific_augmentation and not self.params.train_two_tx_tirem_with_ood_samples:
            original_ratio = self.params.original_ratio

            augmented_two_tx_tensor = torch.tensor(np.array(augmented_two_tx), device=self.params.device)
            augmented_two_tx_rx_tensor = torch.tensor(np.array(augmented_two_tx_rx), device=self.params.device)

            # Find two transmitter samples in the train set
            two_tx_train_mask = torch.sum(train_y[:, :, 0], dim=1) == 2            


            # Extract two transmitter data
            two_tx_train_x = train_x[two_tx_train_mask]
            two_tx_train_y = train_y[two_tx_train_mask]

            # Calculate the number of samples to keep in the train set
            keep_count = int(self.params.original_ratio * len(two_tx_train_x))

            # Keep original_ratio amount in train set, rest goes to test set
            if keep_count > 0:
                train_x = torch.cat([train_x[~two_tx_train_mask], two_tx_train_x[:keep_count]], dim=0)
                train_y = torch.cat([train_y[~two_tx_train_mask], two_tx_train_y[:keep_count]], dim=0)
            else:
                train_x = train_x[~two_tx_train_mask]
                train_y = train_y[~two_tx_train_mask]

            if keep_count < len(two_tx_train_x):
                test_x = torch.cat([test_x, two_tx_train_x[keep_count:]], dim=0)
                test_y = torch.cat([test_y, two_tx_train_y[keep_count:]], dim=0)

            # Add augmented data to the train set
            train_x = torch.cat([train_x, augmented_two_tx_rx_tensor.to(train_x.dtype)], dim=0)
            train_y = torch.cat([train_y, augmented_two_tx_tensor.to(train_y.dtype)], dim=0)

            print("After Processing:")
            print(f"Train: {train_x.shape}, {train_y.shape}")
            print(f"Test: {test_x.shape}, {test_y.shape}")
            print(f"Validation: {val_x.shape}, {val_y.shape}")

            # Create datasets and data loaders
            dataset = torch.utils.data.TensorDataset(train_x, train_y)
            self.data[self.train_key+'_augmented'] = copy.copy(self.data[self.train_key])
            self.data[self.train_key+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.train_key+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
            self.train_key_augmented = self.train_key + '_augmented'       

            self.test_keys_augmented = [key + '_augmented' for key in self.test_keys]

            dataset = torch.utils.data.TensorDataset(test_x, test_y)
            self.data[self.test_keys[0]+'_augmented'] = copy.copy(self.data[self.test_keys[0]])
            self.data[self.test_keys[0]+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.test_keys[0]+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)

            dataset = torch.utils.data.TensorDataset(val_x, val_y)
            self.data[self.test_keys[1]+'_augmented'] = copy.copy(self.data[self.test_keys[1]])
            self.data[self.test_keys[1]+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.test_keys[1]+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)

            print("Test keys for augmented data:", self.test_keys_augmented)    

        elif self.params.tirem_augment_two_tx and not self.params.tirem_two_tx_only_synthetic_on_train and self.params.train_two_tx_tirem_with_ood_samples:
            original_ratio = self.params.original_ratio

            augmented_two_tx_tensor = torch.tensor(np.array(augmented_two_tx), device=self.params.device)
            augmented_two_tx_rx_tensor = torch.tensor(np.array(augmented_two_tx_rx), device=self.params.device)

            augmented_two_tx_rx_tensor = augmented_two_tx_rx_tensor.float()
            augmented_two_tx_tensor = augmented_two_tx_tensor.float()

            ood_augmented_two_tx_tensor = torch.tensor(np.array(ood_augmented_two_tx), device=self.params.device)
            ood_augmented_two_tx_rx_tensor = torch.tensor(np.array(ood_augmented_two_tx_rx), device=self.params.device)

            ood_augmented_two_tx_rx_tensor = ood_augmented_two_tx_rx_tensor.float()
            ood_augmented_two_tx_tensor = ood_augmented_two_tx_tensor.float()

            # print(f"previous augmentation shape: {ood_augmented_two_tx_tensor.shape}, dtype: {ood_augmented_two_tx_tensor.dtype}, device: {ood_augmented_two_tx_tensor.device}")
            # print(f"previous augmentation shape: {augmented_two_tx_rx_tensor.shape}, dtype: {augmented_two_tx_rx_tensor.dtype}, device: {augmented_two_tx_rx_tensor.device}")

            # print(f"new augmentation shape: {augmented_two_tx_tensor.shape}, dtype: {augmented_two_tx_tensor.dtype}, device: {augmented_two_tx_tensor.device}")
            # print(f"new augmentation shape: {ood_augmented_two_tx_rx_tensor.shape}, dtype: {ood_augmented_two_tx_rx_tensor.dtype}, device: {ood_augmented_two_tx_rx_tensor.device}")
            # print(abcd)


            # Add augmented data to the train set
            train_x = torch.cat([train_x, ood_augmented_two_tx_rx_tensor.to(train_x.dtype).float()], dim=0)
            train_y = torch.cat([train_y, ood_augmented_two_tx_tensor.to(train_y.dtype).float()], dim=0)   

            print(f"train_x shape: {train_x.shape}, ood_augmented_two_tx_rx_tensor shape: {ood_augmented_two_tx_rx_tensor.shape}")
            print(f"train_y shape: {train_y.shape}, ood_augmented_two_tx_tensor shape: {ood_augmented_two_tx_tensor.shape}")

            print(f"train_x dtype: {train_x.dtype}, ood_augmented_two_tx_rx_tensor dtype: {ood_augmented_two_tx_rx_tensor.dtype}")
            print(f"train_y dtype: {train_y.dtype}, ood_augmented_two_tx_tensor dtype: {ood_augmented_two_tx_tensor.dtype}")

            print(f"train_x device: {train_x.device}, ood_augmented_two_tx_rx_tensor device: {ood_augmented_two_tx_rx_tensor.device}")
            print(f"train_y device: {train_y.device}, ood_augmented_two_tx_tensor device: {ood_augmented_two_tx_tensor.device}")

            # Create datasets and data loaders
            dataset = torch.utils.data.TensorDataset(train_x, train_y)
            self.data[self.train_key+'_augmented'] = copy.copy(self.data[self.train_key])
            self.data[self.train_key+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.train_key+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
            self.train_key_augmented = self.train_key + '_augmented'   


            self.test_keys_augmented = [key + '_augmented' for key in self.test_keys]
            self.test_keys_augmented.append(self.test_keys[0] + '_augmented_onlySynthetic')

            dataset = torch.utils.data.TensorDataset(test_x, test_y)
            self.data[self.test_keys[0]+'_augmented'] = copy.copy(self.data[self.test_keys[0]])
            self.data[self.test_keys[0]+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.test_keys[0]+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)

            dataset = torch.utils.data.TensorDataset(val_x, val_y)
            self.data[self.test_keys[1]+'_augmented'] = copy.copy(self.data[self.test_keys[1]])
            self.data[self.test_keys[1]+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.test_keys[1]+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)

            random_indices = np.random.choice(augmented_two_tx_tensor.shape[0], 500, replace=False)

            # Select 500 random samples
            augmented_two_tx_tensor = augmented_two_tx_tensor[random_indices]
            augmented_two_tx_rx_tensor = augmented_two_tx_rx_tensor[random_indices]

            print ("Augmented clipped y = ", augmented_two_tx_tensor.shape, " and X = ", augmented_two_tx_rx_tensor.shape)
            # print(abcd)

            dataset = torch.utils.data.TensorDataset(augmented_two_tx_rx_tensor, augmented_two_tx_tensor)
            self.data[self.test_keys[0]+'_augmented_onlySynthetic'] = copy.copy(self.data[self.test_keys[0]])
            self.data[self.test_keys[0]+'_augmented_onlySynthetic'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[self.test_keys[0]+'_augmented_onlySynthetic'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)

            print("Test keys for augmented data:", self.test_keys_augmented) 

        else:
            
            augmented_two_tx = torch.tensor(np.array(augmented_two_tx), device=self.params.device).to(train_y.dtype)
            augmented_two_tx_rx = torch.tensor(np.array(augmented_two_tx_rx), device=self.params.device).to(train_x.dtype)


            accumulated_train_x = torch.cat([train_x, augmented_two_tx_rx], dim=0)
            accumulated_train_y = torch.cat([train_y, augmented_two_tx], dim=0)


            dataset = torch.utils.data.TensorDataset(accumulated_train_x, accumulated_train_y)
            key = self.train_key
            self.data[key+'_augmented'] = copy.copy(self.data[key])
            self.data[key+'_augmented'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[key+'_augmented'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
            self.train_key_augmented = self.train_key + '_augmented'

            print("Augmented X = ", accumulated_train_x.shape," and Augmented Y = ", accumulated_train_y.shape," and the new key = ", self.train_key_augmented)
            # print(" Augmented Y sample = ", augmented_two_tx[:5])
            # print(" Accumulated Y sample = ", accumulated_train_y[:5])
        # print(abcd)

    def introduce_new_randomization(self):
        key = self.train_key
        train_x, train_y = self.data[key].ordered_dataloader.dataset.tensors[:2]
        two_tx_train_mask = torch.sum(train_y[:, :, 0], dim=1) == 2

        two_tx_train_x = train_x[two_tx_train_mask]
        two_tx_train_y = train_y[two_tx_train_mask]

        keep_count = int(0.60 * len(two_tx_train_x))

        train_x = torch.cat([train_x[~two_tx_train_mask], two_tx_train_x[:keep_count]], dim=0)
        train_y = torch.cat([train_y[~two_tx_train_mask], two_tx_train_y[:keep_count]], dim=0)

        pin_memory = self.params.device != torch.device('cuda') 
        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        # key = self.train_key
        self.data[self.train_key] = copy.copy(self.data[self.train_key])
        self.data[self.train_key].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
        self.data[self.train_key].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
        # self.train_key_augmented = self.train_key + '_augmented'




    def add_two_tx_tirem_augmentation_old(self):
        # from tirem_two_transmitters_augmentation import two_transmitter_augmented_data
        # two_transmitter_augmented_data(self)

        from synthetic_augmentations import SyntheticAugmentor
        method = 'tirem_nn'
        method = self.params.augmentation if method is None else method
        if method not in ['linear', 'nearest', 'rbf', 'tirem', 'tirem_nn', 'celf']:
            return
        augmentor = SyntheticAugmentor(method=method, rldataset=self)
        augmentor.train()
        prop_results = augmentor.get_results()
        prop_maps, keys, rx_locs, rx_types, rx_test_preds, rx_test_rss, tx_test_coords = prop_results
        # print("Shape of Propagation Map = ",prop_maps.shape) #(23, 89, 97)
        #so this propagation map is actually storing all the map values 
        #For each of the map coordinate position for each sensor

        print("Rx Locs = ",rx_locs)
        print(abcd)
        
        keys = [key[0] for key in keys]
        # print(keys) [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 25, 26, 27, 32, 33, 35, 36]
 
        if self.params.dataset_index == 6:
            rm13 = keys.index(13)
            keys.pop(rm13)
            rx_locs.pop(rm13)
            rx_types.pop(rm13)
            rm13_mask = np.ones(len(prop_maps), dtype=bool)
            rm13_mask[rm13] = False
            prop_maps = prop_maps[rm13_mask]

        train_data = self.data[self.train_key]
        tx_data = train_data.ordered_dataloader.dataset.tensors[1]
        tx_locs = tx_data[:,:,1:]
        rx_data = train_data.ordered_dataloader.dataset.tensors[0]
        self.prop_maps = torch.tensor(prop_maps, dtype=torch.float32, device=self.params.device)
        grid_bounds = [[1,self.img_width()-1], [1,self.img_height()-1]]
        # print("Grid Bounds = ", grid_bounds) #Grid Bounds =  [[1, 96], [1, 88]]
        if self.params.dataset_index == 6:
            grid_bounds = [[3,70], [15,self.img_height()]]
        grid = np.array(np.meshgrid(np.arange(grid_bounds[0][0],grid_bounds[0][1]), np.arange(grid_bounds[1][0],grid_bounds[1][1]))).reshape(2,-1).T 
        # print("Grid = ",grid)
        # print("Grid shape = ",grid.shape) (4958, 2)
        synthetic_only = False
        synthetic_sample_distance_in_pixels=1
        tx_locs = tx_locs[:,0,:]
        # print("Tx Locs shape = ", tx_locs.shape)torch.Size([3023, 2])
        if synthetic_only:
            new_tx_locs = grid
        else:
            nearest_tx = cdist(tx_locs.squeeze().cpu(), grid+0.5).min(axis=0)
            new_tx_locs = grid[nearest_tx > synthetic_sample_distance_in_pixels]
        new_rss = self.prop_maps[:, new_tx_locs[:,1], new_tx_locs[:,0]].T

        # print("New Tx Locs shape  = ", new_tx_locs.shape) (2654, 2)
        # print("New RSS  shape = ", new_rss.shape) torch.Size([2654, 22])

        new_rss = new_rss.maximum(torch.zeros((1,), device=self.params.device))
        self.prop_keys = keys
        new_tx = torch.zeros((len(new_tx_locs), tx_locs.shape[1], 3), device=self.params.device)
        new_rx = torch.zeros((len(new_tx_locs), rx_data.shape[1], rx_data.shape[2]), device=self.params.device)
        new_tx[:,:,0] = 1
        new_tx[:,:,1:3] = torch.tensor(new_tx_locs, dtype=torch.float32, device=self.params.device).unsqueeze(1)
        new_rx[:,keys,0] = new_rss
        new_rx[:,keys,1:3] = torch.tensor(np.array(rx_locs), device=self.params.device)
        new_rx[:,keys,3] = torch.tensor(keys, dtype=torch.float32, device=self.params.device)
        new_rx[:,keys,4] = torch.tensor(rx_types, device=self.params.device)
        pin_memory = self.params.device != torch.device('cuda') 

        # print("New Tx shape = ", new_tx.shape)  torch.Size([2654, 2, 3])
        # print("New Rx shape = ", new_rx.shape) torch.Size([2654, 40, 5])
        # print(abcd)

        convert_all_inputs=False
        if convert_all_inputs:
            key = self.train_key
            #for key in [self.train_key] + self.test_keys[1:]:
            self.data[key+'_synthetic'] = copy.copy(self.data[key])
            tx_data = self.data[key+'_synthetic'].ordered_dataloader.dataset.tensors[1]
            tx_locs = tx_data[:,0,1:].round().int()
            rx_data = self.data[key+'_synthetic'].ordered_dataloader.dataset.tensors[0]
            pred_rss = self.prop_maps[:, tx_locs[:,1], tx_locs[:,0]].T
            pred_rss = pred_rss.maximum(torch.zeros((1,), device=self.params.device))
            zero_rss_inds = rx_data[:,keys,1] == 0
            rx_data[:,keys,0][zero_rss_inds] = pred_rss[zero_rss_inds]
            rx_data[:,keys,1:3] = torch.tensor(np.array(rx_locs), device=self.params.device)
            rx_data[:,keys,3] = torch.tensor(keys, dtype=torch.float32, device=self.params.device)
            rx_data[:,keys,4] = torch.tensor(rx_types, device=self.params.device)
            if key == self.train_key:
                train_rx, train_tx = rx_data, tx_data
            #self.test_keys = [self.test_keys[0]] + [key+'_synthetic' for key in self.test_keys]
        else:
            train_rx, train_tx = rx_data, tx_data

        if synthetic_only:
            rx, tx = new_rx, new_tx
        else:
            rx, tx = torch.cat((train_rx, new_rx)), torch.cat((train_tx, new_tx))
        dataset = torch.utils.data.TensorDataset(rx, tx)
        key = self.train_key
        self.data[key+'_synthetic'] = copy.copy(self.data[key])
        self.data[key+'_synthetic'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
        self.data[key+'_synthetic'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
        self.train_key = self.train_key + '_synthetic'
        if method == 'fusion':
            rx_data = self.data[self.train_key].ordered_dataloader.dataset.tensors[0]
            tx_data = self.data[self.train_key].ordered_dataloader.dataset.tensors[1]
            coords = tx_data[:,0,1:].round().int().cpu()
            self.synthetic_rss = torch.tensor(self.all_prop_maps[:,:,coords[-len(new_rx):,1], coords[-len(new_rx):,0]]).to(self.params.device)

    def add_synthetic_training_data(self, method=None, synthetic_only=False, convert_all_inputs=True, synthetic_sample_distance_in_pixels=1):

        #Train Key =  0.2testsizegrid5_train
        #Test Key =  ['0.2testsizegrid5_test', '0.2testsizegrid5_train_val']
        from synthetic_augmentations import SyntheticAugmentor
        method = self.params.augmentation if method is None else method
        if method not in ['linear', 'nearest', 'rbf', 'tirem', 'tirem_nn', 'celf']:
            return

        # in_built = True
        # if not in_built:
        augmentor = SyntheticAugmentor(method=method, rldataset=self)
        augmentor.train()
        prop_results = augmentor.get_results()
        prop_maps, keys, rx_locs, rx_types, rx_test_preds, rx_test_rss, tx_test_coords = prop_results


        keys = [key[0] for key in keys]


        #Propagation Maps shape = Prop maps =  (23, 89, 97)
        #Keys are all the receivers: Keys =  [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 25, 26, 27, 32, 33, 35, 36]
        #There are in total 23 receivers and so the first dimension of the prop_maps is 23
        #The location of all the receivers: Rx Locs =  [array([60.24944 , 16.842804], dtype=float32), array([45.488373, 83.37936 ], dtype=float32), array([58.174618, 75.43506 ], dtype=float32), array([58.331776, 45.90307 ], dtype=float32), array([46.82482 , 64.281166], dtype=float32), array([22.11929, 54.5548 ], dtype=float32), array([79.80298 , 38.505154], dtype=float32), array([22.534323, 45.09213 ], dtype=float32), array([41.259506, 33.099636], dtype=float32), array([78.0729  , 59.037415], dtype=float32), array([37.793865, 75.748314], dtype=float32), array([54.521755, 60.593662], dtype=float32), array([52.731426, 71.090775], dtype=float32), array([29.380293, 61.767452], dtype=float32), array([37.419796, 48.146   ], dtype=float32), array([55.702114, 71.062386], dtype=float32), array([ 8.042728, 33.95312 ], dtype=float32), array([54.7031  , 58.371593], dtype=float32), array([42.725594, 66.25744 ], dtype=float32), array([61.50944 , 60.482586], dtype=float32), array([ 3.1636274, 18.324903 ], dtype=float32), array([26.803314, 32.617012], dtype=float32), array([60.672882, 54.184803], dtype=float32)]
        #Type of each receivers: Rx Types =  [4.0, 2.0, 3.0, 3.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, 4.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0]
        # print(abcd)

        if self.params.dataset_index == 6:
            rm13 = keys.index(13)
            #Find the index of key 13 in the list keys.
            #13 is the 10th element (Python indexing starts at 0).
            #rm13 = 9
            keys.pop(rm13)
            #Remove the key at index 9, which is 13.
            #(22 keys now).
            rx_locs.pop(rm13)
            #Remove the receiver location corresponding to key 13.
            rx_types.pop(rm13)
            #Remove the receiver type for key 13.
            rm13_mask = np.ones(len(prop_maps), dtype=bool)
            rm13_mask[rm13] = False
            #Create a boolean mask rm13_mask, of length 23 (len(prop_maps)).
            #Then set the 9th index (index=9) to False.
            prop_maps = prop_maps[rm13_mask]
            #After: prop_maps.shape = (22, 89, 97)

        train_data = self.data[self.train_key]
        if self.params.one_tx:
            #self.train_key tells which dataset to load (for example, 'train'
            tx_data = train_data.ordered_dataloader.dataset.tensors[1]
            #train_data now holds your original training dataset (with TX and RX tensors inside).
            tx_locs = tx_data[:,:,1:]
            #tx_locs hold all the locations of the transmitters in the train set
            rx_data = train_data.ordered_dataloader.dataset.tensors[0]
            #rx_data holds necessary informations
        else: 
            tx_data = train_data.ordered_dataloader.dataset.tensors[1]
            rx_data = train_data.ordered_dataloader.dataset.tensors[0]
            # Step 1: Mask to filter only samples with exactly one TX active
            tx_flags = tx_data[:, :, 0]  # shape: (N, 2)
            one_tx_mask = (tx_flags == 1).sum(dim=1) == 1  # shape: (N,)

            # Step 2: Apply the same mask to both tx_data and rx_data
            tx_data = tx_data[one_tx_mask]
            rx_data = rx_data[one_tx_mask]

            # Step 3: Get active TX coordinates, shape → [N, 1, 2]
            tx_coords = tx_data[:, :, 1:]  # shape: (N, 2, 2)
            tx_flags = tx_data[:, :, 0]
            active_tx_index = tx_flags.argmax(dim=1)  # (N,)
            tx_locs = torch.gather(tx_coords, 1, active_tx_index.view(-1, 1, 1).expand(-1, 1, 2))

        # print("Tx Data shape = ", tx_data.shape)
        # print("Tx locs sample = ", tx_locs[50:70])
        self.prop_maps = torch.tensor(prop_maps, dtype=torch.float32, device=self.params.device)
        #Here we are making the propagation map a tensor
        grid_bounds = [[1,self.img_width()-1], [1,self.img_height()-1]]
        #Defining a rectangular area excluding the border
        if self.params.dataset_index == 6:
            grid_bounds = [[3,70], [15,self.img_height()]]
        #Custom grid bound for the 6th dataset
        grid = np.array(np.meshgrid(np.arange(grid_bounds[0][0],grid_bounds[0][1]), np.arange(grid_bounds[1][0],grid_bounds[1][1]))).reshape(2,-1).T 
        #At first made a 2D array of grids
        #Then reshaped to two points denoting (x,y) coordinate of each grid location
        
        if self.params.synthetic_only:
            new_tx_locs = grid
        #If we’re generating only synthetic data, then:
        # #Use all grid points as new transmitter locations.
        else:
            nearest_tx = cdist(tx_locs.squeeze().cpu(), grid+0.5).min(axis=0)
            #For each grid point, find closest distance to a real TX.
            #cdist(A, B) computes Euclidean distance between each point in A and each point in B.
            #You're adding 0.5 to grid — probably to center the (x,y) coordinate inside the pixel (from corner to center)
            new_tx_locs = grid[nearest_tx > synthetic_sample_distance_in_pixels]
            #Keep the grids that are faraway from the current tx locations in the train set
        new_rss = self.prop_maps[:, new_tx_locs[:,1], new_tx_locs[:,0]].T
        #For each new tx, get the predicted RSS value for each of the receivers
        new_rss = new_rss.maximum(torch.zeros((1,), device=self.params.device))
        #Ensuring new rss values to be non-negative
        self.prop_keys = keys
        #copying receiver IDs
        if self.params.one_tx:
            new_tx = torch.zeros((len(new_tx_locs), tx_locs.shape[1], 3), device=self.params.device)
        else:
            new_tx = torch.zeros((len(new_tx_locs), tx_locs.shape[1] + 1, 3), device=self.params.device)
        #making a new_tx tensor of shape (total_sample, num_tx(1), 3)
        new_rx = torch.zeros((len(new_tx_locs), rx_data.shape[1], rx_data.shape[2]), device=self.params.device)
        #Similarly new receiver tensor of shape (total_sample, number_of_receivers, five_respective_values)
        if self.params.one_tx:
            new_tx[:,:,0] = 1
            new_tx[:,:,1:3] = torch.tensor(new_tx_locs, dtype=torch.float32, device=self.params.device).unsqueeze(1)
        else:
            new_tx[:,0,0] = 1
            new_tx[:,0,1:3] = torch.tensor(new_tx_locs, dtype=torch.float32, device=self.params.device)
            new_tx[:,1,1:3] = 0.0
        new_rx[:,keys,0] = new_rss
        new_rx[:,keys,1:3] = torch.tensor(np.array(rx_locs), device=self.params.device)
        new_rx[:,keys,3] = torch.tensor(keys, dtype=torch.float32, device=self.params.device)
        new_rx[:,keys,4] = torch.tensor(rx_types, device=self.params.device)
        #assigning all the values
        pin_memory = self.params.device != torch.device('cuda') 
        #to speed up if not running on GPUs


        if convert_all_inputs:
            #If convert_all_inputs=True, you're being told to augment the original dataset too, using propagation predictions.
            key = self.train_key
            #Stores the current training key ('train' or 'train_original') into key.
            #for key in [self.train_key] + self.test_keys[1:]:
            self.data[key+'_synthetic'] = copy.copy(self.data[key])
            #Create a copy of the current dataset with a new key like 'train_synthetic'.
            # tx_data = self.data[key+'_synthetic'].ordered_dataloader.dataset.tensors[1]
            # #Extract transmitter tensor from the copied synthetic data.
            # tx_locs = tx_data[:,0,1:].round().int()
            # #Extract x and y coordinates, round them to nearest integer (to index into prop_maps).
            # rx_data = self.data[key+'_synthetic'].ordered_dataloader.dataset.tensors[0]
            #Extract receiver data for this training set.

            tx_data = self.data[key+'_synthetic'].ordered_dataloader.dataset.tensors[1]
            rx_data = self.data[key+'_synthetic'].ordered_dataloader.dataset.tensors[0]   

            print("Primary case = ", tx_data.shape)         

            if self.params.one_tx:
                tx_locs = tx_data[:, 0, 1:].round().int()  # Use all one-TX directly
            else:
                # Only use samples that are one-TX to apply propagation maps
                tx_flags = tx_data[:, :, 0]
                one_tx_mask = (tx_flags == 1).sum(dim=1) == 1
                tx_data = tx_data[one_tx_mask]
                rx_data = rx_data[one_tx_mask]

                tx_locs = tx_data[:, 0, 1:].round().int()  # Use all one-TX directly

            pred_rss = self.prop_maps[:, tx_locs[:,1], tx_locs[:,0]].T
            #Use prop_maps to predict RSS at all receivers for each of these TX locations.
            pred_rss = pred_rss.maximum(torch.zeros((1,), device=self.params.device))
            #Clamp all RSS values to ≥ 0.
            zero_rss_inds = rx_data[:,keys,1] == 0
            #Looks at feature index 1 (x coordinate) to detect which entries were "empty".
            # print("Zero rss inds = ", zero_rss_inds.shape)
            rx_data[:,keys,0][zero_rss_inds] = pred_rss[zero_rss_inds]
            #For those receivers that had 0 RSS, fill in with predicted synthetic RSS.
            rx_data[:,keys,1:3] = torch.tensor(np.array(rx_locs), device=self.params.device)
            rx_data[:,keys,3] = torch.tensor(keys, dtype=torch.float32, device=self.params.device)
            rx_data[:,keys,4] = torch.tensor(rx_types, device=self.params.device)
            #Ensure that all: RX coordinates (x, y), RX ID (key), RX type are correctly filled in and consistent with the latest RX config.
            if key == self.train_key:
                train_rx, train_tx = rx_data, tx_data
            #If we are modifying the current training dataset, then store these updated tensors into train_rx and train_tx.
            #self.test_keys = [self.test_keys[0]] + [key+'_synthetic' for key in self.test_keys]
        else:
            train_rx, train_tx = rx_data, tx_data
            #If convert_all_inputs=False, use the original RX and TX data without modifying it.
            #these are original training data

        print("After Processing = ", train_tx.shape)


        if self.params.synthetic_only:
            #if synthetic only, then it will have only synthetic values under key_synthetic dataset
            #otherwise it will combine true+synthetic -> synthetic
            rx, tx = new_rx, new_tx
            #If synthetic_only: use only the synthetic data (new_rx, new_tx)
        else:
            rx, tx = torch.cat((train_rx, new_rx)), torch.cat((train_tx, new_tx))
            #Otherwise: combine the synthetic data with the (possibly corrected) training data.

        if self.params.augment_two_tx_only:
            rx, tx = train_rx, train_tx

        if self.params.one_tx:
            dataset = torch.utils.data.TensorDataset(rx, tx)
            #You bundle the receiver (rx) and transmitter (tx) tensors into a TensorDataset.
            key = self.train_key
            #Store the current training key in a local variable key.
            self.data[key+'_synthetic'] = copy.copy(self.data[key])
            self.data[key+'_synthetic'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
            self.data[key+'_synthetic'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
            self.train_key = self.train_key + '_synthetic'
            #Then make the new train_key as the synthetic one
        else:
            from tirem_two_transmitters_augmentation import two_transmitter_augmented_data, two_transmitter_augmented_data_new_locations
            num_aug_samples = new_tx.shape[0]
            if method == 'tirem_nn':
                augmented_two_tx, augmented_two_tx_rx = two_transmitter_augmented_data(self, num_samples=num_aug_samples)
                augmented_two_tx = torch.tensor(augmented_two_tx, dtype=torch.float32, device=self.params.device)
                augmented_two_tx_rx = torch.tensor(augmented_two_tx_rx, dtype=torch.float32, device=self.params.device)

                train_data = self.data[self.train_key]
                tx_all = train_data.ordered_dataloader.dataset.tensors[1]
                rx_all = train_data.ordered_dataloader.dataset.tensors[0]
                tx_flags = tx_all[:, :, 0]
                two_tx_mask = (tx_flags == 1).sum(dim=1) == 2
                true_two_tx = tx_all[two_tx_mask]
                true_two_rx = rx_all[two_tx_mask]

                if self.params.synthetic_only:
                    final_tx = torch.cat((tx, augmented_two_tx), dim=0)
                    final_rx = torch.cat((rx, augmented_two_tx_rx), dim=0)
                else:
                    final_tx = torch.cat((tx, true_two_tx, augmented_two_tx), dim=0)
                    final_rx = torch.cat((rx, true_two_rx, augmented_two_tx_rx), dim=0)

                dataset =  torch.utils.data.TensorDataset(final_rx, final_tx)
                key = self.train_key
                self.data[key + '_synthetic'] = copy.copy(self.data[key])
                self.data[key + '_synthetic'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
                self.data[key + '_synthetic'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
                self.train_key = key + '_synthetic'

                print("train key = ", self.train_key)
            elif method == 'rbf':
                from synthetic_augmentations2 import SyntheticAugmentor as SyntheticAugmentorTwoTx
                method = self.params.augmentation if method is None else method
                if method not in ['linear', 'nearest', 'rbf', 'tirem', 'tirem_nn', 'celf']:
                    return

                augmentor = SyntheticAugmentorTwoTx(method=method, rldataset=self, all_params = self.params)
                augmentor.train()
                # print("rx shape = ", rx.shape, " tx shape = ", tx.shape)
                # print("Here printing this Num Aug Samples = ", num_aug_samples)
                # print(abcd)

                augmented_two_tx, augmented_two_tx_rx = augmentor.get_results_rbf(num_aug_samples = num_aug_samples)
                
                # Convert to torch tensors
                augmented_two_tx = torch.tensor(augmented_two_tx, dtype=torch.float32, device=self.params.device)
                augmented_two_tx_rx = torch.tensor(augmented_two_tx_rx, dtype=torch.float32, device=self.params.device)

                # Extract true 2-TX samples from the original training set
                train_data = self.data[self.train_key]
                tx_all = train_data.ordered_dataloader.dataset.tensors[1]
                rx_all = train_data.ordered_dataloader.dataset.tensors[0]

                tx_flags = tx_all[:, :, 0]
                two_tx_mask = (tx_flags == 1).sum(dim=1) == 2
                true_two_tx = tx_all[two_tx_mask]
                true_two_rx = rx_all[two_tx_mask]

                # Combine everything
                if self.params.synthetic_only:
                    final_tx = torch.cat((tx, augmented_two_tx), dim=0)
                    final_rx = torch.cat((rx, augmented_two_tx_rx), dim=0)
                else:
                    final_tx = torch.cat((tx, true_two_tx, augmented_two_tx), dim=0)
                    final_rx = torch.cat((rx, true_two_rx, augmented_two_tx_rx), dim=0)

                print("Final Tx shape = ", final_tx.shape)
                print("Final Rx shape = ", final_rx.shape)

                # Wrap in dataset and update keys
                dataset = torch.utils.data.TensorDataset(final_rx, final_tx)
                key = self.train_key
                self.data[key + '_synthetic'] = copy.copy(self.data[key])
                self.data[key + '_synthetic'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
                self.data[key + '_synthetic'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
                self.train_key = key + '_synthetic'

                print("train key = ", self.train_key)
                print("test keys = ", self.test_keys)

            elif method == 'celf':
                from synthetic_augmentations2 import SyntheticAugmentor as SyntheticAugmentorTwoTx
                method = self.params.augmentation if method is None else method
                if method not in ['linear', 'nearest', 'rbf', 'tirem', 'tirem_nn', 'celf']:
                    return

                augmentor = SyntheticAugmentorTwoTx(method=method, rldataset=self, all_params = self.params)
                augmentor.train()
                # print("rx shape = ", rx.shape, " tx shape = ", tx.shape)
                # print("Num Aug Samples = ", num_aug_samples)
                # print(abcd)
                augmented_two_tx, augmented_two_tx_rx = augmentor.get_results_celf(num_aug_samples = num_aug_samples)

                # Convert to torch tensors
                augmented_two_tx = torch.tensor(augmented_two_tx, dtype=torch.float32, device=self.params.device)
                augmented_two_tx_rx = torch.tensor(augmented_two_tx_rx, dtype=torch.float32, device=self.params.device)

                # Extract true 2-TX samples from the original training set
                train_data = self.data[self.train_key]
                tx_all = train_data.ordered_dataloader.dataset.tensors[1]
                rx_all = train_data.ordered_dataloader.dataset.tensors[0]

                tx_flags = tx_all[:, :, 0]
                two_tx_mask = (tx_flags == 1).sum(dim=1) == 2
                true_two_tx = tx_all[two_tx_mask]
                true_two_rx = rx_all[two_tx_mask]

                # Combine everything
                if self.params.synthetic_only:
                    final_tx = torch.cat((tx, augmented_two_tx), dim=0)
                    final_rx = torch.cat((rx, augmented_two_tx_rx), dim=0)
                else:
                    final_tx = torch.cat((tx, true_two_tx, augmented_two_tx), dim=0)
                    final_rx = torch.cat((rx, true_two_rx, augmented_two_tx_rx), dim=0)

                print("Final Tx shape = ", final_tx.shape)
                print("Final Rx shape = ", final_rx.shape)

                # Wrap in dataset and update keys
                dataset = torch.utils.data.TensorDataset(final_rx, final_tx)
                key = self.train_key
                self.data[key + '_synthetic'] = copy.copy(self.data[key])
                self.data[key + '_synthetic'].dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, pin_memory=pin_memory)
                self.data[key + '_synthetic'].ordered_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=False, pin_memory=pin_memory)
                self.train_key = key + '_synthetic'

                print("train key = ", self.train_key)
                print("test keys = ", self.test_keys)

                print("Synthetic Only = ", self.params.synthetic_only, " Only Two Tx Aug = ", self.params.augment_two_tx_only)

                print(abcd)

        if method == 'fusion':
            rx_data = self.data[self.train_key].ordered_dataloader.dataset.tensors[0]
            tx_data = self.data[self.train_key].ordered_dataloader.dataset.tensors[1]
            coords = tx_data[:,0,1:].round().int().cpu()
            self.synthetic_rss = torch.tensor(self.all_prop_maps[:,:,coords[-len(new_rx):,1], coords[-len(new_rx):,0]]).to(self.params.device)


    def add_synthetic_training_data_just_print(self, method=None, synthetic_only=False, convert_all_inputs=True, synthetic_sample_distance_in_pixels=1):
        print("train_key = ", self.train_key, " test_keys= ", self.test_keys)
        #Train Key =  0.2testsizegrid5_train
        #Test Key =  ['0.2testsizegrid5_test', '0.2testsizegrid5_train_val']
        from synthetic_augmentations import SyntheticAugmentor
        method = self.params.augmentation if method is None else method
        if method not in ['linear', 'nearest', 'rbf', 'tirem', 'tirem_nn', 'celf']:
            return       

        # in_built = True
        # if not in_built:
        mock_values = False
        if mock_values:
            # Simulate receiver information
            keys = [[k] for k in [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 25, 26, 27, 32, 33, 35, 36]]

            # Create mock receiver coordinates
            rx_locs = [np.array([x, y], dtype=np.float32) for x, y in np.random.uniform(low=0, high=90, size=(23, 2))]

            # Create mock receiver types
            rx_types = np.random.choice([2.0, 3.0, 4.0], size=23).tolist()

            # Create mock propagation maps with shape (23 receivers, 89 height, 97 width)
            prop_maps = np.random.rand(23, 89, 97).astype(np.float32)

            # Fake TX test coordinates (not really used if you're just writing logic)
            tx_test_coords = np.random.rand(10, 2)

            # Dummy test predictions and test RSS values
            rx_test_preds = np.random.rand(10, 2)
            rx_test_rss = np.random.rand(10, 23)

            # Final assignment to mimic augmentor.get_results()
            prop_results = (prop_maps, keys, rx_locs, rx_types, rx_test_preds, rx_test_rss, tx_test_coords)

        else:
            augmentor = SyntheticAugmentor(method=method, rldataset=self)
            augmentor.train()
            prop_results = augmentor.get_results()
        prop_maps, keys, rx_locs, rx_types, rx_test_preds, rx_test_rss, tx_test_coords = prop_results

        print("keys = ", keys)
        keys = [key[0] for key in keys]
        # print(abcd)


        #Propagation Maps shape = Prop maps =  (23, 89, 97)
        #Keys are all the receivers: Keys =  [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 25, 26, 27, 32, 33, 35, 36]
        #There are in total 23 receivers and so the first dimension of the prop_maps is 23
        #The location of all the receivers: Rx Locs =  [array([60.24944 , 16.842804], dtype=float32), array([45.488373, 83.37936 ], dtype=float32), array([58.174618, 75.43506 ], dtype=float32), array([58.331776, 45.90307 ], dtype=float32), array([46.82482 , 64.281166], dtype=float32), array([22.11929, 54.5548 ], dtype=float32), array([79.80298 , 38.505154], dtype=float32), array([22.534323, 45.09213 ], dtype=float32), array([41.259506, 33.099636], dtype=float32), array([78.0729  , 59.037415], dtype=float32), array([37.793865, 75.748314], dtype=float32), array([54.521755, 60.593662], dtype=float32), array([52.731426, 71.090775], dtype=float32), array([29.380293, 61.767452], dtype=float32), array([37.419796, 48.146   ], dtype=float32), array([55.702114, 71.062386], dtype=float32), array([ 8.042728, 33.95312 ], dtype=float32), array([54.7031  , 58.371593], dtype=float32), array([42.725594, 66.25744 ], dtype=float32), array([61.50944 , 60.482586], dtype=float32), array([ 3.1636274, 18.324903 ], dtype=float32), array([26.803314, 32.617012], dtype=float32), array([60.672882, 54.184803], dtype=float32)]
        #Type of each receivers: Rx Types =  [4.0, 2.0, 3.0, 3.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, 4.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0]
        # print(abcd)

        if self.params.dataset_index == 6:
            rm13 = keys.index(13)
            #Find the index of key 13 in the list keys.
            #13 is the 10th element (Python indexing starts at 0).
            #rm13 = 9
            keys.pop(rm13)
            #Remove the key at index 9, which is 13.
            #(22 keys now).
            rx_locs.pop(rm13)
            #Remove the receiver location corresponding to key 13.
            rx_types.pop(rm13)
            #Remove the receiver type for key 13.
            rm13_mask = np.ones(len(prop_maps), dtype=bool)
            rm13_mask[rm13] = False
            #Create a boolean mask rm13_mask, of length 23 (len(prop_maps)).
            #Then set the 9th index (index=9) to False.
            prop_maps = prop_maps[rm13_mask]
            #After: prop_maps.shape = (22, 89, 97)

        train_data = self.data[self.train_key]
        if self.params.one_tx:
            #self.train_key tells which dataset to load (for example, 'train'
            tx_data = train_data.ordered_dataloader.dataset.tensors[1]
            #train_data now holds your original training dataset (with TX and RX tensors inside).
            tx_locs = tx_data[:,:,1:]
            #tx_locs hold all the locations of the transmitters in the train set
            rx_data = train_data.ordered_dataloader.dataset.tensors[0]
            #rx_data holds necessary informations
        else: 
            tx_data = train_data.ordered_dataloader.dataset.tensors[1]
            rx_data = train_data.ordered_dataloader.dataset.tensors[0]
            # Step 1: Mask to filter only samples with exactly one TX active
            tx_flags = tx_data[:, :, 0]  # shape: (N, 2)
            one_tx_mask = (tx_flags == 1).sum(dim=1) == 1  # shape: (N,)

            # Step 2: Apply the same mask to both tx_data and rx_data
            tx_data = tx_data[one_tx_mask]
            rx_data = rx_data[one_tx_mask]

            # Step 3: Get active TX coordinates, shape → [N, 1, 2]
            tx_coords = tx_data[:, :, 1:]  # shape: (N, 2, 2)
            tx_flags = tx_data[:, :, 0]
            active_tx_index = tx_flags.argmax(dim=1)  # (N,)
            tx_locs = torch.gather(tx_coords, 1, active_tx_index.view(-1, 1, 1).expand(-1, 1, 2))

        # print("Tx Data shape = ", tx_data.shape)
        # print("Tx locs sample = ", tx_locs[50:70])
        self.prop_maps = torch.tensor(prop_maps, dtype=torch.float32, device=self.params.device)
        #Here we are making the propagation map a tensor
        grid_bounds = [[1,self.img_width()-1], [1,self.img_height()-1]]
        #Defining a rectangular area excluding the border
        if self.params.dataset_index == 6:
            grid_bounds = [[3,70], [15,self.img_height()]]
        #Custom grid bound for the 6th dataset
        grid = np.array(np.meshgrid(np.arange(grid_bounds[0][0],grid_bounds[0][1]), np.arange(grid_bounds[1][0],grid_bounds[1][1]))).reshape(2,-1).T 
        #At first made a 2D array of grids
        #Then reshaped to two points denoting (x,y) coordinate of each grid location

        # print("grid = ", grid)
        test_data = self.data[self.test_keys[0]]
        #self.train_key tells which dataset to load (for example, 'train'
        tx_data_test = test_data.ordered_dataloader.dataset.tensors[1]
        #train_data now holds your original training dataset (with TX and RX tensors inside).
        tx_locs_test = tx_data_test[:,:,1:]
        #tx_locs hold all the locations of the transmitters in the train set
        rx_data_test = test_data.ordered_dataloader.dataset.tensors[0]

        tx_grid = tx_locs_test.squeeze(1).detach().cpu().numpy()  # convert to NumPy if needed
        tx_grid = np.round(tx_grid).astype(int)  # now same shape and type as `grid`
        # grid = tx_grid

        # print("tx_locs = ", tx_locs)
        # print("tx_grid = ", tx_grid)
        grid = tx_grid
        
        new_tx_locs = grid
        new_rss = self.prop_maps[:, new_tx_locs[:,1], new_tx_locs[:,0]].T
        #For each new tx, get the predicted RSS value for each of the receivers
        new_rss = new_rss.maximum(torch.zeros((1,), device=self.params.device))

        # print("rx_data shape = ", rx_data_test.shape)
        # print("new_rss shape = ", new_rss.shape)
        #Ensuring new rss values to be non-negative
        self.prop_keys = keys
        #copying receiver IDs
        if self.params.one_tx:
            new_tx = torch.zeros((len(new_tx_locs), tx_locs.shape[1], 3), device=self.params.device)
        else:
            new_tx = torch.zeros((len(new_tx_locs), tx_locs.shape[1] + 1, 3), device=self.params.device)
        #making a new_tx tensor of shape (total_sample, num_tx(1), 3)
        new_rx = torch.zeros((len(new_tx_locs), rx_data.shape[1], rx_data.shape[2]), device=self.params.device)
        #Similarly new receiver tensor of shape (total_sample, number_of_receivers, five_respective_values)
        if self.params.one_tx:
            new_tx[:,:,0] = 1
            new_tx[:,:,1:3] = torch.tensor(new_tx_locs, dtype=torch.float32, device=self.params.device).unsqueeze(1)
        else:
            new_tx[:,0,0] = 1
            new_tx[:,0,1:3] = torch.tensor(new_tx_locs, dtype=torch.float32, device=self.params.device)
            new_tx[:,1,1:3] = 0.0
        new_rx[:,keys,0] = new_rss
        new_rx[:,keys,1:3] = torch.tensor(np.array(rx_locs), device=self.params.device)
        new_rx[:,keys,3] = torch.tensor(keys, dtype=torch.float32, device=self.params.device)
        new_rx[:,keys,4] = torch.tensor(rx_types, device=self.params.device)
        #assigning all the values
        pin_memory = self.params.device != torch.device('cuda') 
        #to speed up if not running on GPUs

        # min_rss, max_rss = self.get_min_max_rss_from_key(2)
        alias_map = {
            2: 38,
            4: 24,
            12: 39,
            13: 29,
            25: 31
        }
        for rx_array in [rx_data_test, new_rx]:
            # rx_array shape: [N, 40, 5]
            for i in range(rx_array.shape[0]):  # for each sample
                if rx_array is new_rx and self.params.dataset_index == 6:
                    for orig_id, alias_id in alias_map.items():
                        # Find index in current sample where sensor_id == orig_id
                        match_idx = (rx_array[i, :, 3] == orig_id).nonzero(as_tuple=True)
                        if len(match_idx[0]) > 0:
                            match_idx = match_idx[0][0].item()
                            rx_array[i, alias_id] = rx_array[i, match_idx].clone()
                            rx_array[i, alias_id, 3] = alias_id  # update sensor_id
                for j in range(rx_array.shape[1]):  # for each sensor (40 total)
                    sensor_id = int(rx_array[i, j, 3].item())      # 4th element is sensor_id
                    sensor_type = int(rx_array[i, j, 4].item())    # 5th element is sensor_type

                    if sensor_type != 0:
                        min_rss, max_rss = self.get_min_max_rss_from_key(sensor_id)
                        norm_rss = rx_array[i, j, 0].item()  # 1st element is normalized RSS
                        denorm_rss = norm_rss * (max_rss - min_rss) + min_rss
                        rx_array[i, j, 0] = denorm_rss  # write back denormalized RSS

        print("tx_data_test shape = ", tx_data_test.shape," samples = ", tx_data_test[:5])
        print("rx_data_test shape = ", rx_data_test.shape,"  samples = ", rx_data_test[:5])

        print("New tx shape = ", new_tx.shape," samples = ", new_tx[:5])
        print("New rx shape = ", new_rx.shape,"  samples = ", new_rx[:5])

        print("Model Directory = ", self.params.model_dir)

        os.makedirs(self.params.model_dir, exist_ok=True)
        csv_path = os.path.join(self.params.model_dir, "final_report.csv")

        num_samples = tx_data_test.shape[0]
        num_sensors = rx_data_test.shape[1]
        headers = ["sample_no", "tx_x_coords", "tx_y_coords"] + [f"Sensor-{i}" for i in range(num_sensors)]

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for i in range(num_samples):
                coords_tensor = tx_data_test[i, 0]
                tx_x, tx_y = coords_tensor[1].item(), coords_tensor[2].item()

                true_rss = [rx_data_test[i, j, 0].item() for j in range(num_sensors)]
                writer.writerow([f"True-{i+1}", tx_x, tx_y] + true_rss)

                pred_rss = [new_rx[i, j, 0].item() for j in range(num_sensors)]
                writer.writerow([f"Predicted-{i+1}", tx_x, tx_y] + pred_rss)

        print(abcd)









    

    def get_center_point(self):
        if self.params.dataset_index < 6 or self.params.dataset_index == 8:
            center_point = np.array([(self.max_x - self.min_x)/2, (self.max_y - self.min_y)/2])
        elif self.params.dataset_index == 6:
            center_point = np.array([1100,1200])
        elif self.params.dataset_index == 7:
            center_point = np.array([2500,3000])
        return center_point
            

    def filter_bounds(self, boundary, tx_coords=None, rx_coords=None):
        """
        boundary is either:
            a (2,2) numpy array with the bottom left and top right corners of a bounding box
            a shapely Polygon
            a list of shapely Polygons
        """
        if tx_coords is not None:
            if isinstance(boundary, Polygon) or isinstance(boundary[0], Polygon):
                good_inds = []
                for ind, coord in enumerate(tx_coords):
                    point = Point(coord[0], coord[1])
                    if (isinstance(boundary, Polygon) and boundary.contains(point)) or any(bound.contains(point) for bound in boundary):
                        good_inds.append(ind)
                tx_coords = tx_coords[np.array(good_inds).astype(int)]
            else:
                assert(len(boundary) == 2)
                lat_min, lon_min = boundary.min(axis=0)
                lat_max, lon_max = boundary.max(axis=0)
                lat_violation = sum(tx_coords[:,0] < lat_min) + sum(tx_coords[:,0] > lat_max)
                lon_violation = sum(tx_coords[:,1] < lon_min) + sum(tx_coords[:,1] > lon_max)
                if lat_violation or lon_violation:
                    tx_coords = np.empty((0,2))
            if rx_coords is None:
                return tx_coords
        if rx_coords is not None:
            if isinstance(boundary, Polygon):
                good_inds = []
                for ind, coord in enumerate(rx_coords):
                    point = Point(coord[1], coord[2])
                    if boundary.contains(point):
                        good_inds.append(ind)
                rx_coords = rx_coords[np.array(good_inds)]
            else:
                assert(len(boundary) == 2)
                lat_min, lon_min = boundary.min(axis=0)
                lat_max, lon_max = boundary.max(axis=0)
                rx_coords = rx_coords[ ~np.isinf(rx_coords[:,0])]
                rx_coords = rx_coords[ ~(rx_coords[:,1] == 0)]
                rx_coords = rx_coords[ ~(rx_coords[:,1] < lat_min)]
                rx_coords = rx_coords[ ~(rx_coords[:,1] > lat_max)]
                rx_coords = rx_coords[ ~(rx_coords[:,2] < lon_min)]
                rx_coords = rx_coords[ ~(rx_coords[:,2] > lon_max)]
            if tx_coords is None:
                return rx_coords
        return tx_coords, rx_coords
    

    def get_citation(self):
        label = f'Loading data from DS {self.params.dataset_index}\nPlease reference the original work when using this dataset:\n'
        if self.params.dataset_index == 1:
            citation = f"""@article{{patwari2003relative,
  title={{Relative location estimation in wireless sensor networks}},
  author={{Patwari, Neal and Hero, Alfred O and Perkins, Matt and Correal, Neiyer S and {{O'Dea}}, Robert J}},
  journal={{IEEE Transactions on Signal Processing}},
  volume={{51}},
  number={{8}},
  pages={{2137--2148}},
  year={{2003}},
  publisher={{IEEE}}
}}"""
            license = 'Data used by permission under CC BY 4.0 License.'
            url = 'https://dx.doi.org/10.15783/C7630J'
        elif self.params.dataset_index in [2,3,4]:
            citation = f"""@inproceedings{{sarkar2020llocus,
  title={{LLOCUS: learning-based localization using crowdsourcing}},
  author={{Sarkar, Shamik and Baset, Aniqua and Singh, Harsimran and Smith, Phillip and Patwari, Neal and Kasera, Sneha and Derr, Kurt and Ramirez, Samuel}},
  booktitle={{Proceedings of the 21st International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing}},
  pages={{201--210}},
  year={{2020}}
}}"""
        elif self.params.dataset_index == 5:
            citation = f"""@inproceedings{{mitchell2022tldl,
  title={{Deep Learning-based Localization in Limited Data Regimes}},
  author={{Mitchell, Frost and Baset, Aniqua and Patwari, Neal and Kasera, Sneha Kumar and Bhaskara, Aditya}},
  booktitle={{Proceedings of the 2022 ACM Workshop on Wireless Security and Machine Learning}},
  pages={{15--20}},
  year={{2022}}
}}"""
        elif self.params.dataset_index == 6:
            citation = f"""@INPROCEEDINGS{{mitchell2023cutl,
  author={{Mitchell, Frost and Patwari, Neal and Bhaskara, Aditya and Kasera, Sneha Kumar}},
  booktitle={{2023 20th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON)}}, 
  title={{Learning-based Techniques for Transmitter Localization: A Case Study on Model Robustness}}, 
  year={{2023}},
  volume={{}},
  number={{}},
  pages={{133-141}},
  keywords={{Location awareness;Training;Wireless sensor networks;Wireless networks;Radio transmitters;Receivers;Interference;transmitter localization;model robustness;RF spectrum sensing}},
  doi={{10.1109/SECON58729.2023.10287483}}}}
"""
            license = 'Data used by permission under CC BY 4.0 License.'
            url = 'https://doi.org/10.5281/zenodo.7259895'
        elif self.params.dataset_index == 7:
            citation =f"""@dataset{{aernouts_2018_1193563,
  author       = {{Aernouts, Michiel and
                  Berkvens, Rafael and
                  Van Vlaenderen, Koen and
                  Weyn, Maarten}},
  title        = {{{{Sigfox and LoRaWAN Datasets for Fingerprint 
                   Localization in Large Urban and Rural Areas}}}},
  month        = mar,
  year         = 2018,
  publisher    = {{Zenodo}},
  version      = {{1.0}},
  doi          = {{10.5281/zenodo.1193563}},
  url          = {{https://doi.org/10.5281/zenodo.1193563}}
}}
"""
            license = 'Data used by permission under CC BY 4.0 License.'
            url = 'https://doi.org/10.5281/zenodo.1193563'
        elif self.params.dataset_index == 8:
            citation = f"""@misc{{tadikmeas2024,
    author       = "Tadik, S. and Singh, A. and Mitchell, F. and Hu, Y. and Yao, X. and Webb, K. and Sarbhai, A. and Maas, D. and Orange, A. and Van der Merwe, J. and Patwari, N. and Ji, M. and Kasera, Sneha K. and Bhaskara, A. and Durgin, Gregory D.",
    title        = "Salt Lake City 3534 MHz Multi-Transmitter Measurement Campaign",
    year         = "2024",
    month        = "March",
    howpublished = {{\\url{{https://github.com/serhatadik/slc-3534MHz-meas}}}}
}}"""
            license = 'Data used by permission under MIT License'
            url = 'https://github.com/serhatadik/slc-3534MHz-meas'
        if self.params.dataset_index in [2,3,4,5]:
            full_reference = f"{label}\n{citation}\n"
        else:
            full_reference = f"{label}\n{citation}\n{license}\n{url}\n"
        return full_reference


    def load_data(self):
        data_list = []
        #in load_data(), create an emply list named data_list
        print(self.get_citation())
        #print the citation of respective dataset
        if self.params.dataset_index == 1:
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                    #read all the lines of the dataset
                rx_loc_inds = {}
                #receiver location dictionary is empty
                current_ind = 0
                #current index is zero
                for line in lines:
                
                    columns = line.split()
                    #split in three columns "rx_data", "tx_coords" and "metadata"
                    num_trans = int(columns[1])
                    tx_locs = []
                    tx_gains = []
                    for i in range(num_trans):
                        x, y = columns[2+i].split(',')
                        tx_locs.append([ round(float(x)*2) / 2, round(float(y)*2)/2])
                        tx_gains.append(1)
                    rx_tups = []
                    for rx in columns[2+num_trans:]:
                        rss, x, y = rx.split(',')
                        if (x,y) in rx_loc_inds:
                            ind = rx_loc_inds[(x,y)]
                        else:
                            rx_loc_inds[(x,y)] = current_ind
                            ind = current_ind
                            current_ind += 1
                        rx_tups.append([float(rss), float(x), float(y), ind])
                    print(np.array(rx_tups)[:,1:].min(axis=0))
                    min_coords = np.array([0,5,1,0])
                    data_list.append([np.array(tx_locs) + min_coords[1:3], np.array(rx_tups) + min_coords, np.array(tx_gains)])
                self.location_index_dict = rx_loc_inds
        elif self.params.dataset_index < 5:
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    columns = line.split()
                    num_trans = int(columns[1])
                    tx_locs = []
                    tx_gains = []
                    for i in range(num_trans):
                        x, y = columns[2+i].split(',')
                        tx_locs.append([ round(float(x)*2) / 2, round(float(y)*2)/2])
                        tx_gains.append(1)
                    rx_tups = []
                    for rx in columns[2+num_trans:]:
                        rss, x, y = rx.split(',')
                        rx_tups.append([float(rss), float(x), float(y)])
                    data_list.append([np.array(tx_locs), np.array(rx_tups), np.array(tx_gains)])
                max_rx = max([len(ent[1]) for ent in data_list])
                self.location_index_dict = dict(zip(range(max_rx), range(max_rx)))
        elif self.params.dataset_index == 5:
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                preamble = ''
                for line in lines:
                    first_length = len(line.split(',')[0])
                    this_preamble = line[first_length:].split('-')[0]
                    if this_preamble == preamble:
                        continue
                    preamble = this_preamble
                    columns = line.split(',')
                    num_trans = int(columns[1])
                    column_index = 2
                    tx_locs = []
                    tx_gains = []
                    for i in range(num_trans):
                        rss, x, y = columns[column_index:column_index + 3]
                        tx_locs.append([float(x)-1, float(y)-1])
                        tx_gains.append(rss)
                        column_index += 3
                    if self.params.one_tx and (len(tx_locs) > 1):
                        continue
                    rx_tups = []
                    while column_index < len(columns):
                        rss, x, y = columns[column_index:column_index +3]
                        rx_tups.append([float(rss), float(x)-1, float(y)-1])
                        column_index += 3
                    data_list.append([np.array(tx_locs), np.array(rx_tups), np.array(tx_gains)])
                max_rx = max([len(ent[1]) for ent in data_list])
                self.location_index_dict = dict(zip(range(max_rx), range(max_rx)))
        elif self.params.dataset_index == 6: #Loading our powder data
            #here comes our powder dataset
            tx_metadata = []
            #transmitter metadata is an empty list
            location_25p_dict = {}
            location_100p_dict = {}
            #two empty dictionary
            tx_dicts = {}
            #empty dictionary for transmitter
            with open('datasets/frs_data/location_indexes.json', 'r') as f:
                location_index_dict = json.load(f)
                #this dictionary will assign one integer to each sensor. total 39 sensors is in there
            bus_indexes = [location_index_dict[k] for k in location_index_dict if 'bus-' in k]
            #if the sensors are from bus, append there index (in integer) in the list
            bus_rss = {ind:[] for ind in bus_indexes}
            #dictionary of lists for each index of bus sensor to store rss values
            #it is an array of bus indices which will store all the rss values
            #each bus sensor observed throughout the all sensors
            for data_file in self.data_files:
            #read no_tx,one_tx and two_tx.json respectively
                with open(data_file, 'r') as f:
                    tmp_tx = json.load(f)
                tx_dicts = {**tx_dicts, **tmp_tx}
                #keeps all three dataset in the same dictionary separated by a comma
            for key in tx_dicts:
            #take one such sample
            #{"2022-11-23 13:24:40": {"rx_data": [[-91.0425122828992, 40.76414, -111.84759, "bookstore-nuc2-b210"], 
            # [-94.80184162046405, 40.76134, -111.84629, "cbrssdr1-bes-comp"], [-88.11484217641932, 40.76627, -111.84774, "cbrssdr1-browning-comp"], [-97.78666218162046, 40.75807, -111.85325, "cbrssdr1-fm-comp"], [-72.69726015872473, 40.7644, -111.83699, "cbrssdr1-honors-comp"], 
            # [-84.3909920290188, 40.77105, -111.83712, "cbrssdr1-hospital-comp"], [-91.76104326235179, 40.7674, -111.83118, "cbrssdr1-smt-comp"], [-77.59159321030702, 40.76895, -111.84167, "cbrssdr1-ustar-comp"], [-98.127440653848, 40.77105, -111.83712, "cellsdr1-hospital-comp"], [-39.05526912998569, 40.7674, -111.83118, "cellsdr1-smt-comp"], [-89.6430024877518, 40.7672, -111.8381, "cnode-ebc-dd-b210"], [-91.8045827476645, 40.76769, -111.83609, "cnode-guesthouse-dd-b210"], [-95.69567954000047, 40.77281166195038, -111.8409002868012, "cnode-mario-dd-b210"], [-80.66705227472619, 40.77107659230161, -111.84315777334905, "cnode-wasatch-dd-b210"], [-93.206942939429, 40.7677, -111.83816, "ebc-nuc1-b210"], [-75.2622687631013, 40.76148, -111.84201, "garage-nuc2-b210"], [-52.78500747372671, 40.76627, -111.83632, "guesthouse-nuc2-b210"], [-88.90916172097995, 40.76486, -111.84319, "humanities-nuc2-b210"], [-87.72963159011475, 40.7616, -111.85185, "law73-nuc2-b210"], [-81.4524897213561, 40.75786, -111.83634, "madsen-nuc2-b210"], [-88.56834641215616, 40.77006, -111.83784, "moran-nuc2-b210"], [-77.92344427491402, 40.76278, -111.83061, "sagepoint-nuc2-b210"], 
            # [-88.49975115942115, 40.76791, -111.84561, "web-nuc1-b210"]], 
            # "tx_coords": [[40.76521977, -111.83475621]], 
            # "metadata": [{"power": 1, "transport": "walking", "precip": "none", "radio": "TXB"}]}
                arr = np.array([line[:3] + [location_index_dict[line[3]]] for line in tx_dicts[key]['rx_data']])
                #go to rx_data for a key where key is date and time
                #each line basically [-94.80184162046405, 40.76134, -111.84629, "cbrssdr1-bes-comp"]
                #probably rss, x,y, sensor name
                #so in array, take first three values plus the index of the sensor from the name
                for ind in bus_indexes:
                #check if the row include any bus_sensor input
                    if ind in arr[:,3]:
                        bus_rss[ind].append(arr[arr[:,3] == ind,0][0])
                        #filter only the rows with this bus sensor
                        #there will be only one row for each sensor in the 2d array
                        #add its rss value
                tx_dicts[key]['rx_data'] = arr
                #multi-level dictionary
                #key:rx_data:a numpy array having [rss, x_coord, y_coord, sensor_index]
                #that means replace the previous value with a numpy array
                if 'tx_coords' in tx_dicts[key]:
                #if you have one or two transmiters
                    tx_dicts[key]['tx_coords'] = np.array(tx_dicts[key]['tx_coords'])
                    #making numpy array for the coordinates
                    #in case of one transmitter, it will be a 2D array with only one row
                    for one_metadata in tx_dicts[key]['metadata']:
                        #go to the metadata and add one field name 'time' with the key of the row
                        one_metadata['time'] = key
                        if 'precip' not in one_metadata:
                        #also add "precip" means precipitation data in the metadata
                            one_metadata['precip'] = 'none'

                # print("Tx dicts = ", tx_dicts[key])
                # print(abcd)
            default_min_rss = {2: -96, 3: -100, 4: -95}
            default_max_rss = {2: -35, 3: 20, 4: -4}
            #take min and max rss for each of the sensor categories
            #no min or max for bus type
            for name in location_index_dict:
                key = location_index_dict[name]
                #iterate over the sensor dictionary and take its index in key
                rx_type = 1 if 'bus-' in name else 2 if 'cnode' in name else 3 if ('cbrs' in name or 'cell' in name) else 4
                #tx_type based on the sensor name
                if rx_type == 1 and len(bus_rss[key]) > 0:
                    location_25p_dict[key] = np.quantile(bus_rss[key], 0.25)
                    location_100p_dict[key] = max(bus_rss[key])
                elif rx_type != 1:
                    location_25p_dict[key] = default_min_rss[rx_type]
                    location_100p_dict[key] = default_max_rss[rx_type]
            #for bus, take 25th percenteil as the min rss value of the sensor and max as max
            #otherwise just follow the self assigned value.
            self.location_min_rss_dict = location_25p_dict
            self.location_max_rss_dict = location_100p_dict
            #finally consider these values as min and max rss of that particular
            self.load_geotiff('datasets/frs_data/corrected_dsm.tif', 'datasets/frs_data/corrected_buildings.tif') 
            # Img is loaded as a rasterio image, with coordinates in UTM zone 12
            #it will add building_map and elevation_map to the parameters of the object
            #with the origin of the map
            boundary_gps_coordinates = coordinates.CAMPUS_LATLON
            #CAMPUS_LATLON = np.array([
        #     (40.75413,
        #   -111.85390),
        #     (40.77500,
        #     -111.82632),
        #     ])
                #It is actually bottom left and top right corner of the location
            if len(boundary_gps_coordinates) == 4:
                bounds = Polygon(boundary_gps_coordinates)
            else:
                bounds = boundary_gps_coordinates
            #bounds will have either the polygon object or just the numpy array with two tuples

            #Till now it contains all three types of dictionaries: no Tx, One tx and Two Tx
            #Moreover, rx has only 4 data. Categories hasn't been added yet
            #Tx also has only two data= The coordinates
            #The coordinates are in gps


            stationary_dict = {}
            off_campus_dict = {}
            #two empty dictionary for stationary and off_campus transmitters
            for key in tx_dicts:
                if 'tx_coords' not in tx_dicts[key]: continue
                #if no transmitter,just continue
                #So simply ignoring no tx cases
                tx_locs = tx_dicts[key]['tx_coords']
                #else take the coordinates
                if len(tx_locs) != 1 and self.params.one_tx: continue
                #if parameter says only one tx but contains not equal one, then continue
                rx_tups = tx_dicts[key]['rx_data']
                #take rx data
                tx_locs, rx_tups = self.filter_bounds(bounds, tx_coords=tx_locs, rx_coords=rx_tups)
                #send the coordinates to filter_bounds to filter out all the tx or rx out of the boundary
                #return elevation coordinate to be an empty array if it is out of the boundary
                if len(tx_locs) == 0:
                    #if the returned coordinate is empty, store this row or data in the off_campus_dict dictionary
                    off_campus_dict[key] = tx_dicts[key]
                    continue
                # Here, we're adjusting the utm coordinates to a local system, just because we have some precision problems at 32 bits where the affine transform takes place.
                # if len(tx_locs) == 2:
                #     # print("tx_locs = ", tx_locs)
                tx_locs = coordinates.convert_gps_to_utm(tx_locs, origin_to_subtract=self.elevation_map.origin)
                # if len(tx_locs) == 2:
                    # print("Origin Coordinate = ", self.elevation_map.origin)
                    # print("tx_locs = ", tx_locs)
                #it will convert all the transmitter location from gps to utm
                tx_gains = [1] * len(tx_locs)
                #a list of 1's equal to how many transmitters you have
                rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=self.elevation_map.origin)
                #convert the coordinates of receiver sensors from gps to utm as well
                if 'stationary' in tx_dicts[key]['metadata'][0]:
                    stationary_dict.setdefault(tx_dicts[key]['metadata'][0]['stationary'], []).append([tx_locs, rx_tups[:,:3], tx_gains, tx_dicts[key]['metadata'], rx_tups[:,3]])
                    continue
                #if any data point contains stationary in the metadata, just add it to the dictionary
                data_list.append([tx_locs, rx_tups, tx_gains])
                #in datalist append tx_locs, rx_tups(rss, xCoords,ycoords, sensor_idx), and tx_gains(a list containing 1 for each tx)
                #tx_dicts[key]['metadata'][0]['time'] = key
                tx_metadata.append(tx_dicts[key]['metadata'])
                #another list containing metadata for each sample
                #location_inds.append(rx_tups[:,3])

            #tx_dicts still unchanged. the changes are stored in data_list
            #data list pretty much is like [tx_coords, rx_coords(in utm), [1 \times no_tx]]
            for stationary_index in stationary_dict:
                continue
                tx_locs = np.array( [entry[0] for entry in stationary_dict[stationary_index] ] ).mean(axis=0)
                rx_tups = np.concatenate( [entry[1] for entry in stationary_dict[stationary_index] ] )
                tx_gains = [1]
                tx_md = stationary_dict[stationary_index][0][3]
                loc_inds = stationary_dict[stationary_index][0][4]
                data_list.append([tx_locs, rx_tups, tx_gains])
                tx_metadata.append(tx_md)
                location_inds.append(loc_inds)
            if False: #not self.params.force_num_tx: # This should execute when we want empty with no tx. Should be fixed.
                for key in no_tx_dicts:
                    tx_locs = []
                    tx_gains = []
                    rx_tups = no_tx_dicts[key]['rx_data']
                    rx_tups = self.filter_bounds(bounds, rx_coords=rx_tups)
                    rx_tups = rx_tups[ ~np.isinf(rx_tups[:,0])]
                    rx_tups = rx_tups[ ~(rx_tups[:,1] == 0)]
                    rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=self.elevation_map.origin)
                    data_list.append([tx_locs, rx_tups, tx_gains])
                    tx_metadata.append([{}])
                    #location_inds.append(rx_tups[:,3])
            self.location_index_dict = {}
            for key in location_index_dict:
                self.location_index_dict[location_index_dict[key]] = key
            #reversing the key-value relationship
            #'bus_123:5' to '5:'bus_123'
            #self.location_inds = np.array(location_inds)
        elif self.params.dataset_index == 7: #Loading antwerp lorawan data
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                with open('datasets/data_antwerp/chosen_gateways_5_sensors.txt') as f:
                    locs = json.load(f)
                self.load_geotiff('datasets/data_antwerp/antwerp_zoom_dsm.tif') # Img is loaded as a rasterio image, with coordinates in UTM zone 12
                tx_metadata = []
                used_ids = {}
                id_index = 0
                boundary_gps_coordinates = coordinates.ANTWERP_LATLON
                if len(boundary_gps_coordinates) == 4:
                    bounds = Polygon(boundary_gps_coordinates)
                else:
                    bounds = boundary_gps_coordinates
                for sample in data:
                    stationary_dict = {}
                    if sample['hdop'] > 2:
                        continue
                    metadata = {'hdop':sample['hdop'], 'sf': sample['sf']}
                    tx_locs = np.array([[sample['latitude'], sample['longitude'] ]])
                    if len(tx_locs) != 1 and self.params.one_tx: continue
                    rx_tups = []
                    gateway_ids = []
                    for gateway in sample['gateways']:
                        id = gateway['id']
                        if id not in locs: continue
                        #new_lat = coordinates.convert_gps_to_utm( np.array( [[locs[id]['latitude'], locs[id]['longitude']] ] ), origin_to_subtract=origin)
                        rx_tups.append( [ gateway['rssi'], locs[id]['latitude'], locs[id]['longitude'] ] )
                        gateway_ids.append(id)
                        metadata['time'] = gateway['rx_time']['time'][:10]
                    if len(rx_tups) == 0 or len(rx_tups) < self.params.min_sensors:
                        continue
                    for id in gateway_ids:
                        if id not in used_ids:
                            used_ids[id] = id_index
                            id_index += 1
                    for i in range(len(rx_tups)):
                        rx_tups[i].append(used_ids[gateway_ids[i] ])
                    rx_tups = np.array(rx_tups)
                    tx_locs, rx_tups = self.filter_bounds(bounds, tx_coords=tx_locs, rx_coords=rx_tups)
                    if len(tx_locs) == 0 or len(rx_tups) == 0 or len(rx_tups) < self.params.min_sensors:
                        continue
                    tx_locs = coordinates.convert_gps_to_utm(tx_locs, origin_to_subtract=self.elevation_map.origin)
                    rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=self.elevation_map.origin)
                    tx_gains = [1] * len(tx_locs)
                    data_list.append( [tx_locs, rx_tups, tx_gains] )
                    tx_metadata.append(metadata)
                self.location_index_dict = {}
                for key in used_ids:
                    self.location_index_dict[ used_ids[key]] = key
        elif self.params.dataset_index == 8:
            for data_file in self.data_files:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                tx_metadata = []
                used_ids = {}
                id_index = 0
                self.load_geotiff('datasets/frs_data/corrected_dsm.tif', 'datasets/frs_data/corrected_buildings.tif') 
                # Img is loaded as a rasterio image, with coordinates in UTM zone 12
                #it will initialize elevation_map and building_map variable of the object
                boundary_gps_coordinates = coordinates.DENSE_LATLON
                if len(boundary_gps_coordinates) == 4:
                    bounds = Polygon(boundary_gps_coordinates)
                else:
                    bounds = boundary_gps_coordinates
                new_id = 0
                for sample in data:
                    metadata = {'time':sample, 'mobility': data[sample]["metadata"]}
                    lats = [arr[3] for arr in data[sample]["pow_rx_tx"]]
                    for lat in lats:
                        if lat not in used_ids:
                            used_ids[lat] = new_id
                            new_id += 1
                    rx_tups = np.array([[arr[0], arr[3], arr[4], used_ids[arr[3]]] for arr in data[sample]["pow_rx_tx"]])
                    tx_locs = np.array(data[sample]["pow_rx_tx"][0][1:3])
                    tx_locs, rx_tups = self.filter_bounds(bounds, tx_coords=tx_locs[None], rx_coords=rx_tups)
                    if len(tx_locs) == 0: continue
                    tx_locs = coordinates.convert_gps_to_utm(tx_locs, origin_to_subtract=self.elevation_map.origin)
                    rx_tups[:,1:3] = coordinates.convert_gps_to_utm(rx_tups[:,1:3], origin_to_subtract=self.elevation_map.origin)
                    tx_gains = [1] * len(tx_locs)
                    data_list.append( [tx_locs, rx_tups, tx_gains] )
                    tx_metadata.append(metadata)
                self.location_index_dict = {}
                location_names = [
                    'cnode-ebc-dd-b210',
                    'cnode-guesthouse-dd-b210',
                    'cnode-mario-dd-b210',
                    'cnode-moran-dd-b210',
                    'cnode-wasatch-dd-b210',
                    'cbrssdr1-ustar-comp'
                ]
                for key, name in zip(used_ids, location_names):
                    self.location_index_dict[ used_ids[key]] = name
                

        tx_vecs = []
        rx_vecs = []
        #tx_gains = []
        #taking empty list of tx_vecs and rx_vecs

        min_x = 1e9
        min_y = 1e9
        max_x = -1e9
        max_y = -1e9
        min_rss = 1e9
        max_rss = -1e9
        #taking some min-max values
        #all_tx = []
        #all_rx = []
        #tx_count = []
        #rx_count = []
        for entry in data_list:
        #go to each entry of the datalist([tx_locs, rx_tups(rss, xCoords,ycoords, sensor_idx), and tx_gains(a list containing 1 for each tx)])
        #    if len(entry[0]) == 3 and [-1.0, -1.0] in entry[0]:
        #        continue
        #    tx_count.append(len(entry[0]))
            for tx in entry[0]:
        #        all_tx.append(tx)
                min_x = min(min_x, tx[0])
                min_y = min(min_y, tx[1])
                max_x = max(max_x, tx[0])
                max_y = max(max_y, tx[1])
            #iterating over all the entries and get min_x,min_y,max_x,max_y value
        #    rx_count.append(len(entry[1]))
            for rx in entry[1]:
        #        all_rx.append(rx)
                min_x = min(min_x, rx[1])
                min_y = min(min_y, rx[2])
                max_x = max(max_x, rx[1])
                max_y = max(max_y, rx[2])
                min_rss = min(min_rss, rx[0])
                max_rss = max(max_rss, rx[0])
            #similar for rx data as well
        #if self.verbose:
        #    print('X Range: [%i:%i]\tY Range: [%i:%i]\tRSS Range:[%.1f:%.1f]' % (min_x, min_y, max_x, max_y, min_rss, max_rss))
        self.min_x   =  min_x
        self.min_y   = min_y 
        self.max_x   = max_x 
        self.max_y   = max_y 
        #get all the min-max values from all the samples
        self.noise_floor = -114
        self.min_rss = min(self.noise_floor, min_rss)
        transmit_power = 5
        self.max_rss = max(max_rss, transmit_power)

        # print("Minimum RSS = ", self.min_rss) -114
        # print("Maximum RSS = ", self.max_rss) 5
        # print(abcd)
        #min-max rss value from all the values and noise floor, and transmit_powers
        ## Meter scale of 0.5 corresponds to 0.5 meters per pixel
        #if self.forced_meter_scale:
        #    self.meter_scale = self.forced_meter_scale
        #    self.img_size_x = int((max_x - min_x) / self.meter_scale) + 2*self.buffer
        #    self.img_size_y = int((max_y - min_y) / self.meter_scale) + 2*self.buffer
        #    self.img_size = max(self.img_size_x, self.img_size_y)
        #elif self.forced_img_size:
        #    self.img_size = self.forced_img_size
        #    x_scale = (max_x - min_x) / (self.img_size - 2*self.buffer)
        #    y_scale = (max_y - min_y) / (self.img_size - 2*self.buffer)
        #    self.meter_scale = max(x_scale, y_scale)
        #else:
        #    self.meter_scale = 1
        #    self.img_size_x = int((max_x - min_x) / self.meter_scale) + 2*self.buffer
        #    self.img_size_y = int((max_y - min_y) / self.meter_scale) + 2*self.buffer
        #    self.img_size = max(self.img_size_x, self.img_size_y)
        #    while self.img_size < 30:
        #        self.meter_scale = .5 * self.meter_scale
        #        self.img_size = int(max((max_x - min_x) / self.meter_scale, (max_y - min_y) / self.meter_scale ) + 2*self.buffer)
        #self.img_size_x = self.img_size
        #self.img_size_y = self.img_size
        #min_tx_arr = np.array([min_x, min_y])
        min_rx_arr = np.array([self.noise_floor, 0, 0, 0])
        #making a min_rx_arr of size 4 with [noise_floor, 0, 0, 0]
        for entry in data_list:
            if len(entry[0]) == 3 and [-1.0, -1.0] in entry[0]:
                continue
            #if three transmitter, ignore
            if len(entry[0]) == 0:
                continue
                tx_vecs.append( np.array(entry[0])) 
            #if zero transmitter, ignore
            else:
                #tx_vecs.append( self.buffer + (np.array(entry[0]) - min_tx_arr)/ self.meter_scale) 
                tx_vecs.append( np.array(entry[0])) 
                #if one or two transmitter, add their coordinates to the tx_vecs list
            #rx_vecs.append( np.array([0, self.buffer, self.buffer]) + (np.array(entry[1]) - min_rx_arr)/ np.array([self.max_rss - self.noise_floor, self.meter_scale, self.meter_scale])) 
            if self.params.dataset_index < 6 or self.params.dataset_index == 8:
                entry_min_rss = min_rss
                entry_max_rss = max_rss
            elif self.params.dataset_index == 6:
                entry_min_rss = np.array([self.location_min_rss_dict[int(loc_ind)] for loc_ind in entry[1][:,3] ])
                entry_max_rss = np.array([self.location_max_rss_dict[int(loc_ind)] for loc_ind in entry[1][:,3] ])
                #min-max rss for each entry based on the sensor type. if you recall, for bus it was 25p and 100p whereas for the other types, it was hardcoded
            elif self.params.dataset_index == 7:
                entry_min_rss = -126
                entry_max_rss = -70
            rx_vec = np.zeros((len(entry[1]),5))
            #take a new array of size five
            #rx_vec[:,0] = entry[1][:,0]/ np.array([self.max_rss - self.noise_floor, 1, 1, 1])
            rx_vec[:,0] = (entry[1][:,0] - entry_min_rss)/ (entry_max_rss - entry_min_rss)
            #normalize the rss value of each entry
            if self.params.dataset_index == 1:
                rx_vec[:,1:4] = entry[1][:,1:]
            elif self.params.dataset_index < 6:
                rx_vec[:,1:3] = entry[1][:,1:]
            elif self.params.dataset_index == 8:
                rx_vec[:,1:4] = entry[1][:,1:]
                rx_vec[:,4] = 2
            elif self.params.dataset_index >= 6:
                rx_vec[:,1:4] = entry[1][:,1:]
                #keep coordinates and sensor index as it is
                location_names = [self.location_index_dict[id] for id in rx_vec[:,3]]
                #location name for each input sensors
                rx_vec[:,4] = np.array([1 if 'bus-' in name else 2 if 'cnode' in name else 3 if 'cbrs' in name else 4 for name in location_names])
                #fill up the fourth index of new array with the sensor category
                if self.params.remove_mobile:
                    rx_vec = rx_vec[rx_vec[:,4] != 1]
                #if remove_mobile, then drop all with mobile
                # Make common inds
            ### Filter buses because they might be causing more trouble than they are helping with...
            #rx_vec = rx_vec[rx_vec[:,4] != 1]

            rx_vecs.append(rx_vec) 
            #append the mx5 array to the list
            #tx_gains.append(entry[2])
            #self.y_lengths.append(len(entry[0]) )
            #self.x_lengths.append(len(entry[1]) )
        #Setting
        print("In load_data total rx_vec = ", len(rx_vecs))
        print("In load_data total tx_vec = ", len(tx_vecs))

        # print("tx_vecs sample = ", tx_vecs[:5]) [array([[3100.57909358, 1734.1401454 ]]), array([[3094.32463231, 1729.45812595]])]
        # print("rx_vecs sample = ", rx_vecs[0]) [[4.34888760e-02 2.01624153e+03 1.62466279e+03 1.10000000e+01  4.00000000e+00]]


        self.max_num_rx = len(self.location_index_dict)
        #maximum number of receiver can be, keep in max_num_rx
        if self.params.dataset_index == 6:
            self.all_min_rss = np.array([self.location_min_rss_dict[key] for key in sorted(self.location_min_rss_dict)])
            self.all_max_rss = np.array([self.location_max_rss_dict[key] if key in self.location_max_rss_dict else -20 for key in sorted(self.location_min_rss_dict)])
        #all_min_rss and all_max_rss is the numpy arrays of min and max rss values of all the data sensors
        else:
            self.all_min_rss = self.min_rss
            self.all_max_rss = self.max_rss
        if self.params.dataset_index in [6,7,8]:
            print("Calling Samples for None type key")
            self.data[None] = self.Samples(self, np.array(rx_vecs, dtype=object), np.array(tx_vecs, dtype=object), tx_metadata=np.array(tx_metadata, dtype=object))
            #make a Samples object with rx_vecs, tx_vecs and tx_metadata as array and key will be 'None'
        else:
            self.data[None] = self.Samples(self, np.array(rx_vecs, dtype=object), np.array(tx_vecs, dtype=object))
        samples = self.data[None]

        origin = samples.origin
        top_corner = origin + np.array([samples.rectangle_width,  samples.rectangle_height])
        #get bottom-left and top-right points
        self.corners = np.array([origin, top_corner])
        #corners of the object keeps track of these two corner points
        #the process ends with a dictionary named "data" having the key "None" storing all the values


    def get_min_max_rss_from_key(self, key):
        if hasattr(self, 'location_min_rss_dict'):
            min_rss = self.location_min_rss_dict[key]
        else:
            min_rss = self.min_rss
        if hasattr(self, 'location_max_rss_dict'):
            max_rss = self.location_max_rss_dict[key] if key in self.location_max_rss_dict else -20
        else:
            max_rss = self.max_rss
        return min_rss, max_rss


    def load_geotiff(self, img_file, building_file=None):
        self.elevation_map = rasterio.open(img_file)
        self.elevation_map.origin = np.array(self.elevation_map.transform)[:6].reshape(2,3) @ np.array([0, self.elevation_map.shape[0], 1])
        if building_file is not None and os.path.exists(building_file):
            self.building_map = rasterio.open(building_file)
        else:
            self.building_map = None

        
    def separate_random_data(self, data_source_key=None, test_size=0.2, train_size=None, data_key_prefix='', use_folds=False, n_splits=5, num_tx=None, ending_keys=['train', 'test'], random_state=None):
        #in our case: test_size=params.test_size, train_size=train_size, data_key_prefix=data_key_prefix, data_source_key='campus', random_state=0
        if isinstance(num_tx, int):
            num_tx = [num_tx]
        if self.params.one_tx:
            num_tx = [1]
        #assigning num_tx value
        data_source = self.data[data_source_key]
        #take data from source
        if self.params.dataset_index == 6:
            inds = self.filter_inds_by_metadata(source_key=data_source_key)
            #take only the indx from metadata
            x_vecs = copy.deepcopy(data_source.rx_vecs[inds])
            y_vecs = copy.deepcopy(data_source.tx_vecs[inds])
            tx_metadata = copy.deepcopy(data_source.tx_metadata[inds])
            #deepcopy the values
        else:
            x_vecs = copy.deepcopy(data_source.rx_vecs)
            y_vecs = copy.deepcopy(data_source.tx_vecs)
        
        print("in separate random data, x_vecs shape = ", x_vecs.shape, " and data_source_key = ", data_source_key)
        print("in separate random data, y_vecs shape= ", y_vecs.shape, " and data_key_prefix = ", data_key_prefix)


        if num_tx == None or len(num_tx) < 1:
            num_tx = list(range(data_source.max_num_tx+1))
        #take a num_tx list. In our case num_tx = 1. So list of just one [1]
        valid_inds = [i for i, vec in enumerate(y_vecs) if len(vec) in num_tx]
        #only take having valid number of tx indices
        x_vecs = x_vecs[valid_inds]
        y_vecs = y_vecs[valid_inds]

        print("num_tx = ", num_tx)
        print("in separate random data, valid x_vecs shape = ", x_vecs.shape, " and data_source_key = ", data_source_key)
        print("in separate random data, valid y_vecs shape= ", y_vecs.shape, " and data_key_prefix = ", data_key_prefix)

        #y_vecs is still has two values

        count = 0
        unique_locations_dict = {}
        empty_count = 0
        for i, y_vec in enumerate(y_vecs):
            if len(y_vec) == 0: 
                key = empty_count 
                empty_count += 1
            else:
                key = y_vec[:,:2].tobytes()
            if key in unique_locations_dict:
                unique_locations_dict[key].append(i)
                count += 1
            else:
                unique_locations_dict[key] = [i]
                count += 1
        #key of our unique_location dict will be bytes value of each coordinate.
        #making a dictionary which will contain all the indices of the data samples
        #respective to each transmitter

        if use_folds:
            print('Not implemented.')
            #kf = KFold(n_splits=n_splits, random_state=self.random_state, shuffle=True)
            #fold_counter = 0
            #for train_index, test_index in kf.split(x_lengths):
            #    x_vecs_train, x_vecs_test = x_vecs[train_index], x_vecs[test_index]
            #    y_vecs_train, y_vecs_test = y_vecs[train_index], y_vecs[test_index]
            #    x_lengths_train, x_lengths_test = x_lengths[train_index], x_lengths[test_index]
            #    y_lengths_train, y_lengths_test = y_lengths[train_index], y_lengths[test_index]
            #    self.make_data_entry(data_key_prefix+'_other_folds_%i' % fold_counter, x_train, y_train, x_vecs_train, y_vecs_train, x_lengths_train, y_lengths_train, make_tensors=make_tensors)
            #    self.make_data_entry(data_key_prefix+'_fold_%i' % fold_counter, x_test, y_test, x_vecs_test, y_vecs_test, x_lengths_test, y_lengths_test, make_tensors=make_tensors)
            #    fold_counter += 1
        else:
            if len(data_key_prefix) == 0:
                train_key = ending_keys[0]
                test_key = ending_keys[1]
            else:
                train_key = data_key_prefix+'_' + ending_keys[0]
                test_key = data_key_prefix+'_' + ending_keys[1]
                #just appending train and test after each text
            
            train_location_keys, test_location_keys = train_test_split(list(unique_locations_dict.keys()), shuffle=True, test_size=test_size, train_size=train_size, random_state=random_state if random_state is not None else self.params.random_state)
            #split by each location and add index to the train or test list
            train_inds = []
            test_inds = []
            for key in train_location_keys:
                train_inds += unique_locations_dict[key]
            for key in test_location_keys:
                test_inds += unique_locations_dict[key]

            x_vecs_train, x_vecs_test = x_vecs[train_inds], x_vecs[test_inds]
            label_train, label_test = y_vecs[train_inds], y_vecs[test_inds]  
            print("After split")
            print("x_vecs_train = ",x_vecs_train.shape, " x_vecs_test = ",x_vecs_test.shape)
            print("y_vecs_train = ",label_train.shape, " y_vecs_test = ",label_test.shape)

            #still y_vecs is the same
            if self.params.dataset_index == 6:
                metadata_train, metadata_test = tx_metadata[train_inds], tx_metadata[test_inds]
                print("For train key = ", train_key)
                self.data[train_key] = self.Samples(self, x_vecs_train, label_train, tx_metadata=metadata_train)
                print("For test key = ", test_key)
                self.data[test_key] = self.Samples(self, x_vecs_test, label_test, tx_metadata=metadata_test)
            elif self.params.dataset_index == 7:
                self.data[train_key] = self.Samples(self, x_vecs_train, label_train, no_shift=True)
                self.data[test_key] = self.Samples(self, x_vecs_test, label_test, no_shift=True)
            else:
                self.data[train_key] = self.Samples(self, x_vecs_train, label_train)
                self.data[test_key] = self.Samples(self, x_vecs_test, label_test)
            return train_key, test_key


    def set_default_keys(self, aug_train_key=None, train_key=None, test_key=None, test_keys=[]):
        self.aug_train_key = aug_train_key if aug_train_key!=None or not hasattr(self, 'aug_train_key') else self.aug_train_key
        self.train_key = train_key if train_key!=None or not hasattr(self, 'train_key') else self.train_key
        self.test_key = test_key if test_key!=None or not hasattr(self, 'test_key') else self.test_key
        self.test_keys = test_keys if len(test_keys)!=0 or not hasattr(self, 'test_keys') else self.test_keys
        keys =  set([aug_train_key, train_key, test_key] + test_keys)
        for key in keys:
            if key is not None and not hasattr(self.data[key],'dataloader'):
                self.data[key].make_tensors()


    def img_height(self): 
        if hasattr(self, 'train_key'):
            key = self.train_key
        else:
            key = None
        return round(self.data[key].rectangle_height / self.params.meter_scale) + 1


    def img_width(self):
        if hasattr(self, 'train_key'):
            key = self.train_key
        else:
            key = None
        return round(self.data[key].rectangle_width / self.params.meter_scale) + 1


    def make_filtered_sample_source(self, filter_boundaries, data_key, source_key=None, convert_to_utm=True):
        source = self.data[source_key]
        #if np.where((-180 < coords) and (coords < 180), True, False).prod(): #This is a check to see if number is latlon or UTM coordinate. Not very safe.
        if convert_to_utm:
            for i, filter_boundary in enumerate(filter_boundaries):
                filter_boundaries[i] = coordinates.convert_gps_to_utm(filter_boundary, origin_to_subtract=self.elevation_map.origin)
        print("Calling again from make_filtered_sample_source")
        self.data[data_key] = self.Samples(self, rx_vecs=source.rx_vecs, tx_vecs=source.tx_vecs, filter_boundaries=filter_boundaries, tx_metadata=source.tx_metadata if hasattr(source, 'tx_metadata') else None)

    
    def make_missing_samples(self, filtered_key, source_key=None):
        tx_vecs = []
        rx_vecs = []
        tx_metadata = []
        source_samples = self.data[source_key]
        filtered_samples = self.data[filtered_key]
        for i, tx in enumerate(source_samples.tx_vecs):
            if tx in filtered_samples.tx_vecs: continue
            tx_vecs.append(tx)
            rx_vecs.append(source_samples.rx_vecs[i])
            tx_metadata.append(source_samples.tx_metadata[i])
        samp = self.Samples(self, rx_vecs=rx_vecs, tx_vecs=tx_vecs, tx_metadata=tx_metadata)
        return samp


    def print_dataset_stats(self):
        print('Pixel Scale (m):', self.params.meter_scale)
        print('Image height:', self.img_height(), self.img_width())
        print('Meter Scale:', self.params.meter_scale)
        for key in self.data.keys():
            if key is not None and 'synthetic' in key:
                y_lengths = self.data[key].ordered_dataloader.dataset.tensors[1][:,:,0].sum(axis=1).cpu().numpy()
                x_lengths = (self.data[key].ordered_dataloader.dataset.tensors[0][:,:,3] != 0).sum(axis=1).cpu().numpy()
            else:
                y_lengths = [len(vec) for vec in self.data[key].tx_vecs]
                x_lengths = [len(vec) for vec in self.data[key].rx_vecs]
            print('Dataset %s' % key)
            print('  Length:' , len(y_lengths))
            print('  Number of transmitters:', np.unique(y_lengths, return_counts=True) )
            print('  Number of sensors in dataset:', np.unique(x_lengths, return_counts=True) )


    def get_rx_data_by_tx_location(self, source_key=None, combine_sensors=True, use_db_for_combination=True, required_limit=100):
        if source_key is None and hasattr(self, 'train_key'):
            source_key = self.train_key
        source_data = self.data[source_key]
        data = {}
        train_x, train_y = source_data.ordered_dataloader.dataset.tensors[:2]
        train_x, train_y = train_x.cpu().numpy(), train_y.cpu().numpy()
        coords = train_y[:,0,1:]
        for key in self.location_index_dict:
            name = self.location_index_dict[key]
            if isinstance(name, str):
                if 'bus' in name: continue
                if combine_sensors:
                    if 'nuc' in name and 'b210' in name:
                        name = name[:-6]
                    elif 'cellsdr' in name:
                        name = name.replace('cell','cbrs')
            rss = train_x[:,key,0]
            valid_inds = rss != 0
            if valid_inds.sum() < required_limit: continue
            sensor_loc = train_x[valid_inds][0,key,1:3]
            sensor_type = train_x[valid_inds][0,key,4]
            rss = rss[valid_inds]
            if combine_sensors and use_db_for_combination:
                min_rss, max_rss = self.get_min_max_rss_from_key(key)
                rss = rss * (max_rss - min_rss) + min_rss
            few_coords = coords[valid_inds]
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
        for name in data:
            few_coords, rss, sensor_loc, keys, sensor_type = data[name]
            sorted_inds = np.lexsort(few_coords.T)
            few_coords = few_coords[sorted_inds]
            rss = rss[sorted_inds]
            if combine_sensors and use_db_for_combination:
                min_rss = min(self.get_min_max_rss_from_key(key)[0] for key in keys)
                max_rss = min(self.get_min_max_rss_from_key(key)[1] for key in keys)
                rss = (rss - min_rss) / (max_rss - min_rss)
            row_mask = np.append([True], np.any(np.diff(few_coords,axis=0),1))
            data[name][0] = few_coords[row_mask]
            data[name][1] = rss[row_mask]
        return data

    def plot_separation(self, train_key=None, test_key=None, labels=None, plot_rx=False, save_plot=False, data_file=None):
        if train_key is None:
            train_key = self.train_key
        if test_key is None:
            test_key = self.test_key if self.test_key is not None else self.test_keys[0]
        all_train_tx = []
        all_test_tx = []
        all_train_rx = []
        all_test_rx = []
        for entry in self.data[train_key].rx_vecs:
            for rx in entry:
                all_train_rx.append(rx[1:])
        for entry in self.data[test_key].rx_vecs:
            for rx in entry:
                all_test_rx.append(rx[1:])
        for entry in self.data[train_key].tx_vecs:
            for tx in entry:
                all_train_tx.append(tx)
        for entry in self.data[test_key].tx_vecs:
            for tx in entry:
                all_test_tx.append(tx)
        all_train_rx = np.array(all_train_rx)
        all_test_rx = np.array(all_test_rx)
        all_train_tx = np.array(all_train_tx)
        all_test_tx = np.array(all_test_tx)
        train_tx_label = (train_key if labels is None else labels[0])
        test_tx_label = (test_key if labels is None else labels[1])
        rx_label = 'Sensors'
        plt.scatter(all_train_tx[:,0], all_train_tx[:,1], c='tab:red', label=train_tx_label, alpha=0.7, marker='+')
        plt.scatter(all_test_tx[:,0], all_test_tx[:,1], c='tab:green', label=test_tx_label, alpha=0.7, marker='o')
        if plot_rx:
            plt.scatter(all_train_rx[:,0], all_train_rx[:,1], c='blue', label=rx_label, alpha=0.2, marker='^', s=5)
            plt.scatter(all_test_rx[:,0], all_test_rx[:,1], c='blue', alpha=0.2, marker='^', s=5)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        if self.params.dataset_index == 6:
            plt.xlim(60,2290)
            plt.ylim(360,2230)
        plt.legend()
        plt.tight_layout()
        if data_file is None:
            data_file = 'dataset_scatter_scatter_%i.png' % (self.params.dataset_index + 1)
        if save_plot:
            plt.savefig(data_file)
            plt.cla()
        else:
            plt.show()

if __name__ == "__main__":
    #for ds  in [6,7,8]:
    for ds  in [6]:
      for split in ['random', 'grid10', 'radius100']:
        params = LocConfig(ds, data_split=split)
        rldataset = RSSLocDataset(params)
        rx_data = rldataset.data[rldataset.train_key].ordered_dataloader.dataset.tensors[0].cpu().numpy()
        tx_data = rldataset.data[rldataset.train_key].ordered_dataloader.dataset.tensors[1].cpu().numpy()
        rx = rx_data[rx_data[:,:,4] > 1]
        rx = rx[rx[:,0] > 0]
        all_rx = np.unique(rx[:,1:4], axis=0)
        rldataset.print_dataset_stats()
        embed()