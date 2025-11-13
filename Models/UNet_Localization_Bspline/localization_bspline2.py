from collections import OrderedDict
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from models_bspline2 import *
from dataset2 import RSSLocDataset
from attacker import get_random_attack_vec, worst_case_attack, get_random_attack_vec_spec_attack
from attacker import get_random_attack_vec_spec_sensor, worst_case_attack_spec_sensor
from sklearn.cluster import DBSCAN

from scipy.stats import spearmanr
import re
import pandas as pd

import csv
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, label
from itertools import permutations

def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def save_image(image_array, filename):
    """Helper function to save an image from numpy array."""
    
    # Squeeze the image if it has a singleton dimension (e.g., (1, height, width))
    if image_array.shape[0] == 1:
        image_array = image_array.squeeze(0)  # Squeeze the first dimension

    # Check if the array is a 3-channel image (e.g., RGB)
    if image_array.ndim == 3 and image_array.shape[0] == 3:
        # Transpose from (3, height, width) to (height, width, 3) for RGB
        image_array = image_array.transpose(1, 2, 0)

    plt.imshow(image_array, cmap='gray' if image_array.ndim == 2 else None)
    plt.axis('off')  # Turn off axis
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def normalize_image_channel(image_channel):
    """
    Normalizes a numpy array of shape (1, height, width), and keeps the shape intact.

    Parameters:
    - image_channel (numpy.ndarray): A numpy array of shape (1, height, width) to be normalized.

    Returns:
    - numpy.ndarray: A normalized numpy array with the same shape (1, height, width).
    """
    # Ensure the input is in the shape (height, width)
    if len(image_channel.shape) != 2:
        raise ValueError("Input array must have shape (height, width)")

    # Normalize the data (min-max normalization)
    min_val = np.min(image_channel)
    max_val = np.max(image_channel)
    
    if max_val - min_val == 0:
        # Avoid division by zero in case all values are the same
        normalized_data = np.zeros_like(image_channel)
    else:
        normalized_data = (image_channel - min_val) / (max_val - min_val)

    # The shape remains (height, width)
    return normalized_data


def set_negatives_to_zero(image_channel):
    """
    Sets all negative values in a numpy array of shape (height, width) to zero.

    Parameters:
    - image_channel (numpy.ndarray): A numpy array of shape (height, width).

    Returns:
    - numpy.ndarray: The array with negative values set to zero, with the same shape (height, width).
    """
    # Ensure the input is in the shape (height, width)
    if len(image_channel.shape) != 2:
        raise ValueError("Input array must have shape (height, width)")

    # Set negative values to zero
    image_channel[image_channel < 0] = 0

    return image_channel

class DLLocalization():
    def __init__(
            self, 
            rss_loc_dataset: RSSLocDataset, 
            loss_object=SlicedEarthMoversDistance(num_projections=200, scaling=10),
            lr=1e-4,
            should_use_augmented = True
        ):
        #initializing it with rldataset, loss_object=loss_func. in our case, it is the object and the loss function is Sliced earth moverr distance
        self.params = rss_loc_dataset.params
        #take all the parameters from dataset
        self.rss_loc_dataset = rss_loc_dataset
        self.should_use_augmented = should_use_augmented
        #keep the dataset object
        if self.params.include_elevation_map:
            self.rss_loc_dataset.make_elevation_tensors()
        #if you have to make elevation map, ask the object to make elevation tensors
        #it actually convert stored images to tensors
        #for both elevation and buildings
        self.device = self.rss_loc_dataset.params.device
        self.img_size = np.array([self.rss_loc_dataset.img_height(), self.rss_loc_dataset.img_width()])
        #calculate image height and width from rectrangle height and width of the train set
        print(" Image Size = ", self.img_size)
        self.loss_func = loss_object
        #assign the loss function name with parameters

        self.total_sensors = len(self.rss_loc_dataset.location_index_dict)

        self.build_model(self.params.arch, lr=lr)
        #it just initialize model and set the optimizers

    def build_model(self, arch='unet_ensemble', output_channel=1, lr=1e-4):
        """
        Initialize the UNet Ensemble model and setup optimizer
        """
        self.learn_locs = False
        #dunno
        self.calc_distance = False
        #dunno
        n_channels = 1 if not self.params.include_elevation_map else 2
        #separate channel for elevation_map
        #total 1 channel made from coordinates, if include map then two channels
        arch_without_number = ''.join(i for i in arch if not i.isdigit())
        #if architecture did mention any channel number
        channel_scale = get_trailing_number(arch)
        channel_scale = channel_scale if channel_scale is not None else 32
        #by default 32 channels
        depth = 4
        # four layers in total
        #by default only one output channel
        #img_size from the rectangle width and height of the train set data
        #num_models says whether it will be ensemble or not
        #channel scale means value of n
        #input_resolution = how many meters in each pixel
        #extracted elevation tensor at the end
        if arch_without_number == 'unet':
            if self.params.include_elevation_map:
                net_model = EnsembleLocalization(self.params, n_channels, output_channel, self.img_size, self.device, num_models=1, channel_scale=channel_scale, input_resolution=self.params.meter_scale, depth=depth, elevation_map=self.rss_loc_dataset.elevation_tensors[0], total_sensor= self.total_sensors)
            else:
                net_model = EnsembleLocalization(self.params, n_channels, output_channel, self.img_size, self.device, num_models=1, channel_scale=channel_scale, input_resolution=self.params.meter_scale, depth=depth, elevation_map=None, total_sensor= self.total_sensors)
        elif arch_without_number == 'unet_tiny':
            net_model = EnsembleLocalization(self.params, n_channels, output_channel, self.img_size, self.device, num_models=1, channel_scale=8, input_resolution=self.params.meter_scale, depth=3)
        elif arch_without_number == 'unet_ensemble':
            net_model =  EnsembleLocalization(self.params, n_channels, output_channel, self.img_size, self.device, num_models=5, channel_scale=channel_scale, input_resolution=self.params.meter_scale, depth=depth)
        elif arch == 'mlp':
            net_model = MLPLocalization(self.rss_loc_dataset.max_num_rx)
        else:
            raise NotImplementedError
        self.model = net_model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.optimizer = optimizer


    def predict_img_many(self, x_vecs):
        if isinstance(self.loss_func, CoMLoss):
            pred_vecs = self.model.com_predict(x_vecs)
        else:
            pred_vecs = self.model.predict(x_vecs)
        return pred_vecs[:,:2]

    def set_rss_tensor(self):
        if not hasattr(self, 'rss_tensor'):
            train_dataloader = self.rss_loc_dataset.data[self.rss_loc_dataset.train_key].ordered_dataloader
            rss_inds = train_dataloader.dataset.tensors[0][:,:,0] != 0
            all_rss = train_dataloader.dataset.tensors[0][:,:,0][rss_inds]
            self.rss_tensor = torch.quantile(all_rss, torch.tensor([0.1, 0.9], device=self.device))

    def test(self, test_key=None, dataloader=None, num_power_repeats=1, save_images=False, apply_wc_attack=False):
        """Evaluate model on the given dataloader dataset or test keys
        
        Args:
            dataloader      torch.DataLoader -- data to evaluate
            y_vecs          list<np.array> -- ground truth for locations
            num_power_repeats  int -- number of times to repeat testset, if assigning random power each iteration, to get avg performance
        return:
            total_loss      float -- loss from testset
            best_results    dict -- results from best setting of thresh and suppression_size   
            min_fn          float -- misdetection rate
            min_fp          float -- false alarm rate
        """
        self.model.eval()
        #Puts model in evaluation mode.
        all_x_images = []
        all_y_images = []
        all_pred_images = []
        all_pred_vecs = []
        all_error_vecs = []
        #To store outputs across all num_power_repeats
        if dataloader is None or test_key is not None:
            dataloader = self.rss_loc_dataset.data[test_key].ordered_dataloader
        for _ in range(num_power_repeats):
            #To get stable results, test multiple times and average results.
            repeat_pred_vecs = []
            repeat_error_vecs = []
            repeat_x_images = []
            repeat_y_images = []
            repeat_pred_images = []
            #Store predictions, errors, images for each repeat.
            for t, sample in enumerate(dataloader):
                x_vecs = sample[0].to(self.device)
                y_vecs = sample[1].to(self.device)

                # print(" In Localization test, X_vecs = ",type(x_vecs), " and shape of X_vecs = ",x_vecs.shape)
                # print(" In Localization test, y_vecs = ",type(y_vecs), " and shape of y_vecs = ",y_vecs.shape)

                if save_images:
                    pred_imgs, x_img, y_img = self.model((x_vecs, y_vecs))
                    if isinstance(self.loss_func, CoMLoss):
                        pred_vecs = self.model.com_predict(pred_imgs, input_is_pred_img=True)
                    else:
                        pred_vecs = self.model.predict(pred_imgs, input_is_pred_img=True)
                    #pred_vecs = self.model.predict(pred_imgs, input_is_pred_img=True)
                    repeat_x_images.append(x_img.detach().cpu().numpy())
                    repeat_y_images.append(y_img.detach().cpu().numpy())
                    repeat_pred_images.append(pred_imgs.detach().cpu().numpy())
                else:
                    if isinstance(self.loss_func, CoMLoss):
                        pred_vecs = self.model.com_predict(x_vecs)
                    else:
                        pred_vecs = self.model.predict(x_vecs)
                repeat_pred_vecs.append(pred_vecs.detach().cpu().numpy())
                repeat_error_vecs.append(torch.linalg.norm(pred_vecs[:,:2] - y_vecs[:,0,1:3], dim=1).detach().cpu().numpy())
            if save_images:
                all_x_images.append(np.concatenate(repeat_x_images))
                all_y_images.append(np.concatenate(repeat_y_images))
                all_pred_images.append(np.concatenate(repeat_pred_images))
            all_pred_vecs.append(np.concatenate(repeat_pred_vecs))
            all_error_vecs.append(np.concatenate(repeat_error_vecs))
        all_pred_vecs = np.array(all_pred_vecs)
        print("Error on pixels = ", all_error_vecs)
        all_error_vecs = np.array(all_error_vecs) * self.params.meter_scale
        print("Error after multiplying with meters: ", all_error_vecs)

        results = {'preds': all_pred_vecs, 'error': all_error_vecs}
        if save_images:
            results['x_imgs'] = np.array(all_x_images)
            results['y_imgs'] = np.array(all_y_images)
            results['pred_imgs'] = np.array(all_pred_images)
        return results
    
    def test_and_get_results(self, test_key=None, dataloader=None, num_power_repeats=1, save_images=False, apply_wc_attack=False, save_dir=None, passing_key= None):
        """Evaluate model on the given dataloader dataset or test keys
        
        Args:
            dataloader      torch.DataLoader -- data to evaluate
            y_vecs          list<np.array> -- ground truth for locations
            num_power_repeats  int -- number of times to repeat testset, if assigning random power each iteration, to get avg performance
        return:
            total_loss      float -- loss from testset
            best_results    dict -- results from best setting of thresh and suppression_size   
            min_fn          float -- misdetection rate
            min_fp          float -- false alarm rate
        """
        if save_dir is not None:
            folder_name = 'loss'
            parent_dir = os.path.dirname(save_dir)
            loss_folder = os.path.join(parent_dir, folder_name)
            os.makedirs(loss_folder, exist_ok=True)
            loss_file = os.path.join(loss_folder, 'losses.csv')
        else:
            loss_folder = 'loss'
            os.makedirs(loss_folder, exist_ok=True)
            loss_file = os.path.join(loss_folder, 'losses.csv')

        
        with open(loss_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sample ID', 'Original Loss', 'Perturbed Loss', 'Num_Tx'])      

            self.model.eval()
            #Puts model in evaluation mode.
            all_x_images = []
            all_pert_x_images = []
            all_y_images = []
            all_pred_images = []
            all_pert_pred_images = []
            all_pred_vecs = []
            all_pert_pred_vecs = []
            all_error_vecs = []
            all_pert_error_vecs = []
            all_true_num_tx = []
            pred_losses = []
            pert_pred_losses = []
            all_y_coords = []
            all_pred_y_coords = []
            all_pert_pred_y_coords = []
            all_pert_grad = []

            all_true_coords = []
            all_pred_coords = []


            #To store outputs across all num_power_repeats
            if dataloader is None or test_key is not None:
                dataloader = self.rss_loc_dataset.data[test_key].ordered_dataloader
            for _ in range(num_power_repeats):
                #To get stable results, test multiple times and average results.
                repeat_pred_vecs = []
                repeat_true_vecs = []
                repeat_error_vecs = []
                repeat_x_images = []
                repeat_y_images = []
                repeat_pred_images = []
                #Store predictions, errors, images for each repeat.
                for t, sample in enumerate(dataloader):
                    x_vecs = sample[0].to(self.device)
                    y_vecs = sample[1].to(self.device)

                    print("x_vecs shape = ", x_vecs.shape)
                    print("y_vecs shape = ", y_vecs.shape)

                    # print(" In Localization test, X_vecs = ",type(x_vecs), " and shape of X_vecs = ",x_vecs.shape)
                    # print(" In Localization test, y_vecs = ",type(y_vecs), " and shape of y_vecs = ",y_vecs.shape)

                    if save_images:
                        # Enable gradient computation for inputs
                        x_vecs.requires_grad_(True)
                        y_vecs.requires_grad_(True)

                        # Freeze model parameters by disabling gradient computation for them
                        for param in self.model.parameters():
                            param.requires_grad = False

                        # Enable gradient calculation for inputs
                        with torch.set_grad_enabled(True):
                            pred_imgs, x_img, y_img = self.model((x_vecs, y_vecs))
                            if isinstance(self.loss_func, CoMLoss):
                                pred_vecs = self.model.com_predict(pred_imgs, input_is_pred_img=True)
                            else:
                                pred_vecs = self.get_predicted_vec(pred_imgs, input_is_pred_img=True)
                            #pred_vecs = self.model.predict(pred_imgs, input_is_pred_img=True)


                        repeat_x_images.append(x_img.detach().cpu().numpy())
                        repeat_y_images.append(y_img.detach().cpu().numpy())
                        repeat_pred_images.append(pred_imgs.detach().cpu().numpy())
                        for idx in range(x_vecs.size(0)):
                            y_img_sample = y_img[idx].unsqueeze(0)
                            pred_img_sample = pred_imgs[idx].unsqueeze(0)
                            y_vec_sample = y_vecs[idx].unsqueeze(0)

                            # print("Pred Image Idx shape = ", pred_imgs[idx].shape)
                            image = pred_imgs[idx, 0]
                            peaks, _ = self.detect_clusters_and_peaks(image)
                            if len(peaks) == 0:
                                peaks, _ = self.detect_clusters_and_peaks2(image)
                            
                            # print("True vec = ", y_vecs[idx].detach().cpu().numpy())
                            # print("Predicted Vecs = ", peaks)
                            # all_true_coords.append(y_vecs[idx].detach().cpu().numpy())
                            # all_pred_coords.append(peaks)
                            true_vec = y_vecs[idx].detach().cpu().numpy()
                            pred_vec = peaks  # Already like [[x1, y1], [x2, y2]]

                            true_coords = []
                            for i in range(true_vec.shape[0]):
                                if true_vec[i][0] == 1:
                                    true_coords.append([true_vec[i][1], true_vec[i][2]])

                            pred_coords = [list(p) for p in pred_vec]

                            all_true_coords.append(true_coords)
                            all_pred_coords.append(pred_coords)
                        

                    else:
                        if isinstance(self.loss_func, CoMLoss):
                            pred_vecs = self.model.com_predict(x_vecs)
                        else:
                            pred_vecs = self.model.predict(x_vecs)
                    repeat_pred_vecs.append(pred_vecs.detach().cpu().numpy())
                    # repeat_true_vecs.append()
                    repeat_error_vecs.append(torch.linalg.norm(pred_vecs[:,:2] - y_vecs[:,0,1:3], dim=1).detach().cpu().numpy())
                
                if save_images:
                    all_x_images.append(np.concatenate(repeat_x_images))
                    all_y_images.append(np.concatenate(repeat_y_images))
                    all_pred_images.append(np.concatenate(repeat_pred_images))
                all_pred_vecs.append(np.concatenate(repeat_pred_vecs))
                all_error_vecs.append(np.concatenate(repeat_error_vecs))
            all_pred_vecs = np.array(all_pred_vecs)
            # print("Error on pixels = ", all_error_vecs)
            all_error_vecs = np.array(all_error_vecs) * self.params.meter_scale
            # print("Error after multiplying with meters: ", all_error_vecs)

            # Distance function
            def dist(a, b):
                a = np.array(a)
                b = np.array(b)
                return np.linalg.norm(a - b)

            # Counters
            num_one_tx = 0
            true_pred_one_tx = 0
            num_two_tx = 0
            true_pred_two_tx = 0

            # Error list
            errors = []

            # Iterate over samples
            for true_set, pred_set in zip(all_true_coords, all_pred_coords):
                true_set = np.array(true_set)
                pred_set = np.array(pred_set)

                true_len = len(true_set)
                pred_len = len(pred_set)

                # Stats tracking
                if true_len == 1:
                    num_one_tx += 1
                    if pred_len == 1:
                        true_pred_one_tx += 1
                    elif pred_len > 1:
                        close_enough = True
                        for i in range(pred_len):
                            for j in range(i + 1, pred_len):
                                if np.linalg.norm(pred_set[i] - pred_set[j]) > 1.5:
                                    close_enough = False
                                    break
                            if not close_enough:
                                break
                        if close_enough:
                            true_pred_one_tx += 1
                elif true_len == 2:
                    num_two_tx += 1
                    if pred_len == 2:
                        true_pred_two_tx += 1

                sample_errors = []

                if true_len == 1 and pred_len == 1:
                    sample_errors.append(np.linalg.norm(true_set[0] - pred_set[0]))


                elif true_len == 1 and pred_len > 1:
                    dists = [np.linalg.norm(true_set[0] - pred) for pred in pred_set]
                    sample_errors.append(min(dists))
                    # print(abcd)

                elif true_len == 2 and pred_len == 1:
                    dists = [np.linalg.norm(true - pred_set[0]) for true in true_set]
                    sample_errors.append(min(dists))


                elif true_len == 2 and pred_len == 2:
                    used_pred = set()
                    for true in true_set:
                        dists = [np.linalg.norm(true - pred_set[i]) if i not in used_pred else np.inf
                                for i in range(pred_len)]
                        best_idx = np.argmin(dists)
                        sample_errors.append(dists[best_idx])
                        used_pred.add(best_idx)

                elif true_len == 2 and pred_len > 2:
                    used_pred = set()
                    for true in true_set:
                        dists = [np.linalg.norm(true - pred_set[i]) if i not in used_pred else np.inf
                                for i in range(pred_len)]
                        best_idx = np.argmin(dists)
                        sample_errors.append(dists[best_idx])
                        used_pred.add(best_idx)

                else:
                    for true in true_set:
                        dists = [np.linalg.norm(true - pred) for pred in pred_set]
                        sample_errors.append(min(dists))

                
                errors.append(np.mean(sample_errors))
            errors_np = np.array(errors) * self.params.meter_scale
            

            # Print stats
            print("========== Dataset Statistics ==========")
            print("For key = ", passing_key)
            print(f"Total Samples: {len(all_true_coords)}")
            print(f"One TX Samples: {num_one_tx}")
            print(f"Correctly Predicted One TX: {true_pred_one_tx}")
            print(f"Two TX Samples: {num_two_tx}")
            print(f"Correctly Predicted Two TX: {true_pred_two_tx}")
            if num_one_tx > 0:
                print(f"One TX Accuracy: {true_pred_one_tx / num_one_tx * 100:.2f}%")
            else:
                print("One TX Accuracy: N/A")
            if num_two_tx > 0:
                print(f"Two TX Accuracy: {true_pred_two_tx / num_two_tx * 100:.2f}%")
            else:
                print("Two TX Accuracy: N/A")

            # all_true_coords = np.array(all_true_coords)
            # all_pred_coords = np.array(all_pred_coords)
            results = {'true_vecs':all_true_coords, 'preds_vecs': all_pred_coords, 'error': errors_np}
            if save_images:
                results['x_imgs'] = np.array(all_x_images)
                results['y_imgs'] = np.array(all_y_images)
                results['pred_imgs'] = np.array(all_pred_images)
        return results


    def adv_train_step(self, x_vec, x_img, y_vec, epsilon=0.5):
        self.set_rss_tensor()
        rand_select = np.random.random()
        if rand_select < 0.5:
            return
        grad = x_img.grad.data.clone()
        adv_x = get_random_attack_vec(x_vec, grad, self.rss_tensor[0].item(), self.rss_tensor[1].item(), epsilon)
        pred_img, x_img, y_img = self.model((adv_x, y_vec))
        if isinstance(self.loss_func, nn.MSELoss):
            loss = self.loss_func(pred_img, y_img)
        else:
            loss = self.loss_func(pred_img, y_img, y_vec)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def eval_worst_attack(self, x_vecs, y_vecs):
        preds = worst_case_attack(x_vecs, y_vecs, self, self.img_size)


    def train_model(self, num_epochs, train_data_key=None, test_data_keys=None, verbose=True, save_model_file='', load_model=True, load_model_file='', restart_optimizer=False):
        #all the parameter: max_num_epochs=2000, save_model_file=PATH, load_model=should_load_model=True, restart_optimizer=restart_optimizer=False, load_model_file=PATH=samepath
        """Train model using train_data_dict, evaluation on test_data_dict

        Args:
            num_epochs          num epochs for training
            train_data_keys     str or List[str] of keys in rss_loc_dataset.data to train model on
            test_data_keys      str or List[str] of keys in rss_loc_dataset.data to evaluate model on
            save_model_file     str -- filename to save torch model
            load_model          bool -- if true, load the model in load_model_file or save_model_file before resuming training
            load_model_file     str -- filename to load torch model from
            restart_optimizer   bool -- if loading model, restart the optimizer rather than resume from save file
        """

        """
        self.test() is used only for calculating error. So during training, you will get to see missing "y_vec" values in the model
        """
        if train_data_key is None:
            train_data_key = self.rss_loc_dataset.train_key
            #taking train_key into the list
        if test_data_keys is None:
            if len(self.rss_loc_dataset.test_keys) > 0:
                test_data_keys = self.rss_loc_dataset.test_keys
            else:
                test_data_keys = [self.rss_loc_dataset.test_key]

        if self.params.tirem_augment_two_tx and self.params.tirem_two_tx_only_synthetic_on_train and self.should_use_augmented:
            # print("Came Here with Train)
            train_data_key = self.rss_loc_dataset.train_key_augmented
            test_data_keys = self.rss_loc_dataset.test_keys_augmented

        if self.params.tirem_augment_two_tx and not self.params.tirem_two_tx_only_synthetic_on_train and self.params.tirem_two_tx_specific_augmentation and self.should_use_augmented:
            # print("Came Here with Train)
            train_data_key = self.rss_loc_dataset.train_key_augmented
            test_data_keys = self.rss_loc_dataset.test_keys_augmented
        if self.params.tirem_augment_two_tx and not self.params.tirem_two_tx_only_synthetic_on_train and not self.params.tirem_two_tx_specific_augmentation and self.should_use_augmented:
            train_data_key = self.rss_loc_dataset.train_key_augmented

        if self.params.tirem_augment_two_tx and not self.params.tirem_two_tx_only_synthetic_on_train and self.params.train_two_tx_tirem_with_ood_samples and self.should_use_augmented:
            # print("Came Here with Train)
            train_data_key = self.rss_loc_dataset.train_key_augmented
            test_data_keys = self.rss_loc_dataset.test_keys_augmented
        print("Came Here with Train data key = ", train_data_key, " and test keys = ", test_data_keys)

        # for data in self.rss_loc_dataset.data[train_data_key].dataloader:
        #     print(data)

        temp_tst_train_y = self.rss_loc_dataset.data[train_data_key].dataloader.dataset.tensors[1]
        print("Train y shape = ", temp_tst_train_y.shape, " and two transmitters = ", torch.sum(torch.sum(temp_tst_train_y[:, :, 0], dim=1) == 2).item())

        temp_tst_val_y = self.rss_loc_dataset.data[test_data_keys[1]].dataloader.dataset.tensors[1]
        print("Validation y shape = ", temp_tst_val_y.shape, " and two transmitters = ", torch.sum(torch.sum(temp_tst_val_y[:, :, 0], dim=1) == 2).item())
               
        
        # print(abcd)
        #if contains eval set then more than one keys otherwise just one
        train_dataloader = self.rss_loc_dataset.data[train_data_key].dataloader
        #taking the dataloader object to load data
        rss_inds = train_dataloader.dataset.tensors[0][:,:,0] != 0
        #tensors[0] has the shape (n_samples, n_rx_data, 5)
        #taking only those indices with non-zero rss values
        all_rss = train_dataloader.dataset.tensors[0][:,:,0][rss_inds]
        #This now contains only the non-zero RSS values from the dataset. Essentially, it flattens and filters the RSS values from the dataset.
        self.adv_rss_vec = np.quantile(all_rss.cpu(), [0.1, 0.9])
        #This stores a tuple containing the 10th and 90th percentile of the non-zero RSS values in the dataset.
        test_errors = {key:[] for key in test_data_keys}
        #dictionary to store test_errors
        train_loss_arr = np.zeros(num_epochs)
        #train loss for each epoch
        epoch = 0
        best_epoch = 0


    #the whole code section is used to load previously saved models
        if len(load_model_file) == 0:
            load_model_file = save_model_file
        #if preexisting model is there
        for model_ext in ['model_train_val.']:
            load_model_file_ext = load_model_file.replace('model.', model_ext)
            if load_model and os.path.exists(load_model_file_ext):
                try:
                    checkpoint = torch.load(load_model_file_ext)
                    self.load_model(load_model_file_ext)
                    print('Loading model from %s' % load_model_file_ext)
                    if not restart_optimizer:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        epoch = checkpoint['epoch']
                        best_epoch = checkpoint['epoch']
                    break
                except:
                    pass

        saved_rss = train_dataloader.dataset.tensors[0][:,:,0].clone()
        #take all the rss values in saved_rss
        while epoch < num_epochs and epoch - best_epoch <= self.params.better_epoch_limit:
            print(" Training Epoch = ",epoch, " / ",num_epochs)
            # print("Here Goes Training")
            self.model.train()
            epoch_loss = 0
            for t, sample in enumerate(train_dataloader):
                    X_vec = sample[0].to(self.device)
                    # print(X_vec)
                    y_vec = sample[1].to(self.device)
                    print("Training batch = ", t, " X_vec = ", X_vec.shape, " y_vec = ", y_vec.shape)
                    # print(" In Localization train_model, X_vec = ",type(X_vec), " and shape of X_vec = ",X_vec.shape)
                    # print(" In Localization train_model, y_vec = ",type(y_vec), " and shape of y_vec = ",y_vec.shape)
                    pred_img, x_img, y_img = self.model((X_vec, y_vec))
                    if isinstance(self.loss_func, nn.MSELoss):
                        loss = self.loss_func(pred_img, y_img)
                    else:
                        loss = self.loss_func(pred_img, y_img, y_vec)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    if self.params.adv_train:
                        # print("Adversarial training activated")
                        self.adv_train_step(X_vec, x_img, y_vec)

            self.model.eval()
            if verbose:
                endchar = '\n'
            else:
                endchar = '\r'
            if epoch % 10 == 0:
                res = self.test(dataloader=train_dataloader)
                res_max = res['preds'][0,:,2]
                center_point =  torch.tensor([1100,1200]).to(self.device)
                #tr_radial_dist = torch.linalg.norm(train_dataloader.dataset.tensors[1].squeeze()[:,1:3] - center_point, axis=1)
                #plt.scatter(res_max, tr_radial_dist.cpu().detach())
                #plt.show()
                correlation = spearmanr(res['error'][0],res['preds'][0,:,2])[0]
                train_err = res['error'].mean()
                quants = np.quantile(res['error'], [0.25, 0.5, 0.75, 1])
                print('Ep%i Train Loss:%.2e  Tr Mean: %.1f    25/50/75/100%% %.1f %.1f %.1f %.1f %.2f ' % ( epoch+1, epoch_loss, train_err, quants[0], quants[1], quants[2], quants[3], correlation), end=endchar)

            epoch_loss = epoch_loss / (t+1)
            train_loss_arr[epoch] = epoch_loss
            result_string = 'Ep%i Tr%.2e  ' % (epoch+1, epoch_loss)
            should_print = False
            for test_key in test_data_keys:
                res = self.test(test_key)
                correlation = spearmanr(res['error'][0],res['preds'][0,:,2])[0]
                test_err = res['error'].mean()
                test_errors[test_key].append(test_err)

                quants = np.quantile(res['error'], [0.25, 0.5, 0.75, 1])
                result_string += '%s: Mean:%.1f  25/50/75/100%% %.1f %.1f %.1f %.1f %.2f ' % (test_key, test_err, quants[0], quants[1], quants[2], quants[3], correlation)

                if test_err == min(test_errors[test_key]):
                    if 'train_val' in test_key:
                        save_string = 'train_val'
                        best_epoch = epoch
                        # print("Here Goes Validation")
                    elif 'test' in test_key:
                        save_string = 'test'
                        # print("Here Goes the test set")
                    else:
                        continue
                    if len(save_model_file) > 0:
                        torch.save({
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'epoch': epoch,
                                    }, save_model_file.replace('model.', 'model_%s.' % save_string))
            print(result_string, end=endchar)# if should_print else '\r')
            epoch += 1
            if epoch - best_epoch > self.params.better_epoch_limit and best_epoch != 0:
                break
        return train_loss_arr, test_errors
    

    def load_model(self, model_path, device=None):
        checkpoint = torch.load(model_path, map_location=device)
        if 'down1.maxpool_conv.1.double_conv.0.weight' in checkpoint['model_state_dict']:
            new_model_checkpoint = OrderedDict()
            for key in checkpoint['model_state_dict']:
                layer_type = key.split('.')[0]
                if 'up' in layer_type or 'down' in layer_type:
                    layer_number = int(layer_type[-1])
                    new_key = key.replace(layer_type, layer_type[:-1] + ('s.%i' % (layer_number-1)))
                    new_model_checkpoint[new_key] = checkpoint['model_state_dict'][key]
                else:
                    new_model_checkpoint[key] = checkpoint['model_state_dict'][key]
            self.model.load_state_dict(new_model_checkpoint, strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.params.device)




    def load_existing_model(self, num_epochs = 20000, train_data_key=None, test_data_keys=None, verbose=True, save_model_file='', load_model=True, load_model_file='', restart_optimizer=False):
        #all the parameter: max_num_epochs=2000, save_model_file=PATH, load_model=should_load_model=True, restart_optimizer=restart_optimizer=False, load_model_file=PATH=samepath
        """Train model using train_data_dict, evaluation on test_data_dict

        Args:
            num_epochs          num epochs for training
            train_data_keys     str or List[str] of keys in rss_loc_dataset.data to train model on
            test_data_keys      str or List[str] of keys in rss_loc_dataset.data to evaluate model on
            save_model_file     str -- filename to save torch model
            load_model          bool -- if true, load the model in load_model_file or save_model_file before resuming training
            load_model_file     str -- filename to load torch model from
            restart_optimizer   bool -- if loading model, restart the optimizer rather than resume from save file
        """
        if train_data_key is None:
            train_data_key = self.rss_loc_dataset.train_key
            #taking train_key into the list
        if test_data_keys is None:
            if len(self.rss_loc_dataset.test_keys) > 0:
                test_data_keys = self.rss_loc_dataset.test_keys
            else:
                test_data_keys = [self.rss_loc_dataset.test_key]


    #the whole code section is used to load previously saved models
        if len(load_model_file) == 0:
            load_model_file = save_model_file
        #if preexisting model is there
        for model_ext in ['model_train_val.']:
            load_model_file_ext = load_model_file.replace('model.', model_ext)
            if load_model and os.path.exists(load_model_file_ext):
                try:
                    checkpoint = torch.load(load_model_file_ext)
                    self.load_model(load_model_file_ext)
                    print('Loading model from %s' % load_model_file_ext)
                    if not restart_optimizer:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        epoch = checkpoint['epoch']
                        best_epoch = checkpoint['epoch']
                    break
                except:
                    pass


        return self.model
    
    def save_combined_image(self, y_img, pred_img, pert_pred_img, x_img, pert_x_img, sample_id, 
                            pred_loss, pert_pred_loss, filename, y_coords, pred_coords, 
                            pert_pred_coords, predicted_COM, perturbed_predicted_COM, abs_pert_grad):
        """Combine the true input, perturbed input, ground truth, predicted, and center of mass (COM) images,
        and save them as a single image in three rows.
        
        Args:
            y_img (np.array): Ground truth image (single-channel).
            pred_img (np.array): Predicted image by the model (single-channel).
            pert_pred_img (np.array): Perturbed predicted image by the model (single-channel).
            x_img (np.array): True input image (single-channel).
            pert_x_img (np.array): Perturbed input image (single-channel).
            sample_id (int): Sample ID for labeling the images.
            pred_loss (float): Loss for the predicted image.
            pert_pred_loss (float): Loss for the perturbed predicted image.
            filename (str): File path to save the combined image.
            y_coords (np.array): Ground truth coordinates (x, y).
            pred_coords (np.array): Predicted coordinates (x, y).
            pert_pred_coords (np.array): Perturbed predicted coordinates (x, y).
            predicted_COM (np.array): Predicted center of mass image (single-channel).
            perturbed_predicted_COM (np.array): Perturbed predicted center of mass image (single-channel).
        """
        # Squeeze the images if they have a singleton dimension
        y_img = y_img.squeeze(0) if y_img.shape[0] == 1 else y_img
        pred_img = pred_img.squeeze(0) if pred_img.shape[0] == 1 else pred_img
        pert_pred_img = pert_pred_img.squeeze(0) if pert_pred_img.shape[0] == 1 else pert_pred_img
        x_img = x_img.squeeze(0) if x_img.shape[0] == 1 else x_img
        pert_x_img = pert_x_img.squeeze(0) if pert_x_img.shape[0] == 1 else pert_x_img
        predicted_COM = predicted_COM.squeeze(0) if predicted_COM.shape[0] == 1 else predicted_COM
        perturbed_predicted_COM = perturbed_predicted_COM.squeeze(0) if perturbed_predicted_COM.shape[0] == 1 else perturbed_predicted_COM
        
        def format_coords(coords):
            captions = []
            for coord in coords:
                if coord[0] == 1:  # Active transmitter
                    captions.append(f"({coord[1]:.2f}, {coord[2]:.2f})")
            return " | ".join(captions) if captions else "No active transmitters"
        

        y_coords_caption = format_coords(y_coords)
        pred_coords_caption = " | ".join([f"({c[0]:.2f}, {c[1]:.2f})" for c in pred_coords])
        pert_pred_coords_caption = " | ".join([f"({c[0]:.2f}, {c[1]:.2f})" for c in pert_pred_coords])


        # Create a figure with 3 rows: 
        # Row 1 -> True Input and Ground Truth
        # Row 2 -> True Input, Predicted Image, Predicted_COM
        # Row 3 -> Perturbed Input, Perturbed Predicted, Perturbed_Predicted_COM
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Row 1 (True input and Ground Truth)
        axes[0, 0].imshow(x_img, cmap='gray')
        axes[0, 0].set_title(f"True Input\nSample {sample_id}")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(y_img, cmap='gray')
        # y_coord_label = f"Coords: ({y_coords[0]:.2f}, {y_coords[1]:.2f})"
        axes[0, 1].set_title(f"Ground Truth\n{y_coords_caption}")
        axes[0, 1].axis('off')

        # Display abs_pert_grad in the third spot on the first row without an image
        axes[0, 2].text(0.5, 0.5, f"Sum of Absolute Perturbation\n{abs_pert_grad:.4f}", 
                        ha='center', va='center', fontsize=12, wrap=True)
        axes[0, 2].axis('off')  # No image here, only text

        # Row 2 (True Input, Predicted Image, Predicted_COM)
        axes[1, 0].imshow(x_img, cmap='gray')
        axes[1, 0].set_title(f"True Input\nSample {sample_id}")
        axes[1, 0].axis('off')

        # pred_coord_label = f"Coords: ({pred_coords[0]:.2f}, {pred_coords[1]:.2f})"
        axes[1, 1].imshow(pred_img, cmap='gray')
        axes[1, 1].set_title(f"Predicted\nLoss: {pred_loss:.4f}")
        axes[1, 1].axis('off')

        axes[1, 2].imshow(predicted_COM, cmap='gray')
        axes[1, 2].set_title(f"Predicted COM\n{pred_coords_caption}")
        axes[1, 2].axis('off')

        # Row 3 (Perturbed Input, Perturbed Predicted, Perturbed_Predicted_COM)
        axes[2, 0].imshow(pert_x_img, cmap='gray')
        axes[2, 0].set_title(f"Perturbed Input\nSample {sample_id}")
        axes[2, 0].axis('off')

        # pert_pred_coord_label = f"Coords: ({pert_pred_coords[0]:.2f}, {pert_pred_coords[1]:.2f})"
        axes[2, 1].imshow(pert_pred_img, cmap='gray')
        axes[2, 1].set_title(f"Perturbed Predicted\nLoss: {pert_pred_loss:.4f}")
        axes[2, 1].axis('off')

        axes[2, 2].imshow(perturbed_predicted_COM, cmap='gray')
        axes[2, 2].set_title(f"Perturbed Predicted COM\n{pert_pred_coords_caption}")
        axes[2, 2].axis('off')
        
        # Save the combined image
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        print(f"Combined image saved to {filename}")
    
    def generate_ground_truth_like_image(self, pred_img, grid_size=3):
        """
        Generate a 'ground truth-like' image based on the center of mass from the predicted image.

        Args:
            pred_img (numpy array): The predicted UNet image.
            grid_size (int): Size of the neighborhood grid, default is 3x3.
        
        Returns:
            gt_like_img (numpy array): Ground truth-like image.
        """
        # Ensure pred_img is 2D
        original_shape = pred_img.shape
        if len(pred_img.shape) > 2:
            pred_img = np.squeeze(pred_img)  # Remove extra dimensions

        # Initialize the ground-truth-like image with zeros
        gt_like_img = np.zeros_like(pred_img)

        # Find the center of mass (row, col) of the predicted image
        center_row, center_col = center_of_mass(pred_img)

        # Round to the nearest integer (to locate the center pixel on a discrete grid)
        center_row = int(round(center_row))
        center_col = int(round(center_col))

        # Assign the center pixel a value of 0.92
        gt_like_img[center_row, center_col] = 0.92

        # Assign 0.01 to the 8 neighboring pixels (3x3 grid around the center)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:  # Skip the center pixel itself
                    continue
                # Make sure we don't go out of bounds
                if 0 <= center_row + i < gt_like_img.shape[0] and 0 <= center_col + j < gt_like_img.shape[1]:
                    gt_like_img[center_row + i, center_col + j] = 0.01

        # Reshape (unsqueeze) to match the original shape if needed
        while len(gt_like_img.shape) < len(original_shape):
            gt_like_img = np.expand_dims(gt_like_img, axis=0)

        return gt_like_img


    def generate_ground_truth_for_clusters(self, pred_img, grid_size=3):
        """
        Generate ground truth-like images for each cluster in the predicted image.

        Args:
            pred_img (numpy array): The predicted UNet image (may be 2D or higher-dimensional).
            grid_size (int): Size of the neighborhood grid, default is 3x3.

        Returns:
            gt_like_img (numpy array): Ground truth-like image for all clusters with the same dimensionality as input.
        """
        print(type(pred_img))
        print(pred_img.shape)
        # Ensure pred_img is 2D
        original_shape = pred_img.shape
        if len(pred_img.shape) > 2:
            pred_img = np.squeeze(pred_img)  # Remove extra dimensions (e.g., batch or channel dims)

        # Initialize the ground-truth-like image with zeros
        gt_like_img = np.zeros_like(pred_img)

        # Label each connected component (cluster)
        labeled_img, num_clusters = label(pred_img > 0.03)  # Use a threshold to separate clusters (e.g., > 0)
        if num_clusters == 0:
            labeled_img, num_clusters = label(pred_img > 0.03)  # Use a threshold to separate clusters (e.g., > 0)
        print("Num clusters: ", type(num_clusters), " and value = ",num_clusters)

        if num_clusters == 0:
            # No clusters detected; find the pixel with the highest intensity
            max_intensity_index = np.unravel_index(np.argmax(pred_img), pred_img.shape)
            center_row, center_col = max_intensity_index
            
            # Assign the center pixel a value of 0.92
            gt_like_img[center_row, center_col] = 0.92

            # Assign 0.01 to the 8 neighboring pixels (3x3 grid around the center)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:  # Skip the center pixel itself
                        continue
                    # Ensure we don't go out of bounds
                    if 0 <= center_row + i < gt_like_img.shape[0] and 0 <= center_col + j < gt_like_img.shape[1]:
                        gt_like_img[center_row + i, center_col + j] = 0.01

        else: 
            # Iterate through each cluster
            for cluster_label in range(1, num_clusters + 1):
                # Isolate the current cluster
                cluster_mask = (labeled_img == cluster_label)

                # Find the center of mass for the current cluster
                center = center_of_mass(cluster_mask)

                # Ensure we are only working with 2D coordinates
                if len(center) == 2:
                    center_row, center_col = center
                else:
                    raise ValueError("Center of mass calculation returned more than 2 values.")

                # Round to the nearest integer (to locate the center pixel on a discrete grid)
                center_row = int(round(center_row))
                center_col = int(round(center_col))

                # Assign the center pixel a value of 0.92
                if 0 <= center_row < gt_like_img.shape[0] and 0 <= center_col < gt_like_img.shape[1]:
                    gt_like_img[center_row, center_col] = 0.92

                    # Assign 0.01 to the 8 neighboring pixels (3x3 grid around the center)
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if i == 0 and j == 0:  # Skip the center pixel itself
                                continue
                            # Ensure we don't go out of bounds
                            if 0 <= center_row + i < gt_like_img.shape[0] and 0 <= center_col + j < gt_like_img.shape[1]:
                                gt_like_img[center_row + i, center_col + j] = 0.01

        # Reshape (unsqueeze) to match the original shape if needed
        while len(gt_like_img.shape) < len(original_shape):
            gt_like_img = np.expand_dims(gt_like_img, axis=0)

        return gt_like_img

    def get_predicted_vec(self, x, input_is_pred_img=False):
        # if input_is_pred_img:
        #     preds = x
        # else:
        #     single_model_training = self.single_model_training
        #     self.single_model_training = False
        #     preds = self.forward(x)
        #     self.single_model_training = single_model_training
        preds = x.clone()
        
        peaks, peak_locs = torch.max( preds.reshape((preds.shape[0], len(self.model.models), -1)), dim=-1)
        # The new shape is (batch_size, num_models, -1)
        #torch.max is applied along the last dimension (dim=-1), 
        # which finds the maximum value (peaks) and the corresponding index (peak_locs) for each element in the batch.
        peak_locs = unravel_indices(peak_locs, preds.shape[2:])
        #peak_locs are currently flat indices (due to the reshaping), but to get the actual 2D coordinates 
        # (row, column) of the peak in the original image, unravel_indices is used. It takes the flat indices and 
        # reshapes them back into the original 2D shape defined by preds.shape[2:].
        weighted_preds = (peak_locs * peaks.unsqueeze(-1)).sum(dim=1) / peaks.sum(dim=1, keepdim=True)
        #This line calculates a weighted average of the peak locations, using the corresponding peak values as weights.
        #peaks.unsqueeze(-1) adds an extra dimension to peaks so it can be broadcasted across peak_locs.
        #The product peak_locs * peaks weights the locations by their intensity values. The sum of these weighted 
        # locations is divided by the sum of the peak values, giving a weighted prediction of where the transmitter is likely located.
        weighted_preds = torch.hstack((torch.fliplr(weighted_preds), peaks.mean(dim=1, keepdim=True)))
        #torch.hstack horizontally stacks two tensors.
        ##torch.fliplr(weighted_preds) flips the weighted_preds tensor left-to-right (perhaps to switch between coordinate systems).
        #peaks.mean(dim=1, keepdim=True) calculates the mean intensity of the peaks across each batch element and appends it to the weighted predictions.
        #The final result is a tensor that contains the predicted (flipped) coordinates of the transmitter along with the mean peak 
        # intensity for each image in the batch.
        return weighted_preds
    #final output: weighted_preds = tensor([[1, 1, 1],   # First image: predicted coordinates [1, 1] and mean peak intensity 1
    #                     #[1, 1, 0.8]])  # Second image: predicted coordinates [1, 1] and mean peak intensity 0.8

    def detect_clusters_and_peaks(self, image):
        device = image.device

        # Convert image to NumPy for labeling
        image_np = image.detach().cpu().numpy()

        # # Calculate 30th percentile as threshold
        # threshold = np.percentile(image_np[image_np > 0], 30)

        # Label clusters based on the threshold
        labeled_array, num_features = label(image_np > 0.03)

        # Find the peak (maximum value location) for each cluster
        peaks = []
        for cluster_id in range(1, num_features + 1):
            cluster_mask = labeled_array == cluster_id
            cluster_values = image_np * cluster_mask
            peak_idx = np.unravel_index(np.argmax(cluster_values), cluster_values.shape)
            peaks.append([int(peak_idx[1]), int(peak_idx[0])])

        # Convert labeled array back to PyTorch tensor
        labeled_tensor = torch.from_numpy(labeled_array).to(device)

        return peaks, labeled_tensor

    def detect_clusters_and_peaks2(self,image):
        device = image.device

        # Convert image to NumPy for clustering
        image_np = image.detach().cpu().numpy()

        # Get coordinates and values of non-zero pixels
        coords = np.column_stack(np.nonzero(image_np))
        values = image_np[coords[:, 0], coords[:, 1]]

        # Use DBSCAN for clustering
        db = DBSCAN(eps=3, min_samples=5).fit(coords)  # eps and min_samples can be tuned
        labels = db.labels_

        # Initialize data structures
        peaks = []
        unique_labels = set(labels)

        for label_id in unique_labels:
            if label_id == -1:  # Noise points
                continue

            # Get points belonging to the current cluster
            cluster_coords = coords[labels == label_id]
            cluster_values = values[labels == label_id]

            # Find the peak (coordinate with the maximum value)
            max_idx = np.argmax(cluster_values)
            peak = tuple(cluster_coords[max_idx])
            peaks.append([int(peak[1]), int(peak[0])])

        return peaks, labels.reshape(image_np.shape)
    
    # def calculate_loss(self, y_vecs, pred_vecs):
    #     """
    #     Calculate the loss between y_vecs and pred_vecs.
    #     Args:
    #         y_vecs: List of lists, ground truth where the first column indicates presence.
    #         pred_vecs: List of lists, predicted coordinates.

    #     Returns:
    #         loss: Calculated loss as a scalar.
    #     """
    #     # Convert lists to PyTorch tensors
    #     y_vecs = torch.tensor(y_vecs, dtype=torch.float32)  # Shape: (N, 3)
    #     pred_vecs = torch.tensor(pred_vecs, dtype=torch.float32)  # Shape: (M, 2)

    #     # Filter active transmitters from y_vecs
    #     active_mask = y_vecs[:, 0] == 1
    #     y_active = y_vecs[active_mask, 1:]  # Shape: (num_active, 2)

    #     # If no transmitters are active, return zero loss
    #     if y_active.size(0) == 0:
    #         return torch.tensor(0.0), y_active.size(0)

    #     # Pair predictions with ground truths based on minimum distance
    #     distances = torch.cdist(y_active, pred_vecs)  # Shape: (num_active, M)
    #     assignment = distances.argmin(dim=1)  # Best match indices in pred_vecs for each y_active

    #     # Extract matched predictions
    #     pred_matched = pred_vecs[assignment]  # Shape: (num_active, 2)

    #     # Compute loss (MSE between matched pairs)
    #     matching_loss = F.mse_loss(pred_matched, y_active)

    #     # Identify unmatched predictions (extra predictions)
    #     unmatched_mask = torch.ones(pred_vecs.size(0), dtype=torch.bool, device=pred_vecs.device)
    #     unmatched_mask[assignment] = False
    #     extra_preds = pred_vecs[unmatched_mask]  # Shape: (num_extra, 2)

    #     # Compute penalty for extra predictions (MSE to [0, 0])
    #     if extra_preds.size(0) > 0:
    #         extra_penalty = F.mse_loss(extra_preds, torch.zeros_like(extra_preds))
    #     else:
    #         extra_penalty = torch.tensor(0.0, device=pred_vecs.device)

    #     total_loss = matching_loss + extra_penalty

    #     return total_loss.item(), y_active.size(0)


    def calculate_loss(self, y_vecs, pred_vecs):
        """
        Calculate the loss between y_vecs and pred_vecs, addressing three cases:
        1. Both y_vecs and pred_vecs have the same number of active transmitters.
        2. y_vecs has fewer active transmitters than pred_vecs.
        3. y_vecs has more active transmitters than pred_vecs.
        
        Args:
            y_vecs: List of lists, ground truth where the first column indicates presence.
            pred_vecs: List of lists, predicted coordinates.

        Returns:
            loss: Calculated loss as a scalar.
        """
        # Convert lists to PyTorch tensors
        y_vecs = torch.tensor(y_vecs, dtype=torch.float32)  # Shape: (N, 3)
        pred_vecs = torch.tensor(pred_vecs, dtype=torch.float32)  # Shape: (M, 2)

        # Filter active transmitters from y_vecs
        active_mask = y_vecs[:, 0] == 1
        y_active = y_vecs[active_mask, 1:]  # Shape: (num_active, 2)

        # If no transmitters are active, return zero loss
        if y_active.size(0) == 0:
            return torch.tensor(0.0), y_active.size(0)

        # Compute the pairwise distances
        distances = torch.cdist(y_active, pred_vecs)  # Shape: (num_active, M)

        # Three cases to handle
        num_active = y_active.size(0)
        num_pred = pred_vecs.size(0)

        if num_active == num_pred:
            # Case 1: Equal number of active and predicted transmitters
            assignment = distances.argmin(dim=1)  # Best match indices in pred_vecs for each y_active
            pred_matched = pred_vecs[assignment]  # Shape: (num_active, 2)
            matching_loss = F.mse_loss(pred_matched, y_active)

            total_loss = matching_loss

        elif num_active < num_pred:
            # Case 2: Fewer active transmitters than predicted
            assignment = distances.argmin(dim=1)  # Best match indices in pred_vecs for each y_active
            pred_matched = pred_vecs[assignment]  # Matched predictions for active transmitters

            # Compute loss for matched pairs
            matching_loss = F.mse_loss(pred_matched, y_active)

            # Compute penalty for extra predictions (unmatched predictions)
            unmatched_mask = torch.ones(pred_vecs.size(0), dtype=torch.bool, device=pred_vecs.device)
            unmatched_mask[assignment] = False
            extra_preds = pred_vecs[unmatched_mask]  # Shape: (num_extra, 2)
            extra_penalty = F.mse_loss(extra_preds, torch.zeros_like(extra_preds))

            total_loss = matching_loss + extra_penalty

        else:  # num_active > num_pred
            # Case 3: More active transmitters than predicted

            # Find the closest `num_pred` active transmitters using distance matrix
            closest_indices = distances.topk(k=num_pred, dim=0, largest=False).indices.squeeze(1)  # Shape: (num_pred,)
            y_closest = y_active[closest_indices]  # Closest ground truths for predictions

            # Compute loss for matched pairs
            matching_loss = F.mse_loss(pred_vecs, y_closest)

            # Compute penalty for extra active transmitters (unmatched ground truths)
            unmatched_y_mask = torch.ones(y_active.size(0), dtype=torch.bool, device=y_active.device)
            unmatched_y_mask[closest_indices] = False
            extra_y = y_active[unmatched_y_mask]  # Shape: (num_extra, 2)
            extra_penalty = F.mse_loss(extra_y, torch.zeros_like(extra_y))

            total_loss = matching_loss + extra_penalty

        return total_loss.item(), y_active.size(0)

    def draw_img_then_perturb(self, test_key=None, dataloader=None, num_power_repeats=1, save_images=False, apply_wc_attack=False, save_dir=None):
        """Evaluate model on the given dataloader dataset or test keys
        
        Args:
            dataloader      torch.DataLoader -- data to evaluate
            y_vecs          list<np.array> -- ground truth for locations
            num_power_repeats  int -- number of times to repeat testset, if assigning random power each iteration, to get avg performance
        return:
            total_loss      float -- loss from testset
            best_results    dict -- results from best setting of thresh and suppression_size   
            min_fn          float -- misdetection rate
            min_fp          float -- false alarm rate
        """
        # if sensors_to_perturb is not None:
        #     # Convert sensors_to_perturb array to a string, e.g., [1,2,3] -> "1_2_3"
        #     sensor_list = sensors_to_perturb.tolist()
        #     sensors_str = '_'.join(map(str, sensor_list))
        #     # Create the folder name by appending the sensors to the 'loss' name
        #     folder_name = f"loss_{sensors_str}"
        # else:
        #     # Default folder name if sensors_to_perturb is None
        #     folder_name = "loss"

        # Prepare the CSV file to store the losses
        if save_dir is not None:
            folder_name = 'loss'
            parent_dir = os.path.dirname(save_dir)
            loss_folder = os.path.join(parent_dir, folder_name)
            os.makedirs(loss_folder, exist_ok=True)
            loss_file = os.path.join(loss_folder, 'losses.csv')
        else:
            loss_folder = 'loss'
            os.makedirs(loss_folder, exist_ok=True)
            loss_file = os.path.join(loss_folder, 'losses.csv')




        # # Prepare the CSV file to store the losses
        # if save_dir is not None:
        #     loss_file = os.path.join(save_dir, 'losses.csv')
        # else:
        #     loss_file = 'losses.csv'
        
        with open(loss_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sample ID', 'Original Loss', 'Perturbed Loss', 'Num_Tx'])


            self.model.eval()
            all_x_images = []
            all_pert_x_images = []
            all_y_images = []
            all_pred_images = []
            all_pert_pred_images = []
            all_pred_vecs = []
            all_pert_pred_vecs = []
            all_error_vecs = []
            all_pert_error_vecs = []

            all_true_num_tx = []
            pred_losses = []
            pert_pred_losses = []

            all_y_coords = []
            all_pred_y_coords = []
            all_pert_pred_y_coords = []

            all_pert_grad = []

            if dataloader is None or test_key is not None:
                dataloader = self.rss_loc_dataset.data[test_key].ordered_dataloader

            sample_id = 0  # To track which sample is being processed
            for _ in range(num_power_repeats):
                repeat_pred_vecs = []
                repeat_pert_pred_vecs = []
                repeat_error_vecs = []
                repeat_pert_error_vecs = []
                repeat_x_images = []
                repeat_pert_x_images = []
                repeat_y_images = []
                # repeat_perty_images = []
                repeat_pred_images = []
                repeat_pert_pred_images = []

                batch_true_num_tx = []
                batch_pred_losses = []
                batch_pert_pred_losses = []

                batch_y_coords = []
                batch_pred_y_coords = []
                batch_pert_pred_y_coords = []

                batch_pert_grad = []

                for t, sample in enumerate(dataloader):
                    # if t >= 1:  # Stop after processing the first 5 batches
                    #     break
                    x_vecs = sample[0].to(self.device)
                    y_vecs = sample[1].to(self.device)
                    

                    if save_images:
                        # Enable gradient computation for inputs
                        x_vecs.requires_grad_(True)
                        y_vecs.requires_grad_(True)


                        # Freeze model parameters by disabling gradient computation for them
                        for param in self.model.parameters():
                            param.requires_grad = False

                        # Enable gradient calculation for inputs
                        with torch.set_grad_enabled(True):
                            pred_imgs, x_img, y_img = self.model((x_vecs, y_vecs))
                            if isinstance(self.loss_func, CoMLoss):
                                pred_vecs = self.model.com_predict(pred_imgs, input_is_pred_img=True)
                            else:
                                pred_vecs = self.get_predicted_vec(pred_imgs, input_is_pred_img=True)
                            #pred_vecs = self.model.predict(pred_imgs, input_is_pred_img=True)

                        repeat_x_images.append(x_img.detach().cpu().numpy())
                        repeat_y_images.append(y_img.detach().cpu().numpy())
                        repeat_pred_images.append(pred_imgs.detach().cpu().numpy())

                        #perturbing images here
                        epsilon = 0.5
                        self.set_rss_tensor()
                        # rand_select = np.random.random()
                        # if rand_select < 0.5:
                        #     return

                        # Now, we need to compute the gradients w.r.t. x_img
                        # Assuming you need to compute a loss to propagate backward
                        # Example: loss = some_function(pred_vecs, y_vecs)
                        if isinstance(self.loss_func, nn.MSELoss):
                            loss = self.loss_func(pred_imgs, y_img)
                        else:
                            loss = self.loss_func(pred_imgs, y_img, y_vecs)
                        loss.backward()  # Backpropagate to calculate gradients
                        # print(y_vecs.shape)
                        # print(y_vecs[0][0])
                        # print(pred_vecs.shape)
                        # print(pred_vecs[0])


                        grad = x_img.grad.data.clone()
                        pert_x = get_random_attack_vec(x_vecs, grad, self.rss_tensor[0].item(), self.rss_tensor[1].item(), epsilon)
                        pert_pred_img, pert_x_img, pert_y_img = self.model((pert_x, y_vecs))
                        if isinstance(self.loss_func, CoMLoss):
                            pert_pred_vecs = self.model.com_predict(pert_pred_img, input_is_pred_img=True)
                        else:
                            pert_pred_vecs = self.get_predicted_vec(pert_pred_img, input_is_pred_img=True)
                        repeat_pert_x_images.append(pert_x_img.detach().cpu().numpy())
                        # repeat_y_images.append(y_img.detach().cpu().numpy())
                        repeat_pert_pred_images.append(pert_pred_img.detach().cpu().numpy())
                        
                        # print(pred_imgs.shape)
                        # print(self.loss_func)
                        # print(pert_pred_vecs.shape)
                        # print(pert_pred_vecs[0])


                        for idx in range(x_vecs.size(0)):
                            y_img_sample = y_img[idx].unsqueeze(0)
                            pred_img_sample = pred_imgs[idx].unsqueeze(0)
                            y_vec_sample = y_vecs[idx].unsqueeze(0)

                            pert_y_img_sample = pert_y_img[idx].unsqueeze(0)
                            pert_pred_img_sample = pert_pred_img[idx].unsqueeze(0)

                            if isinstance(self.loss_func, nn.MSELoss):
                                original_loss = self.loss_func(pred_img_sample, y_img_sample)
                                pert_loss = self.loss_func(pert_pred_img_sample, pert_y_img_sample)
                            else:
                                original_loss = self.loss_func(pred_img_sample, y_img_sample, y_vec_sample)
                                pert_loss = self.loss_func(pert_pred_img_sample, pert_y_img_sample, y_vec_sample)

                            abs_perturbation = torch.abs(x_vecs[idx, :, 0] - pert_x[idx, :, 0]).sum()
                            batch_pert_grad.append(abs_perturbation.item())


                            print("Pred Image Idx shape = ", pred_imgs[idx].shape)
                            image = pred_imgs[idx, 0]
                            peaks, _ = self.detect_clusters_and_peaks(image)
                            if len(peaks) == 0:
                                peaks, _ = self.detect_clusters_and_peaks2(image)
                            # values, indices = torch.topk(image.detach().flatten(), k=2)
                            # Convert flat indices back to 2D coordinates
                            # coordinates = [divmod(idx.item(), image.size(1)) for idx in indices]
                            # print("Top 2 peak coordinates:", coordinates)
                            calc_pred_loss, true_num_tx = self.calculate_loss(y_vecs[idx].detach().cpu().numpy(), peaks)


                            print("True y coords = ", y_vecs[idx].detach().cpu().numpy())
                            print("Predicted y coords = ", pred_vecs[idx].detach().cpu().numpy())
                            # print(image)
                            print("New 1 Predicted y coords = ", peaks)
                            # print("New 2 Predicted y coords = ", peaks2)
                            print("Calculated Loss = ", calc_pred_loss)
                            print("Perturbed y coords = ", pert_pred_vecs[idx].detach().cpu().numpy())

                            pert_image = pert_pred_img[idx, 0]
                            pert_peaks, _ = self.detect_clusters_and_peaks(pert_image)
                            if len(pert_peaks) == 0:
                                pert_peaks, _ = self.detect_clusters_and_peaks2(pert_image)
                            # values, indices = torch.topk(image.detach().flatten(), k=2)
                            # Convert flat indices back to 2D coordinates
                            # coordinates = [divmod(idx.item(), image.size(1)) for idx in indices]
                            # print("Top 2 peak coordinates:", coordinates)

                            print("New 1 perturbed Predicted y coords = ", pert_peaks)
                            calc_pert_pred_loss, true_num_tx = self.calculate_loss(y_vecs[idx].detach().cpu().numpy(), pert_peaks)

                            batch_true_num_tx.append(true_num_tx)
                            batch_pred_losses.append(calc_pred_loss)
                            batch_pert_pred_losses.append(calc_pert_pred_loss)



                            

                            batch_y_coords.append(y_vecs[idx].detach().cpu().numpy())
                            batch_pred_y_coords.append(peaks)
                            batch_pert_pred_y_coords.append(pert_peaks)

                            writer.writerow([sample_id, calc_pred_loss, calc_pert_pred_loss, true_num_tx])
                            sample_id += 1  # Increment sample counter
                        
                        all_true_num_tx.append(batch_true_num_tx)
                        pred_losses.append(batch_pred_losses)
                        pert_pred_losses.append(batch_pert_pred_losses)




                    else:
                        if isinstance(self.loss_func, CoMLoss):
                            pred_vecs = self.model.com_predict(x_vecs)
                        else:
                            pred_vecs = self.model.predict(x_vecs)
                    repeat_pred_vecs.append(pred_vecs.detach().cpu().numpy())
                    repeat_error_vecs.append(torch.linalg.norm(pred_vecs[:,:2] - y_vecs[:,0,1:3], dim=1).detach().cpu().numpy())

                    repeat_pert_pred_vecs.append(pert_pred_vecs.detach().cpu().numpy())
                    repeat_pert_error_vecs.append(torch.linalg.norm(pert_pred_vecs[:,:2] - y_vecs[:,0,1:3], dim=1).detach().cpu().numpy())
                if save_images:
                    all_x_images.append(np.concatenate(repeat_x_images))
                    all_y_images.append(np.concatenate(repeat_y_images))
                    all_pred_images.append(np.concatenate(repeat_pred_images))

                    all_pert_x_images.append(np.concatenate(repeat_pert_x_images))
                    # all_y_images.append(np.concatenate(repeat_y_images))
                    all_pert_pred_images.append(np.concatenate(repeat_pert_pred_images))
                all_pred_vecs.append(np.concatenate(repeat_pred_vecs))
                all_error_vecs.append(np.concatenate(repeat_error_vecs))

                all_pert_pred_vecs.append(np.concatenate(repeat_pert_pred_vecs))
                all_pert_error_vecs.append(np.concatenate(repeat_pert_error_vecs))
            all_pred_vecs = np.array(all_pred_vecs)
            all_error_vecs = np.array(all_error_vecs) * self.params.meter_scale

            all_pert_pred_vecs = np.array(all_pert_pred_vecs)
            all_pert_error_vecs = np.array(all_pert_error_vecs) * self.params.meter_scale

            pred_losses.append(batch_pred_losses)
            pert_pred_losses.append(batch_pert_pred_losses)

            all_pert_grad.append(batch_pert_grad)
            all_y_coords.append(batch_y_coords)
            all_pred_y_coords.append(batch_pred_y_coords)
            all_pert_pred_y_coords.append(batch_pert_pred_y_coords)

            results = {'preds': all_pred_vecs, 'error': all_error_vecs, 'pert_preds': all_pert_pred_vecs, 'pert_error': all_pert_error_vecs}
            if save_images:
                results['x_imgs'] = np.array(all_x_images)
                results['y_imgs'] = np.array(all_y_images)
                results['pred_imgs'] = np.array(all_pred_images)

                results['pert_x_imgs'] = np.array(all_pert_x_images)
                # results['y_imgs'] = np.array(all_y_images)
                results['pert_pred_imgs'] = np.array(all_pert_pred_images)


            os.makedirs(save_dir, exist_ok=True)

            base_dir = os.path.dirname(save_dir)
            save_dir_one_tx = os.path.join(base_dir, "one_tx")
            os.makedirs(save_dir_one_tx, exist_ok=True)
            save_dir_two_tx = os.path.join(base_dir, "two_tx")
            os.makedirs(save_dir_two_tx, exist_ok=True)

            if 'x_imgs' in results and 'y_imgs' in results and 'pred_imgs' in results:
                x_imgs = results['x_imgs']
                y_imgs = results['y_imgs']
                pred_imgs = results['pred_imgs']

                pert_x_imgs = results['pert_x_imgs']
                # y_imgs = results['y_imgs']
                pert_pred_imgs = results['pert_pred_imgs']

                # print("X_images shape = ", x_imgs.shape)
                # print("Y_images shape = ",y_imgs.shape)
                # print("Pred_images shape = ",pred_imgs.shape)
                # X_images shape =  (1, 320, 3, 89, 97)
                # Y_images shape =  (1, 320, 1, 89, 97)
                # Pred_images shape =  (1, 320, 1, 89, 97)
                s_id = 0
                total_sample = 0
                total_one = 0
                true_one = 0
                total_two = 0
                true_two = 0
                for i in range(x_imgs.shape[0]):
                    for j in range(x_imgs.shape[1]):
                        x_img_filename = os.path.join(save_dir, f"x_image_{i}_{j}.png")
                        y_img_filename = os.path.join(save_dir, f"y_image_{i}_{j}.png")
                        pred_img_filename = os.path.join(save_dir, f"pred_image_{i}_{j}.png")

                        pert_x_img_filename = os.path.join(save_dir, f"pert_x_image_{i}_{j}.png")
                        # y_img_filename = os.path.join(save_dir, f"y_image_{i}_{j}.png")
                        pert_pred_img_filename = os.path.join(save_dir, f"pert_pred_image_{i}_{j}.png")
                        
                        # print(x_imgs[i][j][1].shape)
                        # print(pert_x_imgs[i][j][1].shape)

                        # printable_x_img = set_negatives_to_zero(x_imgs[i][j][1])
                        # printable_pert_x_img = set_negatives_to_zero(pert_x_imgs[i][j][1])


                        # print(type(pert_x_imgs))
                        # Save each image
                        # save_image(x_imgs[i][j][1], x_img_filename) #only the second channel
                        # # save_image(printable_x_img, x_img_filename) #only the second channel
                        # save_image(y_imgs[i][j], y_img_filename)
                        # save_image(pred_imgs[i][j], pred_img_filename)

                        # # save_image(printable_pert_x_img, pert_x_img_filename) #only the second channel
                        # save_image(pert_x_imgs[i][j][1], pert_x_img_filename) #only the second channel
                        # # save_image(y_imgs[i][j], y_img_filename)
                        # save_image(pert_pred_imgs[i][j], pert_pred_img_filename)
                        
                        # print(f"Saved input image {i}_{j} to: {x_img_filename}")
                        # print(f"Saved ground truth image {i}_{j} to: {y_img_filename}")
                        # print(f"Saved predicted image {i}_{j} to: {pred_img_filename}")

                        # print(f"Saved perturbed input image {i}_{j} to: {pert_x_img_filename}")
                        # # print(f"Saved ground truth image {i}_{j} to: {y_img_filename}")
                        # print(f"Saved perturbed predicted image {i}_{j} to: {pert_pred_img_filename}")

                        combined_img_filename = os.path.join(save_dir, f"combined_image_{i}_{j}.png")
                        # print(type(pred_imgs[i][j]))
                        # print(pred_imgs[i][j].shape)
                        if self.params.one_tx:
                            predicted_com = self.generate_ground_truth_like_image(pred_imgs[i][j])
                            pert_pred_com = self.generate_ground_truth_like_image(pert_pred_imgs[i][j])
                        
                        else: 
                            # print(type(pred_imgs[i][j]))
                            # print(pred_imgs[i][j].shape)
                            # print(type(pert_pred_imgs[i][j]))
                            # print(pert_pred_imgs[i][j].shape)
                            predicted_com = self.generate_ground_truth_for_clusters(pred_img = pred_imgs[i][j])
                            pert_pred_com = self.generate_ground_truth_for_clusters(pred_img = pert_pred_imgs[i][j])                           

                        # print(self.params.one_tx)

                        self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                                                 x_imgs[i][j][1], pert_x_imgs[i][j][1], s_id,
                                                pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                                                all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                                                predicted_com, pert_pred_com, all_pert_grad[i][j])
                        

                        if all_true_num_tx[i][j] == 1:
                            total_sample = total_sample + 1
                            total_one = total_one + 1
                            if len(all_pred_y_coords[i][j]) == 1:
                                true_one = true_one + 1

                            # print("true = ", all_y_coords[i][j]," type = ", type(all_y_coords[i][j]), " predicted = ", all_pred_y_coords[i][j]," type = ", type(all_pred_y_coords[i][j]))
                            combined_img_filename = os.path.join(save_dir_one_tx, f"combined_image_{i}_{j}.png")
                            self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                                                    x_imgs[i][j][1], pert_x_imgs[i][j][1], s_id,
                                                    pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                                                    all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                                                    predicted_com, pert_pred_com, all_pert_grad[i][j])
                            
                        elif all_true_num_tx[i][j] == 2:
                            total_sample = total_sample + 1
                            total_two = total_two + 1
                            if len(all_pred_y_coords[i][j]) == 2:
                                true_two = true_two + 1

                            # print("true = ", all_y_coords[i][j]," type = ", type(all_y_coords[i][j]), " predicted = ", all_pred_y_coords[i][j]," type = ", type(all_pred_y_coords[i][j]))
                            combined_img_filename = os.path.join(save_dir_two_tx, f"combined_image_{i}_{j}.png")
                            self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                                                    x_imgs[i][j][1], pert_x_imgs[i][j][1], s_id,
                                                    pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                                                    all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                                                    predicted_com, pert_pred_com, all_pert_grad[i][j])

                        # if s_id == 55:
                        #     print(abcd)
                        
                        # self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                        #                          printable_x_img, printable_pert_x_img, s_id,
                        #                         pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                        #                         all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                        #                         predicted_com, pert_pred_com)
                        
                        s_id = s_id+1
            else:
                print("No images found in results to save.")

        loss_file = os.path.join(loss_folder, 'losses.csv')
        df = pd.read_csv(loss_file)

        df_1 = df[df['Num_Tx'] == 1]
        df_2 = df[df['Num_Tx'] == 2]

        loss_file_one_tx = os.path.join(loss_folder, 'losses_1.csv')
        loss_file_two_tx = os.path.join(loss_folder, 'losses_2.csv')

        df_1.to_csv(loss_file_one_tx, index=False)
        df_2.to_csv(loss_file_two_tx, index=False)


        one_tx_percentage = (true_one / total_one) * 100 if total_one != 0 else 0
        two_tx_percentage = (true_two / total_two) * 100 if total_two != 0 else 0

        final_report = os.path.join(loss_folder, 'final_report.csv')

        with open(final_report, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Writing header
            writer.writerow([
                "Total Sample", "Total One Tx", "Truly Classified One Tx", "One Tx Percentage",
                "Total Two Tx", "Truly Classified Two Tx", "Two Tx Percentage"
            ])
            
            # Writing data
            writer.writerow([
                total_sample, total_one, true_one, one_tx_percentage,
                total_two, true_two, two_tx_percentage
            ])

        print(f"File '{final_report}' has been created successfully.")


        return results
    

    def draw_img_then_perturb_fixed_attack(self, test_key=None, dataloader=None, num_power_repeats=1, save_images=False, apply_wc_attack=False, attack_id_loc=1, save_dir=None):
        """Evaluate model on the given dataloader dataset or test keys
        
        Args:
            dataloader      torch.DataLoader -- data to evaluate
            y_vecs          list<np.array> -- ground truth for locations
            num_power_repeats  int -- number of times to repeat testset, if assigning random power each iteration, to get avg performance
        return:
            total_loss      float -- loss from testset
            best_results    dict -- results from best setting of thresh and suppression_size   
            min_fn          float -- misdetection rate
            min_fp          float -- false alarm rate
        """
        # if sensors_to_perturb is not None:
        #     # Convert sensors_to_perturb array to a string, e.g., [1,2,3] -> "1_2_3"
        #     sensor_list = sensors_to_perturb.tolist()
        #     sensors_str = '_'.join(map(str, sensor_list))
        #     # Create the folder name by appending the sensors to the 'loss' name
        #     folder_name = f"loss_{sensors_str}"
        # else:
        #     # Default folder name if sensors_to_perturb is None
        #     folder_name = "loss"

        # Prepare the CSV file to store the losses
        if save_dir is not None:
            folder_name = 'loss'
            parent_dir = os.path.dirname(save_dir)
            loss_folder = os.path.join(parent_dir, folder_name)
            os.makedirs(loss_folder, exist_ok=True)
            loss_file = os.path.join(loss_folder, 'losses.csv')
        else:
            loss_folder = 'loss'
            os.makedirs(loss_folder, exist_ok=True)
            loss_file = os.path.join(loss_folder, 'losses.csv')




        # # Prepare the CSV file to store the losses
        # if save_dir is not None:
        #     loss_file = os.path.join(save_dir, 'losses.csv')
        # else:
        #     loss_file = 'losses.csv'
        
        with open(loss_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sample ID', 'Original Loss', 'Perturbed Loss', 'Num_Tx'])


            self.model.eval()
            all_x_images = []
            all_pert_x_images = []
            all_y_images = []
            all_pred_images = []
            all_pert_pred_images = []
            all_pred_vecs = []
            all_pert_pred_vecs = []
            all_error_vecs = []
            all_pert_error_vecs = []

            all_true_num_tx = []
            pred_losses = []
            pert_pred_losses = []

            all_y_coords = []
            all_pred_y_coords = []
            all_pert_pred_y_coords = []

            all_pert_grad = []

            if dataloader is None or test_key is not None:
                dataloader = self.rss_loc_dataset.data[test_key].ordered_dataloader

            sample_id = 0  # To track which sample is being processed
            for _ in range(num_power_repeats):
                repeat_pred_vecs = []
                repeat_pert_pred_vecs = []
                repeat_error_vecs = []
                repeat_pert_error_vecs = []
                repeat_x_images = []
                repeat_pert_x_images = []
                repeat_y_images = []
                # repeat_perty_images = []
                repeat_pred_images = []
                repeat_pert_pred_images = []

                batch_true_num_tx = []
                batch_pred_losses = []
                batch_pert_pred_losses = []

                batch_y_coords = []
                batch_pred_y_coords = []
                batch_pert_pred_y_coords = []

                batch_pert_grad = []

                for t, sample in enumerate(dataloader):
                    # if t >= 1:  # Stop after processing the first 5 batches
                    #     break
                    x_vecs = sample[0].to(self.device)
                    y_vecs = sample[1].to(self.device)
                    

                    if save_images:
                        # Enable gradient computation for inputs
                        x_vecs.requires_grad_(True)
                        y_vecs.requires_grad_(True)


                        # Freeze model parameters by disabling gradient computation for them
                        for param in self.model.parameters():
                            param.requires_grad = False

                        # Enable gradient calculation for inputs
                        with torch.set_grad_enabled(True):
                            pred_imgs, x_img, y_img = self.model((x_vecs, y_vecs))
                            if isinstance(self.loss_func, CoMLoss):
                                pred_vecs = self.model.com_predict(pred_imgs, input_is_pred_img=True)
                            else:
                                pred_vecs = self.get_predicted_vec(pred_imgs, input_is_pred_img=True)
                            #pred_vecs = self.model.predict(pred_imgs, input_is_pred_img=True)

                        repeat_x_images.append(x_img.detach().cpu().numpy())
                        repeat_y_images.append(y_img.detach().cpu().numpy())
                        repeat_pred_images.append(pred_imgs.detach().cpu().numpy())

                        #perturbing images here
                        epsilon = 0.5
                        self.set_rss_tensor()
                        # rand_select = np.random.random()
                        # if rand_select < 0.5:
                        #     return

                        # Now, we need to compute the gradients w.r.t. x_img
                        # Assuming you need to compute a loss to propagate backward
                        # Example: loss = some_function(pred_vecs, y_vecs)
                        if isinstance(self.loss_func, nn.MSELoss):
                            loss = self.loss_func(pred_imgs, y_img)
                        else:
                            loss = self.loss_func(pred_imgs, y_img, y_vecs)
                        loss.backward()  # Backpropagate to calculate gradients
                        # print(y_vecs.shape)
                        # print(y_vecs[0][0])
                        # print(pred_vecs.shape)
                        # print(pred_vecs[0])


                        grad = x_img.grad.data.clone()
                        # pert_x = get_random_attack_vec(x_vecs, grad, self.rss_tensor[0].item(), self.rss_tensor[1].item(), epsilon)
                        pert_x = get_random_attack_vec_spec_attack(x_vecs = x_vecs, grad = grad, rss_low = self.rss_tensor[0].item(), rss_high = self.rss_tensor[1].item(), attack_id = attack_id_loc, epsilon = epsilon)
                        pert_pred_img, pert_x_img, pert_y_img = self.model((pert_x, y_vecs))
                        if isinstance(self.loss_func, CoMLoss):
                            pert_pred_vecs = self.model.com_predict(pert_pred_img, input_is_pred_img=True)
                        else:
                            pert_pred_vecs = self.get_predicted_vec(pert_pred_img, input_is_pred_img=True)
                        repeat_pert_x_images.append(pert_x_img.detach().cpu().numpy())
                        # repeat_y_images.append(y_img.detach().cpu().numpy())
                        repeat_pert_pred_images.append(pert_pred_img.detach().cpu().numpy())
                        
                        # print(pred_imgs.shape)
                        # print(self.loss_func)
                        # print(pert_pred_vecs.shape)
                        # print(pert_pred_vecs[0])


                        for idx in range(x_vecs.size(0)):
                            y_img_sample = y_img[idx].unsqueeze(0)
                            pred_img_sample = pred_imgs[idx].unsqueeze(0)
                            y_vec_sample = y_vecs[idx].unsqueeze(0)

                            pert_y_img_sample = pert_y_img[idx].unsqueeze(0)
                            pert_pred_img_sample = pert_pred_img[idx].unsqueeze(0)

                            if isinstance(self.loss_func, nn.MSELoss):
                                original_loss = self.loss_func(pred_img_sample, y_img_sample)
                                pert_loss = self.loss_func(pert_pred_img_sample, pert_y_img_sample)
                            else:
                                original_loss = self.loss_func(pred_img_sample, y_img_sample, y_vec_sample)
                                pert_loss = self.loss_func(pert_pred_img_sample, pert_y_img_sample, y_vec_sample)

                            abs_perturbation = torch.abs(x_vecs[idx, :, 0] - pert_x[idx, :, 0]).sum()
                            batch_pert_grad.append(abs_perturbation.item())


                            print("Pred Image Idx shape = ", pred_imgs[idx].shape)
                            image = pred_imgs[idx, 0]
                            peaks, _ = self.detect_clusters_and_peaks(image)
                            if len(peaks) == 0:
                                peaks, _ = self.detect_clusters_and_peaks2(image)
                            # values, indices = torch.topk(image.detach().flatten(), k=2)
                            # Convert flat indices back to 2D coordinates
                            # coordinates = [divmod(idx.item(), image.size(1)) for idx in indices]
                            # print("Top 2 peak coordinates:", coordinates)
                            calc_pred_loss, true_num_tx = self.calculate_loss(y_vecs[idx].detach().cpu().numpy(), peaks)


                            print("True y coords = ", y_vecs[idx].detach().cpu().numpy())
                            print("Predicted y coords = ", pred_vecs[idx].detach().cpu().numpy())
                            # print(image)
                            print("New 1 Predicted y coords = ", peaks)
                            # print("New 2 Predicted y coords = ", peaks2)
                            print("Calculated Loss = ", calc_pred_loss)
                            print("Perturbed y coords = ", pert_pred_vecs[idx].detach().cpu().numpy())

                            pert_image = pert_pred_img[idx, 0]
                            pert_peaks, _ = self.detect_clusters_and_peaks(pert_image)
                            if len(pert_peaks) == 0:
                                pert_peaks, _ = self.detect_clusters_and_peaks2(pert_image)
                            # values, indices = torch.topk(image.detach().flatten(), k=2)
                            # Convert flat indices back to 2D coordinates
                            # coordinates = [divmod(idx.item(), image.size(1)) for idx in indices]
                            # print("Top 2 peak coordinates:", coordinates)

                            print("New 1 perturbed Predicted y coords = ", pert_peaks)
                            calc_pert_pred_loss, true_num_tx = self.calculate_loss(y_vecs[idx].detach().cpu().numpy(), pert_peaks)

                            batch_true_num_tx.append(true_num_tx)
                            batch_pred_losses.append(calc_pred_loss)
                            batch_pert_pred_losses.append(calc_pert_pred_loss)



                            

                            batch_y_coords.append(y_vecs[idx].detach().cpu().numpy())
                            batch_pred_y_coords.append(peaks)
                            batch_pert_pred_y_coords.append(pert_peaks)

                            writer.writerow([sample_id, calc_pred_loss, calc_pert_pred_loss, true_num_tx])
                            sample_id += 1  # Increment sample counter
                        
                        all_true_num_tx.append(batch_true_num_tx)
                        pred_losses.append(batch_pred_losses)
                        pert_pred_losses.append(batch_pert_pred_losses)




                    else:
                        if isinstance(self.loss_func, CoMLoss):
                            pred_vecs = self.model.com_predict(x_vecs)
                        else:
                            pred_vecs = self.model.predict(x_vecs)
                    repeat_pred_vecs.append(pred_vecs.detach().cpu().numpy())
                    repeat_error_vecs.append(torch.linalg.norm(pred_vecs[:,:2] - y_vecs[:,0,1:3], dim=1).detach().cpu().numpy())

                    repeat_pert_pred_vecs.append(pert_pred_vecs.detach().cpu().numpy())
                    repeat_pert_error_vecs.append(torch.linalg.norm(pert_pred_vecs[:,:2] - y_vecs[:,0,1:3], dim=1).detach().cpu().numpy())
                if save_images:
                    all_x_images.append(np.concatenate(repeat_x_images))
                    all_y_images.append(np.concatenate(repeat_y_images))
                    all_pred_images.append(np.concatenate(repeat_pred_images))

                    all_pert_x_images.append(np.concatenate(repeat_pert_x_images))
                    # all_y_images.append(np.concatenate(repeat_y_images))
                    all_pert_pred_images.append(np.concatenate(repeat_pert_pred_images))
                all_pred_vecs.append(np.concatenate(repeat_pred_vecs))
                all_error_vecs.append(np.concatenate(repeat_error_vecs))

                all_pert_pred_vecs.append(np.concatenate(repeat_pert_pred_vecs))
                all_pert_error_vecs.append(np.concatenate(repeat_pert_error_vecs))
            all_pred_vecs = np.array(all_pred_vecs)
            all_error_vecs = np.array(all_error_vecs) * self.params.meter_scale

            all_pert_pred_vecs = np.array(all_pert_pred_vecs)
            all_pert_error_vecs = np.array(all_pert_error_vecs) * self.params.meter_scale

            pred_losses.append(batch_pred_losses)
            pert_pred_losses.append(batch_pert_pred_losses)

            all_pert_grad.append(batch_pert_grad)
            all_y_coords.append(batch_y_coords)
            all_pred_y_coords.append(batch_pred_y_coords)
            all_pert_pred_y_coords.append(batch_pert_pred_y_coords)

            results = {'preds': all_pred_vecs, 'error': all_error_vecs, 'pert_preds': all_pert_pred_vecs, 'pert_error': all_pert_error_vecs}
            if save_images:
                results['x_imgs'] = np.array(all_x_images)
                results['y_imgs'] = np.array(all_y_images)
                results['pred_imgs'] = np.array(all_pred_images)

                results['pert_x_imgs'] = np.array(all_pert_x_images)
                # results['y_imgs'] = np.array(all_y_images)
                results['pert_pred_imgs'] = np.array(all_pert_pred_images)


            os.makedirs(save_dir, exist_ok=True)

            base_dir = os.path.dirname(save_dir)
            save_dir_one_tx = os.path.join(base_dir, "one_tx")
            os.makedirs(save_dir_one_tx, exist_ok=True)
            save_dir_two_tx = os.path.join(base_dir, "two_tx")
            os.makedirs(save_dir_two_tx, exist_ok=True)

            if 'x_imgs' in results and 'y_imgs' in results and 'pred_imgs' in results:
                x_imgs = results['x_imgs']
                y_imgs = results['y_imgs']
                pred_imgs = results['pred_imgs']

                pert_x_imgs = results['pert_x_imgs']
                # y_imgs = results['y_imgs']
                pert_pred_imgs = results['pert_pred_imgs']

                # print("X_images shape = ", x_imgs.shape)
                # print("Y_images shape = ",y_imgs.shape)
                # print("Pred_images shape = ",pred_imgs.shape)
                # X_images shape =  (1, 320, 3, 89, 97)
                # Y_images shape =  (1, 320, 1, 89, 97)
                # Pred_images shape =  (1, 320, 1, 89, 97)
                s_id = 0
                for i in range(x_imgs.shape[0]):
                    for j in range(x_imgs.shape[1]):
                        x_img_filename = os.path.join(save_dir, f"x_image_{i}_{j}.png")
                        y_img_filename = os.path.join(save_dir, f"y_image_{i}_{j}.png")
                        pred_img_filename = os.path.join(save_dir, f"pred_image_{i}_{j}.png")

                        pert_x_img_filename = os.path.join(save_dir, f"pert_x_image_{i}_{j}.png")
                        # y_img_filename = os.path.join(save_dir, f"y_image_{i}_{j}.png")
                        pert_pred_img_filename = os.path.join(save_dir, f"pert_pred_image_{i}_{j}.png")
                        
                        # print(x_imgs[i][j][1].shape)
                        # print(pert_x_imgs[i][j][1].shape)

                        # printable_x_img = set_negatives_to_zero(x_imgs[i][j][1])
                        # printable_pert_x_img = set_negatives_to_zero(pert_x_imgs[i][j][1])


                        # print(type(pert_x_imgs))
                        # Save each image
                        # save_image(x_imgs[i][j][1], x_img_filename) #only the second channel
                        # # save_image(printable_x_img, x_img_filename) #only the second channel
                        # save_image(y_imgs[i][j], y_img_filename)
                        # save_image(pred_imgs[i][j], pred_img_filename)

                        # # save_image(printable_pert_x_img, pert_x_img_filename) #only the second channel
                        # save_image(pert_x_imgs[i][j][1], pert_x_img_filename) #only the second channel
                        # # save_image(y_imgs[i][j], y_img_filename)
                        # save_image(pert_pred_imgs[i][j], pert_pred_img_filename)
                        
                        # print(f"Saved input image {i}_{j} to: {x_img_filename}")
                        # print(f"Saved ground truth image {i}_{j} to: {y_img_filename}")
                        # print(f"Saved predicted image {i}_{j} to: {pred_img_filename}")

                        # print(f"Saved perturbed input image {i}_{j} to: {pert_x_img_filename}")
                        # # print(f"Saved ground truth image {i}_{j} to: {y_img_filename}")
                        # print(f"Saved perturbed predicted image {i}_{j} to: {pert_pred_img_filename}")

                        combined_img_filename = os.path.join(save_dir, f"combined_image_{i}_{j}.png")
                        # print(type(pred_imgs[i][j]))
                        # print(pred_imgs[i][j].shape)
                        if self.params.one_tx:
                            predicted_com = self.generate_ground_truth_like_image(pred_imgs[i][j])
                            pert_pred_com = self.generate_ground_truth_like_image(pert_pred_imgs[i][j])
                        
                        else: 
                            # print(type(pred_imgs[i][j]))
                            # print(pred_imgs[i][j].shape)
                            # print(type(pert_pred_imgs[i][j]))
                            # print(pert_pred_imgs[i][j].shape)
                            predicted_com = self.generate_ground_truth_for_clusters(pred_img = pred_imgs[i][j])
                            pert_pred_com = self.generate_ground_truth_for_clusters(pred_img = pert_pred_imgs[i][j])                           

                        # print(self.params.one_tx)

                        self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                                                 x_imgs[i][j][1], pert_x_imgs[i][j][1], s_id,
                                                pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                                                all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                                                predicted_com, pert_pred_com, all_pert_grad[i][j])
                        

                        if all_true_num_tx[i][j] == 1:
                            combined_img_filename = os.path.join(save_dir_one_tx, f"combined_image_{i}_{j}.png")
                            self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                                                    x_imgs[i][j][1], pert_x_imgs[i][j][1], s_id,
                                                    pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                                                    all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                                                    predicted_com, pert_pred_com, all_pert_grad[i][j])
                            
                        elif all_true_num_tx[i][j] == 2:
                            combined_img_filename = os.path.join(save_dir_two_tx, f"combined_image_{i}_{j}.png")
                            self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                                                    x_imgs[i][j][1], pert_x_imgs[i][j][1], s_id,
                                                    pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                                                    all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                                                    predicted_com, pert_pred_com, all_pert_grad[i][j])

                        
                        # self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                        #                          printable_x_img, printable_pert_x_img, s_id,
                        #                         pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                        #                         all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                        #                         predicted_com, pert_pred_com)
                        
                        s_id = s_id+1
            else:
                print("No images found in results to save.")

        return results
    

    def draw_img_then_perturb_spec_sensor(self, sensors_to_perturb, test_key=None, dataloader=None, num_power_repeats=1, save_images=False, apply_wc_attack=False, save_dir=None):
        """Evaluate model on the given dataloader dataset or test keys
        
        Args:
            dataloader      torch.DataLoader -- data to evaluate
            y_vecs          list<np.array> -- ground truth for locations
            num_power_repeats  int -- number of times to repeat testset, if assigning random power each iteration, to get avg performance
        return:
            total_loss      float -- loss from testset
            best_results    dict -- results from best setting of thresh and suppression_size   
            min_fn          float -- misdetection rate
            min_fp          float -- false alarm rate
        """
        if sensors_to_perturb is not None:
            # Convert sensors_to_perturb array to a string, e.g., [1,2,3] -> "1_2_3"
            sensor_list = sensors_to_perturb.tolist()
            sensors_str = '_'.join(map(str, sensor_list))
            # Create the folder name by appending the sensors to the 'loss' name
            folder_name = f"loss_{sensors_str}"
        else:
            # Default folder name if sensors_to_perturb is None
            folder_name = "loss"

        # Prepare the CSV file to store the losses
        if save_dir is not None:
            parent_dir = os.path.dirname(save_dir)
            loss_folder = os.path.join(parent_dir, folder_name)
            os.makedirs(loss_folder, exist_ok=True)
            loss_file = os.path.join(loss_folder, 'losses.csv')
        else:
            loss_folder = 'loss'
            os.makedirs(loss_folder, exist_ok=True)
            loss_file = os.path.join(loss_folder, 'losses.csv')

        
        with open(loss_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sample ID', 'Original Loss', 'Perturbed Loss'])


            self.model.eval()
            all_x_images = []
            all_pert_x_images = []
            all_y_images = []
            all_pred_images = []
            all_pert_pred_images = []
            all_pred_vecs = []
            all_pert_pred_vecs = []
            all_error_vecs = []
            all_pert_error_vecs = []

            pred_losses = []
            pert_pred_losses = []

            all_y_coords = []
            all_pred_y_coords = []
            all_pert_pred_y_coords = []

            all_pert_grad = []

            if dataloader is None or test_key is not None:
                dataloader = self.rss_loc_dataset.data[test_key].ordered_dataloader

            sample_id = 0  # To track which sample is being processed
            for _ in range(num_power_repeats):
                repeat_pred_vecs = []
                repeat_pert_pred_vecs = []
                repeat_error_vecs = []
                repeat_pert_error_vecs = []
                repeat_x_images = []
                repeat_pert_x_images = []
                repeat_y_images = []
                # repeat_perty_images = []
                repeat_pred_images = []
                repeat_pert_pred_images = []

                batch_pred_losses = []
                batch_pert_pred_losses = []

                batch_y_coords = []
                batch_pred_y_coords = []
                batch_pert_pred_y_coords = []

                batch_pert_grad = []

                for t, sample in enumerate(dataloader):
                    if t >= 5:  # Stop after processing the first 5 batches
                        break
                    x_vecs = sample[0].to(self.device)
                    y_vecs = sample[1].to(self.device)
                    

                    if save_images:
                        # Enable gradient computation for inputs
                        x_vecs.requires_grad_(True)
                        y_vecs.requires_grad_(True)


                        # Freeze model parameters by disabling gradient computation for them
                        for param in self.model.parameters():
                            param.requires_grad = False

                        # Enable gradient calculation for inputs
                        with torch.set_grad_enabled(True):
                            pred_imgs, x_img, y_img = self.model((x_vecs, y_vecs))
                            if isinstance(self.loss_func, CoMLoss):
                                pred_vecs = self.model.com_predict(pred_imgs, input_is_pred_img=True)
                            else:
                                pred_vecs = self.model.predict(pred_imgs, input_is_pred_img=True)
                            #pred_vecs = self.model.predict(pred_imgs, input_is_pred_img=True)

                        repeat_x_images.append(x_img.detach().cpu().numpy())
                        repeat_y_images.append(y_img.detach().cpu().numpy())
                        repeat_pred_images.append(pred_imgs.detach().cpu().numpy())

                        #perturbing images here
                        epsilon = 0.5
                        self.set_rss_tensor()
                        # rand_select = np.random.random()
                        # if rand_select < 0.5:
                        #     return

                        # Now, we need to compute the gradients w.r.t. x_img
                        # Assuming you need to compute a loss to propagate backward
                        # Example: loss = some_function(pred_vecs, y_vecs)
                        if isinstance(self.loss_func, nn.MSELoss):
                            loss = self.loss_func(pred_imgs, y_img)
                        else:
                            loss = self.loss_func(pred_imgs, y_img, y_vecs)
                        loss.backward()  # Backpropagate to calculate gradients
                        # print(y_vecs.shape)
                        # print(y_vecs[0][0])
                        # print(pred_vecs.shape)
                        # print(pred_vecs[0])


                        grad = x_img.grad.data.clone()
                        pert_x = get_random_attack_vec_spec_sensor(sensors_to_perturb,x_vecs, grad, self.rss_tensor[0].item(), self.rss_tensor[1].item(), epsilon)
                        pert_pred_img, pert_x_img, pert_y_img = self.model((pert_x, y_vecs))
                        if isinstance(self.loss_func, CoMLoss):
                            pert_pred_vecs = self.model.com_predict(pert_pred_img, input_is_pred_img=True)
                        else:
                            pert_pred_vecs = self.model.predict(pert_pred_img, input_is_pred_img=True)
                        repeat_pert_x_images.append(pert_x_img.detach().cpu().numpy())
                        # repeat_y_images.append(y_img.detach().cpu().numpy())
                        repeat_pert_pred_images.append(pert_pred_img.detach().cpu().numpy())

                        print(pert_pred_vecs.shape)
                        print(pert_pred_vecs[0])


                        for idx in range(x_vecs.size(0)):
                            y_img_sample = y_img[idx].unsqueeze(0)
                            pred_img_sample = pred_imgs[idx].unsqueeze(0)
                            y_vec_sample = y_vecs[idx].unsqueeze(0)

                            pert_y_img_sample = pert_y_img[idx].unsqueeze(0)
                            pert_pred_img_sample = pert_pred_img[idx].unsqueeze(0)

                            if isinstance(self.loss_func, nn.MSELoss):
                                original_loss = self.loss_func(pred_img_sample, y_img_sample)
                                pert_loss = self.loss_func(pert_pred_img_sample, pert_y_img_sample)
                            else:
                                original_loss = self.loss_func(pred_img_sample, y_img_sample, y_vec_sample)
                                pert_loss = self.loss_func(pert_pred_img_sample, pert_y_img_sample, y_vec_sample)

                            batch_pred_losses.append(original_loss.item())
                            batch_pert_pred_losses.append(pert_loss.item())

                            abs_perturbation = torch.abs(x_vecs[idx, :, 0] - pert_x[idx, :, 0]).sum()
                            batch_pert_grad.append(abs_perturbation.item())

                            batch_y_coords.append(y_vecs[idx, 0, 1:].detach().cpu().numpy())
                            batch_pred_y_coords.append(pred_vecs[idx, :2].detach().cpu().numpy())
                            batch_pert_pred_y_coords.append(pert_pred_vecs[idx, :2].detach().cpu().numpy())

                            writer.writerow([sample_id, original_loss.item(), pert_loss.item()])
                            sample_id += 1  # Increment sample counter
                        
                        pred_losses.append(batch_pred_losses)
                        pert_pred_losses.append(batch_pert_pred_losses)




                    else:
                        if isinstance(self.loss_func, CoMLoss):
                            pred_vecs = self.model.com_predict(x_vecs)
                        else:
                            pred_vecs = self.model.predict(x_vecs)
                    repeat_pred_vecs.append(pred_vecs.detach().cpu().numpy())
                    repeat_error_vecs.append(torch.linalg.norm(pred_vecs[:,:2] - y_vecs[:,0,1:3], dim=1).detach().cpu().numpy())

                    repeat_pert_pred_vecs.append(pert_pred_vecs.detach().cpu().numpy())
                    repeat_pert_error_vecs.append(torch.linalg.norm(pert_pred_vecs[:,:2] - y_vecs[:,0,1:3], dim=1).detach().cpu().numpy())
                if save_images:
                    all_x_images.append(np.concatenate(repeat_x_images))
                    all_y_images.append(np.concatenate(repeat_y_images))
                    all_pred_images.append(np.concatenate(repeat_pred_images))

                    all_pert_x_images.append(np.concatenate(repeat_pert_x_images))
                    # all_y_images.append(np.concatenate(repeat_y_images))
                    all_pert_pred_images.append(np.concatenate(repeat_pert_pred_images))
                all_pred_vecs.append(np.concatenate(repeat_pred_vecs))
                all_error_vecs.append(np.concatenate(repeat_error_vecs))

                all_pert_pred_vecs.append(np.concatenate(repeat_pert_pred_vecs))
                all_pert_error_vecs.append(np.concatenate(repeat_pert_error_vecs))
            all_pred_vecs = np.array(all_pred_vecs)
            all_error_vecs = np.array(all_error_vecs) * self.params.meter_scale

            all_pert_pred_vecs = np.array(all_pert_pred_vecs)
            all_pert_error_vecs = np.array(all_pert_error_vecs) * self.params.meter_scale

            pred_losses.append(batch_pred_losses)
            pert_pred_losses.append(batch_pert_pred_losses)

            all_pert_grad.append(batch_pert_grad)
            all_y_coords.append(batch_y_coords)
            all_pred_y_coords.append(batch_pred_y_coords)
            all_pert_pred_y_coords.append(batch_pert_pred_y_coords)

            results = {'preds': all_pred_vecs, 'error': all_error_vecs, 'pert_preds': all_pert_pred_vecs, 'pert_error': all_pert_error_vecs}
            if save_images:
                results['x_imgs'] = np.array(all_x_images)
                results['y_imgs'] = np.array(all_y_images)
                results['pred_imgs'] = np.array(all_pred_images)

                results['pert_x_imgs'] = np.array(all_pert_x_images)
                # results['y_imgs'] = np.array(all_y_images)
                results['pert_pred_imgs'] = np.array(all_pert_pred_images)

            sensor_list = sensors_to_perturb.tolist()
            sensor_str = '_'.join(map(str, sensor_list)) 
            # save_dir = os.path.join(save_dir, f"_{sensor_str}")
            save_dir = save_dir  + "_" + sensor_str

            os.makedirs(save_dir, exist_ok=True)

            if 'x_imgs' in results and 'y_imgs' in results and 'pred_imgs' in results:
                x_imgs = results['x_imgs']
                y_imgs = results['y_imgs']
                pred_imgs = results['pred_imgs']

                pert_x_imgs = results['pert_x_imgs']
                # y_imgs = results['y_imgs']
                pert_pred_imgs = results['pert_pred_imgs']

                # print("X_images shape = ", x_imgs.shape)
                # print("Y_images shape = ",y_imgs.shape)
                # print("Pred_images shape = ",pred_imgs.shape)
                # X_images shape =  (1, 320, 3, 89, 97)
                # Y_images shape =  (1, 320, 1, 89, 97)
                # Pred_images shape =  (1, 320, 1, 89, 97)
                s_id = 0
                for i in range(x_imgs.shape[0]):
                    for j in range(x_imgs.shape[1]):
                        x_img_filename = os.path.join(save_dir, f"x_image_{i}_{j}.png")
                        y_img_filename = os.path.join(save_dir, f"y_image_{i}_{j}.png")
                        pred_img_filename = os.path.join(save_dir, f"pred_image_{i}_{j}.png")

                        pert_x_img_filename = os.path.join(save_dir, f"pert_x_image_{i}_{j}.png")
                        # y_img_filename = os.path.join(save_dir, f"y_image_{i}_{j}.png")
                        pert_pred_img_filename = os.path.join(save_dir, f"pert_pred_image_{i}_{j}.png")

                        # printable_x_img = set_negatives_to_zero(x_imgs[i][j][1])
                        # printable_pert_x_img = set_negatives_to_zero(pert_x_imgs[i][j][1])
                        
                        # Save each image
                        save_image(x_imgs[i][j][1], x_img_filename) #only the second channel
                        # save_image(printable_x_img, x_img_filename) #only the second channel
                        save_image(y_imgs[i][j], y_img_filename)
                        save_image(pred_imgs[i][j], pred_img_filename)
                        
                        
                        save_image(pert_x_imgs[i][j][1], pert_x_img_filename) #only the second channel
                        # save_image(printable_pert_x_img, pert_x_img_filename) #only the second channel
                        # save_image(y_imgs[i][j], y_img_filename)
                        save_image(pert_pred_imgs[i][j], pert_pred_img_filename)
                        
                        print(f"Saved input image {i}_{j} to: {x_img_filename}")
                        print(f"Saved ground truth image {i}_{j} to: {y_img_filename}")
                        print(f"Saved predicted image {i}_{j} to: {pred_img_filename}")

                        print(f"Saved perturbed input image {i}_{j} to: {pert_x_img_filename}")
                        # print(f"Saved ground truth image {i}_{j} to: {y_img_filename}")
                        print(f"Saved perturbed predicted image {i}_{j} to: {pert_pred_img_filename}")

                        combined_img_filename = os.path.join(save_dir, f"combined_image_{i}_{j}.png")

                        if self.params.one_tx:
                            predicted_com = self.generate_ground_truth_like_image(pred_imgs[i][j])
                            pert_pred_com = self.generate_ground_truth_like_image(pert_pred_imgs[i][j])
                        
                        else: 
                            predicted_com = self.generate_ground_truth_for_clusters(pred_imgs[i][j])
                            pert_pred_com = self.generate_ground_truth_for_clusters(pert_pred_imgs[i][j])                           

                        # print(self.params.one_tx)

                        self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                                                 x_imgs[i][j][1], pert_x_imgs[i][j][1], s_id,
                                                pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                                                all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                                                predicted_com, pert_pred_com, all_pert_grad[i][j])
                        
                        # self.save_combined_image(y_imgs[i][j], pred_imgs[i][j], pert_pred_imgs[i][j],
                        #                          printable_x_img, printable_pert_x_img, s_id,
                        #                         pred_losses[i][j], pert_pred_losses[i][j], combined_img_filename,
                        #                         all_y_coords[i][j], all_pred_y_coords[i][j], all_pert_pred_y_coords[i][j],
                        #                         predicted_com, pert_pred_com)
                        
                        s_id = s_id+1
            else:
                print("No images found in results to save.")

        return results