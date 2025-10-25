import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from locconfig import LocConfig
import ot
from b_spline_silu import NonLinearBSplineSiLU


class SlicedEarthMoversDistance(nn.Module):
    def __init__(self, num_projections=100, reduction='mean', scaling=1.0, p=1, normalize=True, device='cuda') -> None:
        super().__init__()
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'none':
            self.reduction = torch.nn.Identity()
        elif reduction == 'sum':
            self.reduction = torch.sum
        self.num_proj = num_projections
        self.eps = 1e-6
        self.scaling = scaling
        self.p = p
        self.normalize = normalize

    def forward(self, X, Y, *args):
        batch_tuple = X.shape[:-2]
        flat_X = X.reshape(batch_tuple + (-1,))

        # If max is 0, add epsilon
        max_vals, max_inds = flat_X.max(dim=-1)
        should_max = max_vals[:,0] < self.eps
        flat_X[should_max,0,max_inds[should_max,0]] = self.eps
        X = torch.mean(X, dim=1, keepdim=True)

        x = X[0,0]
        y = Y[0,0]
        x_coords = torch.nonzero(x > 0).float() / self.scaling
        y_coords = torch.nonzero(y > 0).float() / self.scaling
        dists = []
        if self.normalize:
            loss, projections = ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0]/x.sum(), y[y>0]/y.sum(),p=self.p, n_projections=self.num_proj, log=True)
        else:
            loss, projections = ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0], y[y>0],p=self.p, n_projections=self.num_proj, log=True)
        projections = projections['projections']
        for x, y in zip(X[1:],Y[1:]):
            x = x[0]
            y = y[0]
            x_coords = torch.nonzero(x > 0).float() / self.scaling
            y_coords = torch.nonzero(y > 0).float() / self.scaling
            if self.normalize:
                loss += ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0]/x.sum(), y[y>0]/y.sum(),p=self.p, projections=projections)
            else:
                loss += ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0], y[y>0],p=self.p, projections=projections)
        return loss


class CoMLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, pred_img, truth_img, tx_truth_coords):
        mean_pred = pred_img.mean(axis=1)
        centers_of_mass = get_centers_of_mass(mean_pred)
        error = torch.linalg.norm(tx_truth_coords[:,0,1:] - centers_of_mass, axis=1)
        return error.mean()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, dropout=0):
    #all my params values: self.n_channels=2, channel_scale=32, kernel_size=3, dropout=0
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            #mid_channel is also of 32 channels
        self.padding = kernel_size // 2
        #required padding
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=self.padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    #our passed parameters: channel_scale*mult[i], channel_scale*mult[i+1], kernel_size=kernel_size, dropout=dropout

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #doing 2x2 maxpool and then adding another doubleconv lineup
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    #all parameters: channel_scale*mult[depth-i] // factor, channel_scale*mult[depth-1-i] // factor, bilinear=bilinear, res_channels=self.use_residual*channel_scale*mult[depth-i] // factor, kernel_size=kernel_size, dropout=dropout

    def __init__(self, in_channels, out_channels, mid_channels=None, res_channels=None, bilinear=True, kernel_size=3, dropout=0):
        super().__init__()

        if res_channels is None:
            res_channels = out_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #scale factor mean double the height and width
            self.conv = DoubleConv(in_channels+res_channels, out_channels, mid_channels=mid_channels, kernel_size=kernel_size, dropout=dropout)
            #similar to any other double convolution layer
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            #if not bilinear, then you already have double input channels
            #no residuals and using direct from the previous layer
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=mid_channels, kernel_size=kernel_size, dropout=dropout)
        self.use_res = res_channels != 0
        #if using residuals or not

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.use_res:
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
        else:
            return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class Vec2Im(nn.Module):
    #passed parameters  config, img_shape, device, elevation_map=elevation_map
    def __init__(self, config: LocConfig,
                img_size: np.ndarray, 
                device,
                force_random_power_on_eval=False,
                elevation_map: torch.FloatTensor=None,
                max_num_rx=34,
                num_rx_categories=4,
                force_dropout_on_eval=False,
                spline_degree = 8, 
                num_spline_control_points = 15,
                sensor_value_range = [-2, 2]
    ):
        super(Vec2Im, self).__init__()
        self.img_size = torch.tensor(img_size).to(device)
        #take a tensor of image size
        self.device = device
        self.n_channels = 3 if config.include_elevation_map else 2
        #n_channels = 3 if there is elevation map otherwise 2 only
        self.config = config
        #config file copy

        self.force_random_power_on_eval = force_random_power_on_eval
        #false in our case
        self.elevation_map = elevation_map
        #taking elevation map
        self.force_dropout_on_eval = force_dropout_on_eval
        #false in our case

        self.rand_generator = torch.Generator(device=self.device)
        self.rand_generator.manual_seed(config.random_state)
        # self.device_weights = nn.Parameter(torch.ones(max_num_rx))
        # self.device_bias = nn.Parameter(torch.zeros(max_num_rx))
        # self.category_weights = nn.Parameter(torch.ones(num_rx_categories+1))
        # self.category_bias = nn.Parameter(torch.zeros(num_rx_categories+1))
        # Device and category-specific NonLinearBSplineSiLU objects
        self.device_splines = nn.ModuleList(
            [NonLinearBSplineSiLU(k = spline_degree, num = num_spline_control_points, grid_range= sensor_value_range) for _ in range(max_num_rx)]
        )
        self.category_splines = nn.ModuleList(
            [NonLinearBSplineSiLU(k = spline_degree, num = num_spline_control_points, grid_range= sensor_value_range) for _ in range(num_rx_categories + 1)]
        )

    
    def dropout(self, rx_vec):
        num_sensors = (rx_vec[:,:,0] != 0).sum(axis=1)
        #Counting the number of non-zero sensors for each sample in rx_vec
        for i, num in enumerate(num_sensors):
            nonzeros = rx_vec[i,:,0].nonzero()
            if num > self.config.min_dropout_inputs:
                num_indices = torch.randint(self.config.min_dropout_inputs, num, size=(1,)).long().item()
                to_remove = torch.randperm(num)[num_indices:]
                #Randomly selecting some sensors to be "dropped out" (set to zero) if the number of sensors exceeds a threshold (self.config.min_dropout_inputs).
                rx_vec[i,nonzeros[to_remove]] = 0
                #Returning the modified rx_vec where some sensor data has been zeroed out.
        return rx_vec

    
    def forward(self, x):
        rect_ids = None
        y_vecs = None
        if isinstance(x,tuple) and len(x) == 3:
            x_vecs, rect_ids, y_vecs = x
        elif isinstance(x,tuple) and len(x) == 2:
            x_vecs, y_vecs = x
            #taking x_val(rx data) and y_vecs(tx location) from the passed argument
        else:
            x_vecs = x

        if x_vecs is not None:
            print(" x_vecs shape = ", x_vecs.shape)
        else: 
            print(" x_vecs shape = None")

        if y_vecs is not None:
            print(" y_vecs shape = ", y_vecs.shape)
        else: 
            print(" y_vecs shape = None")

        if len(x_vecs.shape) < 3:
            #if just two coordinates then it has only one sample in the batch
            batch_size = 1
            x_vecs = x_vecs.unsqueeze(0)
        batch_size = x_vecs.shape[0]
        #otherwise more than one sample, more likely 32 samples at a batch

        # Make x_img
        x_img = torch.zeros((batch_size, self.n_channels, self.img_size[0], self.img_size[1]), device=self.device)
        #creating the image with dimension: (batch_size, num_channels, img_shape_x, img_shapey)
        # print("Image Size = ", self.img_size)
        # print("y_vecs = ", y_vecs)

        with torch.no_grad():
            if self.config.sensor_dropout and (self.training or self.force_dropout_on_eval):
                x_vecs = self.dropout(x_vecs)
                #it just dropout on training or force dropout on evaluation
        all_powers = x_vecs[:,:,0].clone()
        #it is (n_samples, n_rx_data) with each position having their corresponding rss value
        power_inds =  all_powers != 0
        if self.config.device_multiplication:
            all_device_ids = x_vecs[:, :, -2].long()  # Assuming -2 index is for device IDs
            flat_device_ids = all_device_ids.flatten()  # Flatten to 1D
            flat_powers = all_powers.flatten()  # Flatten to align with flat_device_ids

            # Prepare an empty tensor for transformed powers
            transformed_powers = torch.empty_like(flat_powers)
            unique_devices = torch.unique(flat_device_ids)
            for device_id in unique_devices:
                # Get indices for the current device_id
                device_indices = (flat_device_ids == device_id).nonzero(as_tuple=True)[0]
                
                # Apply the device-specific spline transformation
                device_spline = self.device_splines[device_id]
                transformed_powers[device_indices] = device_spline(
                    flat_powers[device_indices].unsqueeze(1)
                ).squeeze()  # Apply transformation and remove extra dimension
            # Reshape transformed powers back to the original shape of all_powers
            all_powers = transformed_powers.view(all_powers.shape)

        #power_inds has the same shape as all_powers (or x_vecs[:,:,0]), which is (n_samples, n_rx_data)
        # if self.config.device_multiplication:
        #     all_dbias = (all_powers != 0)*self.device_bias
        #     #device_bias is all zeros
        #     #non-zeros rss will have bias whereas zero rss doesn't
        #     all_powers = all_powers*self.device_weights
        #     #device_weights is all ones
        #     all_powers += all_dbias

        # if self.config.device_multiplication:
        #     # print("Here")
        #     for i in range(all_powers.shape[1]):
        #         device_id = i #assuming device ID aligns with the position in the array
        #         all_powers[:, i] = self.device_splines[device_id](all_powers[:, i].unsqueeze(1)).squeeze()
        if self.config.category_multiplication:
            all_device_categories = x_vecs[:, :, -1].long()  # Extract categories per sensor as integers
            flat_categories = all_device_categories.flatten()  # Flatten to 1D for efficient indexing
            flat_powers = all_powers.flatten()  # Flatten to align with flat_categories

            # Prepare an empty tensor to store the transformed powers
            transformed_powers = torch.empty_like(flat_powers)

            # Apply transformations for each unique category
            unique_categories = torch.unique(flat_categories)

            for category_id in unique_categories:
                # Select the indices where the current category_id matches
                category_indices = (flat_categories == category_id).nonzero(as_tuple=True)[0]

                # Apply the category-specific spline transformation
                category_spline = self.category_splines[category_id]
                transformed_powers[category_indices] = category_spline(
                    flat_powers[category_indices].unsqueeze(1)
                ).squeeze()  # Apply transformation and remove extra dimension

            # Reshape transformed powers back to the original shape of all_powers
            all_powers = transformed_powers.view(all_powers.shape)

        # if self.config.category_multiplication:
        #     all_device_categories = x_vecs[:,:,-1]
        #     #contains category of each sensor
        #     all_cweights = torch.take(self.category_weights, all_device_categories.long())
        #     #This selects the corresponding category-specific weights from self.category_weights based on the sensor category IDs (all_device_categories).
        #     all_cbias = (power_inds)*torch.take(self.category_bias, all_device_categories.long())
        #     #This selects the corresponding category-specific biases from self.category_bias based on the sensor category IDs.
        #     all_powers = all_powers * all_cweights
        #     #The RSS values (all_powers) are scaled by the category-specific weights (all_cweights).
        #     all_powers += all_cbias
        # if self.config.category_multiplication:
        #     all_device_categories = x_vecs[:, :, -1]  # Categories per sensor
        #     for i in range(all_device_categories.shape[1]):
        #         category_id = int(all_device_categories[:, i].item())
        #         all_powers[:, i] = self.category_splines[category_id](all_powers[:, i])
        # if self.config.category_multiplication:
        #     # Obtain all category IDs for all sensors across samples
        #     all_device_categories = x_vecs[:, :, -1].long()  # Shape (batch_size, num_sensors)
        #     batch_size, num_sensors = all_device_categories.shape

        #     # Stack the outputs by mapping each sensor's power values to the corresponding spline based on category
        #     # Collect the category-specific spline outputs for each position in all_powers
        #     splines_applied = torch.stack([
        #         self.category_splines[cat_id[i].item()](all_powers[:, i]) for i in range(all_device_categories.shape[1]) for cat_id in all_device_categories
        #     ], dim=1)

        #     # Update all_powers with the transformed values
        #     all_powers = splines_applied
        # if self.config.category_multiplication:
        #     # print("In Vec2Im the shape of x_vecs = ",x_vecs.shape)
        #     # Apply category-wise NonLinearBSplineSiLU transformations
        #     all_device_categories = x_vecs[:, :, -1]  # Categories per sensor
        #     # print("In Vec2Im the shape of all_device_categories = ",all_device_categories.shape)
        #     # for i in range(all_device_categories.shape[1]):
        #     #     category_id = int(all_device_categories[:, i].item())
        #     #     all_powers[:, i] = self.category_splines[category_id](all_powers[:, i])
        #     for j in range(all_device_categories.shape[0]):  # Iterate over samples
        #         for i in range(all_device_categories.shape[1]):  # Iterate over sensors in each sample
        #             category_id = int(all_device_categories[j, i].item())  # Category of current sensor in current sample
        #             # print("at sample = ", j," and the sensor = ", i, " Category ID = ", category_id)
        #             print(all_powers[j, i])
        #             all_powers[j, i] = self.category_splines[category_id](all_powers[j, i].unsqueeze(0).unsqueeze(1)).squeeze()  # Apply transformation


        coords = x_vecs[:,:,1:3].round().long().cpu().numpy()
        #taking coordinates of each rx from x_vecs for each sample
        if self.config.apply_rss_noise and (self.training or self.force_random_power_on_eval): 
            #We should only set random power in train mode.
            all_powers[power_inds] = all_powers[power_inds] + ((torch.rand(all_powers[power_inds].shape, generator=self.rand_generator, device=self.device) - 0.5) * 2*self.config.power_limit)
        if self.config.apply_power_scaling and (self.training or self.force_random_power_on_eval): 
            all_powers[power_inds] += (torch.rand(1, generator=self.rand_generator, device=self.device) - 0.5) * self.config.scale_limit*2
        x_img[torch.arange(batch_size).repeat_interleave(all_powers.shape[-1]), 0, coords[:,:,1].flatten(), coords[:,:,0].flatten()] = all_powers.flatten()
        #x_img has the shape (batch_size, channels, height, width)
        #torch.arange(batch_size).repeat_interleave(all_powers.shape[-1]):
        #torch.arange(batch_size): This generates a tensor of integers from 0 to batch_size - 1. Essentially, it represents the batch indices.
        #For example, if batch_size = 4, this would generate [0, 1, 2, 3].
        #.repeat_interleave(all_powers.shape[-1]): This repeats each batch index n_rx_data times, where n_rx_data is the number of sensors per sample (the size of the last dimension of all_powers).
        #This 0 refers to the channel index in x_img. Here, it's assigning the values to the first channel, which likely represents the processed RSS values (all_powers).

        #coords[:,:,1].flatten() and coords[:,:,0].flatten():
        #coords[:,:,1] and coords[:,:,0] represent the y and x coordinates respectively of the sensors in the 2D space (from the x_coord and y_coord).
        #.flatten(): This reshapes the 2D coordinates into 1D arrays so that the indexing can be applied in a flattened form. For instance, if coords is of shape (batch_size, n_rx_data, 2), the flattened result will have shape (batch_size * n_rx_data).

        #all_powers.flatten():
        #all_powers is the RSS values that were adjusted based on device/category scaling and biasing in earlier steps.
        #.flatten(): This converts the 2D tensor of RSS values into a 1D array to match the indexing dimensions of coords.


        ### TODO: This should be including the inputs with RSS noise and power scaling, but without category/device multiplication.
        x_img[torch.arange(batch_size).repeat_interleave(all_powers.shape[-1]), 1, coords[:,:,1].flatten(), coords[:,:,0].flatten()] = x_vecs[:,:,0].flatten()
        #Summary:The first line assigns the adjusted RSS values (after applying device/category scaling and bias) to the first channel of x_img.
        #The second line assigns the raw RSS values (without device/category modifications) to the second channel of x_img.


        # Make y_img
        tx_marker_size = 3
        #Defines the size of the marker (in pixels) for the transmitter (TX) in the y_img. The marker will be a 3x3 grid centered on the transmitter's location.
        tx_marker_value = self.config.tx_marker_value
        #This is a configurable value, likely determining the intensity or type of marker that will be placed at the transmitter's location in the image.
        #0.01 for sides and 0.92 for center
        # print("y_vecs shape during image generation= ", y_vecs.shape)
        if y_vecs is not None:
            # print("y_vecs shape during image generation= ", y_vecs.shape)
            y_img = torch.zeros((batch_size, 1, self.img_size[0], self.img_size[1]), device=self.device)
            #y_img shape (batch_size, 1, img_height, img_width)
            y_vecs = y_vecs.clone()
            pad = tx_marker_size // 2
            if isinstance(tx_marker_value, float):
                marker_value = tx_marker_value
            else:
                x_grid,y_grid = np.meshgrid( np.linspace(-(tx_marker_size//2), tx_marker_size//2, tx_marker_size),  np.linspace(-(tx_marker_size//2), tx_marker_size//2, tx_marker_size) )
                dst = np.sqrt(x_grid*x_grid + y_grid*y_grid)
                marker_value = np.exp(-( (dst)**2 / (2.0*tx_marker_value[1]**2)))
                #If tx_marker_value is not a float (likely a list or array), the code generates a Gaussian-shaped marker using a meshgrid of distances from the center.
                #The marker_value is calculated using the Gaussian function exp(-(dst^2 / 2σ^2)), where dst is the distance from the center, and σ is the standard deviation (from tx_marker_value[1]).

            ind0, ind1 = torch.where(y_vecs[:,:,0])
            #Finds the indices (ind0, ind1) where the transmitter (TX) exists (i.e., where y_vecs[:,:,0] is non-zero). These indices correspond to valid TX samples.
            coords = y_vecs[ind0, ind1, 1:3].round().long()
            #Extracts the x and y coordinates of the transmitter from y_vecs. It rounds them to the nearest integer and converts them to long integers for indexing.
            y_img[ind0, 0, coords[:,1], coords[:,0]] = 1.0 - 8*marker_value
            #Places the center of the marker at the transmitter location (coords). The intensity of the marker at the center is set to 1.0 - 8 * marker_value.
            pads = [
                [-1,-1],
                [-1, 0],
                [-1, 1],
                [ 0,-1],
                [ 0, 1],
                [ 1,-1],
                [ 1, 0],
                [ 1, 1],
            ]
            #pads array:This defines offsets for surrounding pixels to create a marker around the center.
            for shift in pads:
                # print("shift = ", shift)
                # print("coords = ", coords)
                edge_coords = coords.clone() + torch.Tensor(shift).long().to(self.device)
                #For each shift in the pads, the corresponding edge coordinates (edge_coords) are calculated.
                valid_inds = (edge_coords.min(axis=1)[0] >= 0) * (edge_coords[:,0] < self.img_size[1].cpu()) * (edge_coords[:,1] < self.img_size[0].cpu())
                #valid_inds ensures that the coordinates are valid (i.e., within the bounds of the image).
                y_img[ind0[valid_inds], 0, edge_coords[valid_inds,1], edge_coords[valid_inds,0]] = marker_value
                #Places the marker value at the edge coordinates, creating a 3x3 marker around the transmitter location.

        if self.config.include_elevation_map and self.elevation_map is not None:
            size = self.elevation_map.shape
            x_img[:,-1,1:1+size[0],1:1+size[1]] = self.elevation_map
        #If elevation data is included in the configuration, and self.elevation_map is available, it adds an elevation map to the last channel of x_img.
        #Places the elevation map in the last channel of x_img. The elevation map is assumed to be smaller than x_img, and it's placed starting from index 1 to avoid boundary issues.
        if self.config.adv_train or self.config.testing_white_box:
            #If adversarial training is enabled, the model will treat x_img as a differentiable input, meaning it can compute the gradients with respect to it.
            x_img = x_img.detach()
            #Detaches the tensor from the computation graph to ensure it can be manipulated independently.
            x_img.requires_grad = True
            #Enables gradient computation for x_img, which is necessary for adversarial training. This allows the model to compute the gradients of the loss with respect to x_img, enabling the creation of adversarial examples.
            x_img.retain_grad()
            #Ensures that the gradients for x_img are retained even after the backward pass, allowing them to be used later.
        # else:
        #     #If adversarial training is enabled, the model will treat x_img as a differentiable input, meaning it can compute the gradients with respect to it.
        #     x_img = x_img.detach()
        #     #Detaches the tensor from the computation graph to ensure it can be manipulated independently.
        #     x_img.requires_grad = True
        #     #Enables gradient computation for x_img, which is necessary for adversarial training. This allows the model to compute the gradients of the loss with respect to x_img, enabling the creation of adversarial examples.
        #     x_img.retain_grad()
        #     #Ensures that the gradients for x_img are retained even after the backward pass, allowing them to be used later.
        if y_vecs is not None:
            return x_img, y_img, y_vecs
        else:
            return x_img 



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, channel_scale=64, kernel_size=3, out_kernel_size=1, bilinear=True, depth=4, use_residual=True, dropout=0):
        super(UNet, self).__init__()
        #my paramerters n_channels = 1, n_classes = 1, channel_scale=32, kernel_size=3, out_kernel_size=1, bilinear=true, depth=3, use_residual=True
        self.n_channels = n_channels + 1
        #adding one more channel dunno why
        self.n_classes = n_classes
        self.use_residual = int(use_residual)
        #taking value = 1

        mult = [1,2,4,8,16,32,64,128,256]
        #multipliers by each layer
        if depth > 7:
            raise NotImplementedError
        #can't implement more than 7 layer
        
        self.inc = DoubleConv(self.n_channels, channel_scale, kernel_size=kernel_size, dropout=dropout)
        #incorporating the first conv opration line up
        factor = 2 if bilinear else 1
        #as bilinear = True, factor will be True
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        #moduleList container to stack down and up layers
        for i in range(depth):
            if i < depth-1:
                self.downs.append(Down(channel_scale*mult[i], channel_scale*mult[i+1], kernel_size=kernel_size, dropout=dropout))
                # for i=0,1,2 = n=input and 2n is output
            else:
                self.downs.append(Down(channel_scale*mult[i], channel_scale*mult[i+1] // factor, kernel_size=kernel_size, dropout=dropout))
                #in last layer, keep the input and output multiplier same and ended up in 8 channels
        for i in range(depth):
            if i < depth-1:
                self.ups.append(Up(channel_scale*mult[depth-i] // factor, channel_scale*mult[depth-1-i] // factor, bilinear=bilinear, res_channels=self.use_residual*channel_scale*mult[depth-i] // factor, kernel_size=kernel_size, dropout=dropout))
            #for up, it is a bit complicated.
            #take each from previous layer
            #output layer is half of the input
            #bilinear=true
            else:
                self.ups.append(Up(channel_scale*mult[depth-i] // factor, channel_scale*mult[depth-1-i], bilinear=bilinear, res_channels=self.use_residual*channel_scale*mult[depth-i] // factor, kernel_size=kernel_size, dropout=dropout))
        self.outc = nn.Sequential(
            OutConv(channel_scale, n_classes, kernel_size=out_kernel_size),
            #just run 1x1 kernel on the last layer and produce a single channel output
            nn.Sigmoid(),
            nn.ReLU())

    def forward(self, x):
        x1 = self.inc(x)
        x_values = [x1]
        for down in self.downs:
            x_values.append( down(x_values[-1]) )
        x = x_values[-1]
        ind = -2
        for up in self.ups:
            x = up(x, x_values[ind])
            ind += -1
        logits = self.outc(x)
        return logits


class MLPLocalization(nn.Module):
    def __init__(self, in_features, out_features=2, channel_scale=256, device=torch.device('cuda')):
        super().__init__()
        num_features = [
            in_features,
            channel_scale*1,
            channel_scale*8,
            channel_scale*8,
            out_features
                        ]
        self.layers = nn.Sequential(
            nn.Linear(num_features[0], num_features[1]),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[2]),
            nn.LeakyReLU(),
            nn.Linear(num_features[2], num_features[3]),
            nn.LeakyReLU(),
            nn.Linear(num_features[3], num_features[4]),
            nn.LeakyReLU(),
        )
        self.to(device)

    def forward(self, x):
        rss = x[:,:,0]
        return self.layers(rss)
    

class EnsembleLocalization(nn.Module):
    def __init__(self, config, n_channels, n_classes, img_shape, device, elevation_map=None, num_models=20, channel_scale=32, kernel_size=3, out_kernel_size=1, scales=None, bilinear=True, depth=3, use_residual=True, input_resolution=5, single_model_training=True, total_sensor=34):
        #parameters: self.params, n_channels=1, output_channel=1, self.img_size, self.device, num_models=1, channel_scale=channel_scale, input_resolution=self.params.meter_scale, depth=depth, elevation_map=self.rss_loc_dataset.elevation_tensors[0]
        super(EnsembleLocalization, self).__init__()
        #n_channels = 1 cause image will be from the coordinates
        #n_classes says only 1 output channel
        #img_shape from the previous layer
        #num_models = 1 in our case
        #channel scale = 32 measns number of channel in between factor n=32
        #kernel_size = 3x3
        #out_kernel_size = 1x1 means the kernel on the very last layer

        self.vec2im = Vec2Im(config, img_shape, device, elevation_map=elevation_map, max_num_rx = total_sensor)
        #vec2Im makes the image
        self.single_model_training = single_model_training
        #training only one model
        self.output_shape = img_shape
        #output should be an image so it is of the same shape as an image
        self.n_channels = n_channels
        #number of input channels is 1 as not including map
        #but dunno why vec_to_img has one extra channel
        self.n_classes = n_classes
        #how many channel in output
        self.return_preds = False
        #return predictions false
        models = []
        #list to store all the models
        for i in range(num_models):
            mod = nn.Sequential( 
                #nn.MaxPool2d(img_scale//input_resolution),
                UNet(n_channels, n_classes, channel_scale, kernel_size, out_kernel_size, bilinear, depth, use_residual),
                #nn.ConvTranspose2d(1, 1, kernel_size=img_scale//input_resolution, stride=img_scale//input_resolution),
                nn.Upsample((self.output_shape[0], self.output_shape[1]), mode='bilinear')
                )
            models.append(mod)
        #else:
        #    for i in range(num_models):
        #        mod = UNet(n_channels, n_classes, channel_scale, kernel_size, out_kernel_size, bilinear, depth, use_residual)
        #        models.append(mod)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        if isinstance(x, tuple):
            x_img, y_img, y_vecs = self.vec2im(x)
        else:
            x_img = self.vec2im(x)

        if self.training and self.single_model_training:
            random_ind = torch.randint(len(self.models), (1,))
            preds = self.models[random_ind](x_img)
        else:
            preds = []
            for model in self.models:
                preds.append(model(x_img))
            preds = torch.cat(preds, dim=1)
            # preds = preds.mean(dim=1, keepdim=True)
            # avg = avg / avg.view(*avg.size()[:2], -1).sum(dim=2, keepdims=True).unsqueeze(-1)
        if isinstance(x, tuple):
            return preds, x_img, y_img
        else:
            return preds
        #soft_avg = self.softmax(avg.view(*avg.size()[:2], -1)).view_as(avg)
    
    def predict(self, x, input_is_pred_img=False):
        if input_is_pred_img:
            preds = x
        else:
            single_model_training = self.single_model_training
            self.single_model_training = False
            preds = self.forward(x)
            self.single_model_training = single_model_training
        
        peaks, peak_locs = torch.max( preds.reshape((preds.shape[0], len(self.models), -1)), dim=-1)
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

    
    def com_predict(self, x, input_is_pred_img=False):
        if input_is_pred_img:
            preds = x
        else:
            single_model_training = self.single_model_training
            self.single_model_training = False
            preds = self.forward(x)
            self.single_model_training = single_model_training
        
        peaks, peak_locs = torch.max( preds.reshape((preds.shape[0], len(self.models), -1)), dim=-1)
        mean_pred = preds.mean(axis=1)
        centers_of_mass = get_centers_of_mass(mean_pred)
        new_preds = torch.hstack((centers_of_mass, peaks.mean(dim=1, keepdim=True)))
        return new_preds 
        

def unravel_indices(indices: torch.LongTensor, shape, ) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')
    coord = torch.stack(coord[::-1], dim=-1)
    return coord


def get_centers_of_mass(tensor):
#taken from:https://gitlab.liu.se/emibr12/wasp-secc/blob/cb02839115da475c2ad593064e3b9daf2531cac3/utils/tensor_utils.py    
    """
    Args:
        tensor (Tensor): Size (*,height,width)
    Returns:
        Tuple (Tensor): Tuple of two tensors of sizes (*)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    width = tensor.size(-1)
    height = tensor.size(-2)
    
    x_coord_im = torch.linspace(0,width,width).repeat(height,1).to(device)
    y_coord_im = torch.linspace(0,width,height).unsqueeze(0).transpose(0,1).repeat(1,width).to(device)
    
    x_mean = torch.mul(tensor,x_coord_im).sum(-1).sum(-1)/torch.add(tensor.sum(-1).sum(-1),1e-10)
    y_mean = torch.mul(tensor,y_coord_im).sum(-1).sum(-1)/torch.add(tensor.sum(-1).sum(-1),1e-10)
    
    return torch.stack((y_mean, x_mean)).T


class TiremMLP(torch.nn.Module):
    def __init__(self, num_features=[14,200], device='cuda', dropout=0.01, input_dropout=0.1) -> None:
        super().__init__()
        self.tirem_bias = nn.Parameter(torch.ones(1))
        self.layers = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(num_features[0], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], 1),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.to(device)

    def forward(self, x, tirem_pred):
        #tirem_bounded = nn.functional.relu(tirem_pred + self.tirem_bias, inplace=True)
        return self.layers(x)# + tirem_bounded[:,None]