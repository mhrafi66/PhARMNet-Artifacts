"""NRDZ TIREM Propagation Model."""

import numpy as np
import pandas as pd
import scipy.io as sio
from time import time
from numba import jit
from ctypes import *
from common import *
from tirem_params import *
import math
from datetime import date
import concurrent.futures
import multiprocessing as mp
from multiprocessing import shared_memory
import rasterio
import utm

tirem_dll = None

def main():
    """Entry point into tirem code."""
    load_tirem_lib() #Trying to call some tirem library which will have some dll file
    bs_endpoint_name = "USTAR" #Take a bsendpoint name as string
    bs_is_tx = 0 #that means the point is a receiver
    txheight = 1.5 #transmitter height
    rxheight = 3 #receiver height
    bs_lon = -111.84167 #logtitude of the basestation
    bs_lat = 40.76895 #latitude of the basestation
    freq = 462.7 #Transmission frequency in MHz
    polarz = 'H' #Horizontal Polarization
    run_date = tirem_pred(bs_endpoint_name, bs_is_tx, txheight, rxheight, bs_lon, bs_lat, freq, polarz, generate_features=True)
    #f = np.load(bs_endpoint_name + '_tirem_rssi_' + str(freq) + run_date+".npz")
    #f = np.load(bs_endpoint_name + '_diff_parameter_' + str(freq) + run_date+".npy")
    #f = np.load(bs_endpoint_name + '_diff_param_obstacle_with_max_h_over_r_' + str(freq) + run_date+".npy")
    #f = np.load(bs_endpoint_name + '_elevation_angles_NLOS_' + str(freq) + run_date+".npy")
    #f = np.load(bs_endpoint_name + '_elevation_angles_tx_rx_' + str(freq) + run_date+".npy")
    #f = np.load(bs_endpoint_name + '_LOS_NLOS_' + str(freq) + run_date+".npy")
    #f = np.load(bs_endpoint_name + '_number_knife_edges_' + str(freq) + run_date+".npy")
    #f = np.load(bs_endpoint_name + '_number_obstacles_' + str(freq) + run_date+".npy")
    #f = np.load(bs_endpoint_name + '_shadowing_angles_' + str(freq) + run_date + ".npy")


def tirem_pred(bs_endpoint_name, bs_is_tx, txheight, rxheight, bs_lon, bs_lat, freq,
               polarz, generate_features=1, map_type="corrected", map_filedir="/uufs/chpc.utah.edu/common/home/u1472438/dl-image-localization-main/tirem_prop/corrected_dsm.tif", gain=0, first_call=0,
               extsn=0,
               refrac=450, conduc=50, permit=81, humid=80, side_len=0.5, sampling_interval=0.5):
    """
    Code for running TIREM and generating features for augmented modeling. Adapted from SLC_TIREM_v3_Python.

    :param bs_endpoint_name: basestation endpoint name e.g. "cbrssdr1-ustar-comp"
    :param bs_is_tx: boolean, is the basestation a transmitter (1) or a receiver (0)
    :param txheight: transmitter height (m) -DO NOT INCLUDE THE BUILDING HEIGHT OR
    ELEVATION, JUST THE TX HEIGHT W.R.TO THE SURFACE IT STAYS ON
    :param rxheight: receiver height (m) -DO NOT INCLUDE THE BUILDING HEIGHT OR
    ELEVATION, JUST THE RX HEIGHT W.R.TO THE SURFACE IT STAYS ON
    :param bs_lon: basestation longitude
    :param bs_lat: basestation latitude
    :param freq: transmit frequency in MHz
    :param polarz: polarization of the wave ('H' for horizontal, 'V' for vertical)
            produced by the Tx antenna
    :param generate_features: boolean, Generate only TIREM predictions (0) / also generate
                       features (1)
    :param map_type: Fusion map ("fusion") or lidar DSM map ("lidar"). The fusion map
              struct has to have the fields "dem" for digital elevation map and
              "hybrid_bldg" for building heights. The lidar map has to have the field
              "data" having combined information of elevations and building heights.
    :param map_filedir: map file directory including the filename and extension
    gain: antenna gain in dB
    :param gain: antenna gain in dB
    :param first_call: boolean, 1 loads the TIREM library for people using it for
               the first time. #what is a tirem library? dunno
    :param extsn: boolean, 0 is false; anything else is true. False = new profile,
           true = extension of last profile terrain
    :param refrac: surface refractivity, range: (200 to 450.0) "N-Units"
    :param conduc: conductivity, range: (0.00001 to 100.0) S/m
    :param permit: relative permittivity of earth surface, range: (1 to 1000)
    :param humid: humidity, units g/m^3, range: (0 to 110.0)
    :param side_len: grid_cell side length (m)
    :param sampling_interval: sampling interval along the tx-rx link. e.g. 0.5
                       means a given grid cell is sampled twice. Side length
                       of 0.5 and sampling interval of 0.5 mean that the
                       Tx-Rx horizontal array step size 0.5*0.5 = 0.25 m.
    :return: run_date, the date the script completed its run.
    Saves tirem rssi and generated features to the current folder.
    Serhat Tadik, Aashish Gottipati, Michael A. Varner, Gregory D. Durgin
    """
    if map_filedir.endswith('mat'):
        # load the map
        map_struct = sio.loadmat(map_filedir)['SLC']
        #If the map was in mat format, then read with sio

        # Define a new struct named SLC
        SLC = map_struct[0][0]
        column_map = dict(zip([name for name in SLC.dtype.names], [i for i in range(len(SLC.dtype.names))]))

        bs_x, bs_y, idx = lon_lat_to_grid_xy(np.array([bs_lon]), np.array([bs_lat]), SLC,
                                             column_map)  # Longitude - latitude
        #getting the (x,y) grid from the latitude and longtitude value

    elif map_filedir.endswith('tif'):
        alt_coords = pd.read_csv('/uufs/chpc.utah.edu/common/home/u1472438/dl-image-localization-main/tirem_prop/alt_coords.csv') #where is the file? dunno
        alt_coords.iloc[:, 0] = alt_coords.iloc[:, 0].str.lstrip('#')
        # alt_coords = pd.read_csv('/uufs/chpc.utah.edu/common/home/u1472438/dl-image-localization-main/tirem_prop/alt_coords.csv', comment='#') #where is the file? dunno
        #read the given file with five columns: Station, Lat, Long, Height(m), Type(Fixed,Rooftop,Dense)
        alt_lat = np.array(alt_coords['Latitude (deg N)'])
        #Take all the latitude
        alt_lon = np.array(alt_coords['Longitude (deg E)'])
        #take all the longitude
        #take the latitude and logitude stored somewhere
        #what is alt? dunno
        # print("Alt Coords = ",alt_coords)
        # print("Alt Lat = ", alt_lat)
        # print("Alt Long = ", alt_lon)
        # print(abcd)

        dsm_corners = np.array([
            [1447.88347094,  492.35956159],
            [3847.88347094, 2712.35956159]
        ])		#why these hardcoded values? dunno
        #Take the dsm_corners. But why these hardcoded values?
        dsm_object = rasterio.open(map_filedir) #read the map
        # print(dsm_object."""meta)"""
        """
        {'driver': 'GTiff', 'dtype': 'float32', 'nodata': -3.4028234663852886e+38, 'width': 9805, 'height': 7223, 'count': 1, 
        'crs': CRS.from_epsg(26912), 'transform': Affine(0.5000000000000014, 0.0, 426446.249999999,
       0.0, -0.5000000000000019, 4514907.249999999)}
        """
        dsm_object.origin = np.array(dsm_object.transform)[:6].reshape(2,3) @ np.array([0, dsm_object.shape[0], 1])
        """print(dsm_object.origin)
            [ 426446.25 4511295.75]
        """
        
        #Find the origin of the dsm_object
        dsm_map = dsm_object.read(1)     # a np.array containing elevation values
        # print("dsm_map = ", dsm_map.shape) (7223, 9805)
        """print("dsm_map = ", dsm_map)
        [[1457.7  1457.65 1457.67 ... 1722.61 1722.9  1723.13]
        [1457.67 1457.62 1457.66 ... 1722.59 1722.8  1723.3 ]
        [1457.61 1457.62 1457.66 ... 1722.51 1722.74 1723.03]
        ...
        [1304.19 1304.19 1304.24 ... 1497.29 1489.08 1489.05]
        [1304.21 1304.23 1304.22 ... 1489.16 1497.78 1489.1 ]
        [1304.19 1304.19 1304.21 ... 1493.13 1489.1  1493.08]]  
        """
        utm_transform = np.array(dsm_object.transform).reshape((3,3))
        # print("utm_transform 1 = ", utm_transform) (3,3)
        utm_transform[:2,2] -= dsm_object.origin
        # print("utm_transform 2 = ", utm_transform) (3,3)
        inv_transform = np.linalg.inv(utm_transform)
        # print("inv_transform = ", inv_transform) (3,3)
        a_ind = (inv_transform @ np.array([dsm_corners[0,0], dsm_corners[0,1], 1])).round().astype(int)
        b_ind = (inv_transform @ np.array([dsm_corners[1,0], dsm_corners[1,1], 1])).round().astype(int)
        #what is this a_ind and b_ind? dunno
        # print("a_ind = ", a_ind) [2896 6238    1]
        # print("b_ind = ", b_ind) [7696 1798    1]

        def lat_lon_to_xy(alt_lat, alt_lon, origin=np.array([a_ind[0], b_ind[1]])):
            alt_idx = np.vstack(utm.from_latlon(alt_lat, alt_lon)[:2]) - dsm_object.origin[:,None]
            alt_idx = inv_transform @ np.vstack((alt_idx, np.ones(len(alt_lat))))
            xy = (alt_idx[:2] - np.array([a_ind[0], b_ind[1]])[:,None]).T
            return xy
        
        sub_img = np.flipud(dsm_map[b_ind[1]:a_ind[1]+1, a_ind[0]:b_ind[0]+1])
        """print("Sub_img = ", sub_img)
        [[1382.52 1382.52 1382.51 ... 1472.83 1472.87 1472.82]
        [1382.5  1382.53 1382.54 ... 1472.92 1472.81 1472.85]
        [1382.52 1382.55 1382.54 ... 1472.81 1472.81 1472.8 ]
        ...
        [1417.87 1417.87 1417.89 ... 1718.58 1718.74 1718.91]
        [1417.83 1417.91 1417.89 ... 1718.7  1718.86 1719.08]
        [1417.81 1417.9  1417.9  ... 1718.83 1719.06 1719.21]]
        print("Sub_img shape = ", sub_img.shape) (4441, 4801)
        
        """
        bs_coords = lat_lon_to_xy(alt_lat, alt_lon)
        """print("bs_coords 1 = ", bs_coords)
        [[1472.64107787 1330.96935185]
        [2735.01488759 1400.14562373]
        [1125.8355318  2176.90376744]
        [ 410.26623974 2762.13940056]
        [1875.18195064 2023.37136242]
        [3021.2471879  3585.9108668 ]
        [3999.41980461 2503.90839596]
        [2789.48404777  875.77787989]
        [2065.53451198 2769.90104226]
        [3039.71442011 1719.22929409]
        [3913.1605419  1478.23023339]
        [2926.82394111 2139.12858684]
        [3838.46161885 3753.14093932]
        [1349.67751665 2799.62759142]
        [ 167.38403871 3513.03110698]
        [1116.31220035 1701.83607192]
        [2149.56149746 1120.55112242]
        [2907.9052647   689.51178187]
        [1899.07458046  643.88103646]
        [2284.91191622  261.02277793]
        [2646.63141517  915.71121937]
        [3086.16616899 1450.56720701]
        [2745.78752319 1544.79253298]
        [2350.86383615 1217.51141463]]
        """
        bs_coords[:,1] = len(sub_img) - bs_coords[:,1]
        """print("bs_coords 2 = ", bs_coords)
        [[1472.64107787 3110.03064815]
        [2735.01488759 3040.85437627]
        [1125.8355318  2264.09623256]
        [ 410.26623974 1678.86059944]
        [1875.18195064 2417.62863758]
        [3021.2471879   855.0891332 ]
        [3999.41980461 1937.09160404]
        [2789.48404777 3565.22212011]
        [2065.53451198 1671.09895774]
        [3039.71442011 2721.77070591]
        [3913.1605419  2962.76976661]
        [2926.82394111 2301.87141316]
        [3838.46161885  687.85906068]
        [1349.67751665 1641.37240858]
        [ 167.38403871  927.96889302]
        [1116.31220035 2739.16392808]
        [2149.56149746 3320.44887758]
        [2907.9052647  3751.48821813]
        [1899.07458046 3797.11896354]
        [2284.91191622 4179.97722207]
        [2646.63141517 3525.28878063]
        [3086.16616899 2990.43279299]
        [2745.78752319 2896.20746702]
        [2350.86383615 3223.48858537]]            
        """


    # generate the map
    if map_type == "corrected":
        slc_map = sub_img.copy().astype(np.float64)
        #This slc_map actually storing all the elevation
        nrows, ncols = slc_map.shape
        # print(nrows, ncols) 4441 4801
    else:
        if map_type == "fusion":
            slc_map = SLC[column_map['dem']] + 0.3048 * SLC[column_map['hybrid_bldg']]
        elif map_type == "lidar":
            slc_map = SLC[column_map['data']]
        nrows = int(SLC[column_map['nrows']])
        ncols = int(SLC[column_map['ncols']])
    shape = slc_map.shape
    shm = shared_memory.SharedMemory(create=True, size=slc_map.nbytes, name='tirem_np')
    shared_slc_map = np.ndarray(slc_map.shape, dtype=slc_map.dtype, buffer=shm.buf)
    # print("shared_slc_map = ", shared_slc_map.shape) (4441, 4801)
    shared_slc_map[:] = slc_map[:]
    if map_type == "corrected":
        loop_options = zip(alt_coords['Station'], bs_coords, alt_coords['Height (m)'], alt_coords['Type'])
    else:
        loop_options = [[bs_endpoint_name, [bs_x, bs_y], rxheight, 'Rooftop_old']]
    for name, coords, height, station_type in loop_options:
        print('Producing TIREM map for ', name)
        bs_endpoint_name = name
        bs_x, bs_y = np.rint(coords).astype(int)
        rxheight = height
        if station_type == 'Rooftop':
            rxheight = 3
        elif name == 'EBC-DD':
            rxheight = 3
        # parameters
        input_params = {'bs_endpoint_name': bs_endpoint_name, 'bs_is_tx': bs_is_tx, 'txheight': txheight,
                        'rxheight': rxheight,
                        'bs_lon': bs_lon, 'bs_lat': bs_lat, 'bs_x': bs_x, 'bs_y': bs_y, 'freq': freq,
                        'polarz': polarz + '   ', 'generate_features': generate_features,
                        'map_type': map_type, 'map_filedir': map_filedir, 'gain': gain, 'first_call': first_call,
                        'extsn': extsn,
                        'refrac': refrac, 'conduc': conduc, 'permit': permit, 'humid': humid, 'side_len': side_len,
                        'sampling_interval': sampling_interval}

        params = Params(**input_params)

        ## Get tirem loss for all points

        # determine the tx/rx grid (raster) coordinates
        if bs_is_tx:
            tx = np.array([params.bs_x, params.bs_y])
        else:
            rx = np.array([params.bs_x, params.bs_y])
        ##Store the basestation coordinates to rx

        # the gain
        EIRP = params.gain

        ## Run TIREM
        t = time()

        # Initialize variables
        wavelength = 300 / params.freq
        tirem_rssi = np.ones((nrows, ncols)) * np.NaN
        distances = np.ones((nrows, ncols)) * np.NaN

        t0 = time()
        if generate_features:
            # line-of-sight / non-line-of-sight
            LOS = np.ones((nrows, ncols))

            # number of blocking obstructions
            number_obstacles = np.zeros((nrows, ncols))

            # number of knife edges
            number_knife_edges = np.zeros((nrows, ncols))

            # elevation angle between Tx & Rx with no consideration of obstacles in between
            elevation_angles_tx_rx = np.zeros((nrows, ncols))

            # elevation angles considering the obstacles in between Tx & Rx
            elevation_angles_NLOS = np.zeros((nrows, ncols, 2))

            # shadowing angles for the first and last knife-edges
            first_last_shadowing_angles = np.zeros((nrows, ncols, 2))

            # diffraction angles for the first and last knife-edges
            first_last_diffraction_angles = np.zeros((nrows, ncols, 2))

            # Fresnel-Kirchhoff diffraction parameter
            diff_parameter = np.zeros((nrows, ncols, 2))

            # Fresnel-Kirchhoff diffraction parameter for the obstacle with the
            # largest h (height of the obstacle above the tx-rx link) /r (fresnel
            # zone at that point) ratio
            diff_param_obstacle_with_max_h_over_r = np.zeros((nrows, ncols))

        # Determine tx / rx height from the map
        if bs_is_tx:
            tx_elevation = slc_map[tx[1], tx[0]]
        else:
            rx_elevation = slc_map[rx[1], rx[0]]
        # Loop through the map pixels


        grid = np.array(np.meshgrid(range(ncols), range(nrows))).T.reshape(-1,2)
        # print("Grid shape = ", grid.shape) (21321241, 2)
        if bs_is_tx:
            rx_vec = grid + 1
            vec = rx_vec
            tx_vec = tx
            #rx_elevation = slc_map[grid[:,1], grid[:,0]]
        else:
            tx_vec = grid + 1
            vec = tx_vec
            rx_vec = rx
            #tx_elevation = slc_map[grid[:,1], grid[:,0]]
        distances[grid[:,1], grid[:,0]] = np.linalg.norm(rx_vec - tx_vec, 2, axis=1) * side_len
        # print("Grid shape = ",grid.shape) (21321241, 2)
        # print("Distances shape = ",distances.shape) (4441, 4801)

        finished_elements = 0
        t0 = time()
        #tirem_parallel(157, tx if bs_is_tx else rx, bs_is_tx, side_len, shape, sampling_interval, EIRP, params, generate_features)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(tirem_parallel, a, tx if bs_is_tx else rx, bs_is_tx, side_len, shape, sampling_interval, EIRP, params, generate_features): a for a in range(ncols)}
            for future in concurrent.futures.as_completed(futures):
                a = futures[future]
                try:
                    row, result = future.result()
                    #tirem_rssi[:,row] = result
                    LOS[:,row], number_obstacles[:,row], number_knife_edges[:,row], elevation_angles_tx_rx[:,row], elevation_angles_NLOS[:,row], first_last_shadowing_angles[:,row], first_last_diffraction_angles[:,row], diff_parameter[:,row], diff_param_obstacle_with_max_h_over_r[:,row] = result
                    #LOS[:,row], number_obstacles[:,row], first_last_shadowing_angles[:,row], diff_param_obstacle_with_max_h_over_r[:,row] = result
                    finished_elements += 1
                    print(f"{(time() - t0)/finished_elements:.4f}    {round(finished_elements/ncols, 4)*100:.2f}   {finished_elements}/{ncols}    ", end='\r')
                except Exception as exc:
                    print('%i generated an exception %s' % (a, exc))

        #for a in range(ncols):

        #    # Display progress
        #    if np.mod(a, 50) == 0:
        #        print("Progress: ")
        #        print(str(100 * a / ncols) + '   %')
        #    print('Row Time', time() - trow, tinit, tbuild, ttirem, twrap)
        #    trow = time()
        #    tinit = 0.0
        #    tbuild = 0.0
        #    ttirem = 0.0
        #    twrap = 0.0

        #    #for b in range(nrows):

        elapsed = time() - t
        print("Elapsed time is:" + str(elapsed))
        today = date.today()

        #np.savez(bs_endpoint_name + '_tirem_rssi_' + str(freq) + today.strftime("%b-%d-%Y") + '.npz', tirem_rssi, distances, params)

        if generate_features:
            np.save(bs_endpoint_name + '_LOS_NLOS_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy', LOS)
            np.save(bs_endpoint_name + '_number_obstacles_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                    number_obstacles)
            np.save(bs_endpoint_name + '_number_knife_edges_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                    number_knife_edges)
            np.save(bs_endpoint_name + '_elevation_angles_tx_rx_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                    elevation_angles_tx_rx)
            np.save(bs_endpoint_name + '_elevation_angles_NLOS_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                    elevation_angles_NLOS)
            np.save(bs_endpoint_name + '_shadowing_angles_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                    first_last_shadowing_angles)
            np.save(bs_endpoint_name + '_diffraction_angles_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy',
                    first_last_diffraction_angles)
            np.save(bs_endpoint_name + '_diff_parameter_' + str(freq) + today.strftime("%b-%d-%Y") + '.npy', diff_parameter)
            np.save(bs_endpoint_name + '_diff_param_obstacle_with_max_h_over_r_' + str(freq) + today.strftime(
                "%b-%d-%Y") + '.npy', diff_param_obstacle_with_max_h_over_r)

    return today.strftime("%b-%d-%Y")

def tirem_parallel(a, x_val, bs_is_tx, side_len, shape, sampling_interval, EIRP, params, generate_features):
    existing_shm = shared_memory.SharedMemory(name='tirem_np')
    slc_map = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
    nrows = len(slc_map)
    tmp_res = np.ones(nrows) * np.NaN
    LOS = np.ones((nrows))
    number_obstacles = np.zeros((nrows))
    number_knife_edges = np.zeros((nrows))
    elevation_angles_tx_rx = np.zeros((nrows))
    elevation_angles_NLOS = np.zeros((nrows, 2))
    first_last_shadowing_angles = np.zeros((nrows, 2))
    first_last_diffraction_angles = np.zeros((nrows, 2))
    diff_parameter = np.zeros((nrows, 2))
    diff_param_obstacle_with_max_h_over_r = np.zeros((nrows))

    wavelength = 300 / params.freq
    if bs_is_tx:
        tx_elevation = slc_map[x_val[1], x_val[0]]
    else:
        rx_elevation = slc_map[x_val[1], x_val[0]]
    
    time1 = 0
    time1a = 0
    time2 = 0
    time3 = 0
    time3a = 0
    time4 = 0
    time5 = 0

    for b in range(nrows):
            flag = 0
            cntr = 0
            if bs_is_tx:
                rx = np.array([a, b]) + np.array([1, 1])
                tx = x_val
            else:
                tx = np.array([a, b]) + np.array([1, 1])
                rx = x_val
            if (rx == tx).all() == 0:
                # build arrays and calculate TIREM's predictions
                t0 = time()
                [d_array, e_array] = build_arrays(side_len, sampling_interval, tx, rx, slc_map)
                #tmp_res[b] = EIRP - get_tirem_loss(d_array, e_array, params)

                if generate_features:

                    # Get
                    if bs_is_tx:
                        rx_elevation = slc_map[b, a]
                    else:
                        tx_elevation = slc_map[b, a]
                    # Get the nonzero elements of the distance and elevation arrays

                    nonzero_e_array_indices = [e_array != 0]
                    d_array_nonzero = d_array[tuple(nonzero_e_array_indices)]
                    e_array_nonzero = e_array[tuple(nonzero_e_array_indices)]

                    # Calculate the total Rx and Tx heights and the slope between them
                    rx_total_height = rx_elevation + params.rxheight
                    tx_total_height = tx_elevation + params.txheight
                    slope = (rx_total_height - tx_total_height) / max(d_array_nonzero)

                    # Elevation angle
                    elevation_angles_tx_rx[b] = 90 - math.degrees(math.atan(slope))

                    # Blocking obstruction information, bo_info, initialization
                    test1 = (d_array_nonzero * slope + tx_total_height) < (e_array_nonzero)
                    ind = test1.argmax()
                    ind = ind if test1[ind] else -1
                    inds = []
                    while ind >= 0:
                        inds.append(ind)
                        test1[ind:ind+201] = 0
                        ind = test1.argmax()
                        ind = ind if test1[ind] else -1
                    inds = np.array(inds).astype(int)
                    h_arr = e_array_nonzero - (d_array_nonzero * slope + tx_total_height)
                    r_arr = np.sqrt(wavelength * d_array_nonzero[inds] * (d_array_nonzero[-1] - d_array_nonzero[inds])) * np.sqrt(1 + slope ** 2) / d_array_nonzero[-1]
                    bo_info = np.vstack((d_array_nonzero[inds], h_arr[inds], r_arr)).T
                    number_obstacles[b] = len(inds)
                    if number_obstacles[b]:
                        LOS[b] = 0
                    #d1 = max(np.sqrt((d_array_nonzero[inds[0]] * slope) ** 2 + d_array_nonzero[inds[0]] ** 2), 0.25)
                    #d2 = max(np.sqrt(((d_array_nonzero[-1] - d_array_nonzero[inds[0]]) * slope) ** 2 + (d_array_nonzero[-1] - d_array_nonzero[inds[0]]) ** 2), 0.25)
                    #diff_parameter[b, 0] = h[inds[0]] * np.sqrt(2 * (d1 + d2) / (wavelength * d1 * d2))
                    

                    # find the obstacle with largest h / r ratio and calculate the diffraction parameter for it
                    if (bo_info == np.array([[0, 0, 0]])).all() == 0:
                        idxx = (bo_info[:, 1] / bo_info[:, 2]) == max(bo_info[:, 1] / bo_info[:, 2])
                        d1_ = max(bo_info[idxx, 0], 0.25)
                        h_ = bo_info[idxx, 1]
                        diff_param_obstacle_with_max_h_over_r[b] = h_ * np.sqrt(2 * (d_array_nonzero[-1] * (np.sqrt(1 + slope ** 2))) / (wavelength * d1_ * (d_array_nonzero[-1] - d1_) * (1 + slope ** 2)))

                    # knife edge information, ke_info, initialization
                    d_ke = 0
                    i = 0
                    ke_ind = (((d_array_nonzero - d_ke) * slope + tx_total_height) < (e_array_nonzero)).argmax()
                    first_last_shadowing_angles[b, 0] = math.degrees(math.atan(
                                    (e_array_nonzero[ke_ind] - tx_total_height) / d_array_nonzero[ke_ind])) - math.degrees(
                                    math.atan(slope))
                    saved_inds = []
                    while ke_ind > 0:
                    #while len(ke_inds):
                        saved_inds.append(ke_ind)
                        d_ke = d_array_nonzero[ke_ind]
                        tx_total_height = e_array_nonzero[ke_ind]
                        slope = (rx_total_height - tx_total_height) / (max(d_array_nonzero) - d_array_nonzero[ke_ind])
                        obstacles = ((d_array_nonzero - d_ke) * slope + tx_total_height) < (e_array_nonzero)
                        obstacles[:ke_ind+201] = 0
                        ke_ind = obstacles.argmax()
                    ke_inds = np.array(saved_inds).astype(int)
                    ke_info = np.array([d_array_nonzero[ke_inds], e_array_nonzero[ke_inds]]).T
                    number_knife_edges[b] = len(ke_inds)
                    #ke_info = np.array([[0, 0]])
                    #flag = 0
                    #cntr = 0
                    #d_ke = 0
                    #tx_total_height = tx_elevation + params.txheight
                    #slope = (rx_total_height - tx_total_height) / max(d_array_nonzero)
                    #for i in range(len(e_array_nonzero) - 1):
                    #    cntr = cntr + 1
                    #    # different from blocking obstacle, change the slope each time you come across an obstacle
                    #    if ((d_array_nonzero[i] - d_ke) * slope + tx_total_height) < (e_array_nonzero[i]):
                    #        # calculate number of knife edges
                    #        if flag == 0 or cntr > 200:
                    #            #number_knife_edges[b, a] += 1
                    #            # if it's the first ke, calculate the shadowing angle
                    #            if flag == 0:
                    #                first_last_shadowing_angles[b, 0] = math.degrees(math.atan(
                    #                    (e_array_nonzero[i] - tx_total_height) / d_array_nonzero[i])) - math.degrees(
                    #                    math.atan(slope))

                    #            # update ke_info with distance and elevation information
                    #            ke_info = np.append(ke_info, [[d_array_nonzero[i], e_array_nonzero[i]]], 0)
                    #            if flag == 0:
                    #                ke_info = np.delete(ke_info, 0, 0)
                    #            # update parameters
                    #            d_ke = d_array_nonzero[i]
                    #            tx_total_height = e_array_nonzero[i]
                    #            slope = (rx_total_height - tx_total_height) / (max(d_array_nonzero) - d_array_nonzero[i])

                    #            cntr = 0
                    #            flag = 1

                    # redefine the original slope in case it changes in the previous loop
                    rx_total_height = rx_elevation + params.rxheight
                    tx_total_height = tx_elevation + params.txheight
                    slope = (rx_total_height - tx_total_height) / max(d_array_nonzero)

                    # if there is at least one ke, calculate NLOS elevation angles, shadowing angle for the
                    # last ke, and diffraction angles.
                    if (ke_info == np.array([[0, 0]])).all() == 0:

                        h = ke_info[-1, 1] - ((ke_info[-1, 0] - d_array_nonzero[-1]) * slope + rx_total_height)
                        d1 = max(np.sqrt(((ke_info[-1, 0] - d_array_nonzero[-1]) * slope) ** 2 + (ke_info[-1, 0] - d_array_nonzero[-1]) ** 2), 0.25)
                        d2 = max(np.sqrt(((d_array_nonzero[0] - ke_info[-1, 0]) * slope) ** 2 + (d_array_nonzero[0] - ke_info[-1, 0]) ** 2), 0.25)

                        diff_parameter[b, 1] = h * np.sqrt(2 * (d1 + d2) / (wavelength * d1 * d2))
                        first_last_shadowing_angles[b, 1] = math.degrees(math.atan((ke_info[-1, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[-1, 0]))) + math.degrees(math.atan(slope))
                        elevation_angles_NLOS[b, 0] = 90 - math.degrees(math.atan((ke_info[0, 1] - tx_total_height) / max((ke_info[0, 0] - d_array_nonzero[0]),0.25)))
                        elevation_angles_NLOS[b, 1] = 90 - math.degrees(math.atan((ke_info[-1, 1] - rx_total_height) / max((d_array_nonzero[-1] - ke_info[-1, 0]),0.25)))

                        if ke_info.shape[0] == 1:
                            slope_ke_to_rx = (ke_info[0, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[0, 0])
                            first_last_diffraction_angles[b, 0] = math.degrees(math.atan((ke_info[0, 1] - tx_total_height) / ke_info[0, 0])) + math.degrees(math.atan(slope_ke_to_rx))
                            first_last_diffraction_angles[b, 1] = first_last_diffraction_angles[b, 0]

                        if ke_info.shape[0] != 1:

                            slope_1 = (ke_info[0, 1] - tx_total_height) / ke_info[0, 0]
                            slope_2 = (ke_info[0, 1] - ke_info[1, 1]) / (ke_info[1, 0] - ke_info[0, 0])
                            first_last_diffraction_angles[b, 0] = math.degrees(math.atan(slope_1)) + math.degrees(math.atan(slope_2))

                            # if the encountered obstacle isn't actually an obstacle that the waves would diffract from
                            # (e.g.the first encountered obstacle doesn't block the link between the Tx and the second
                            # obstacle), correct the variables
                            cnt_ke = 0
                            cnt2 = 0
                            while cnt_ke < ke_info.shape[0]:
                                cnt2 = cnt2 + 1
                                cnt_ke = cnt_ke + 1
                                if cnt_ke == 1:
                                    slope_1 = (ke_info[cnt_ke-1, 1] - tx_total_height) / ke_info[cnt_ke-1, 0]
                                else:
                                    slope_1 = (ke_info[cnt_ke-1, 1] - ke_info[cnt_ke - 2, 1]) / (ke_info[cnt_ke-1, 0] - ke_info[cnt_ke - 2, 0])

                                if cnt_ke == ke_info.shape[0]:
                                    slope_2 = (ke_info[cnt_ke-1, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[cnt_ke-1, 0])
                                else:
                                    slope_2 = (ke_info[cnt_ke-1, 1] - ke_info[cnt_ke, 1]) / (ke_info[cnt_ke , 0] - ke_info[cnt_ke-1, 0])
                                cnt = cnt_ke
                                idx = 0
                                while math.degrees(math.atan(slope_1)) < -1 * math.degrees(math.atan(slope_2)):
                                    idx = idx + 1
                                    if ke_info.shape[0] == cnt + 1:

                                        if ke_info.shape[0]- 1 - idx == 0:
                                            slope_1 = (ke_info[-1, 1] - tx_total_height) / ke_info[-1, 0]
                                        else:
                                            slope_1 = (ke_info[-1, 1] - ke_info[-2 - idx, 1]) / (ke_info[-1, 0] - ke_info[- 2 - idx, 0])

                                        slope_2 = (ke_info[-1, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[-1, 0])
                                    else:
                                        if cnt2 == 1:
                                            slope_1 = (ke_info[cnt, 1] - tx_total_height) / ke_info[cnt, 0]
                                        else:
                                            slope_1 = (ke_info[cnt, 1] - ke_info[cnt_ke - idx - 1, 1]) / (ke_info[cnt, 0] - ke_info[cnt_ke - idx - 1, 0])

                                        slope_2 = (ke_info[cnt, 1] - ke_info[cnt + 1, 1]) / (
                                                    ke_info[cnt + 1, 0] - ke_info[cnt , 0])

                                    if cnt2 == 1:
                                        h__ = ke_info[cnt, 1] - (ke_info[cnt, 0] * slope + tx_total_height)
                                        d1__ = math.sqrt((ke_info[cnt, 0] * slope) ** 2 + ke_info[cnt, 0] ** 2)
                                        d2__ = math.sqrt(((d_array_nonzero[-1] - ke_info[cnt, 0]) * slope) ** 2 + (d_array_nonzero[-1] - ke_info[cnt , 0]) ** 2)
                                        diff_parameter[b, 0] = h__ * math.sqrt(2 * (d1__ + d2__) / (wavelength * d1__ * d2__))

                                        first_last_diffraction_angles[b, 0] = math.degrees(math.atan(slope_1)) + math.degrees(math.atan(slope_2))
                                        first_last_shadowing_angles[b, 0] = math.degrees(math.atan(slope_1))

                                    cnt = cnt + 1
                                    cnt_ke = cnt_ke + 1
                                    number_knife_edges[b] = number_knife_edges[b] - 1
                                    #if number_knife_edges[b, a] < 0:
                                    #    print(number_knife_edges[b, a])

                            #if first_last_diffraction_angles[b, a, 0] < 0:
                            #    print(first_last_diffraction_angles[b, a, 0])


                            slope_last_ke_to_rx = (ke_info[-1, 1] - rx_total_height) / (d_array_nonzero[-1] - ke_info[-1, 0])
                            slope_ke_beforelast_to_ke_last = (ke_info[-2, 1] - ke_info[-1, 1]) / (ke_info[-1, 0] - ke_info[-2, 0])
                            first_last_diffraction_angles[b, 1] = math.degrees(math.atan(slope_last_ke_to_rx)) - math.degrees(math.atan(slope_ke_beforelast_to_ke_last))
    return a, (LOS, number_obstacles, number_knife_edges, elevation_angles_tx_rx, elevation_angles_NLOS, first_last_shadowing_angles, first_last_diffraction_angles, diff_parameter, diff_param_obstacle_with_max_h_over_r)
    return a, (tmp_res, LOS, number_obstacles, first_last_shadowing_angles, diff_param_obstacle_with_max_h_over_r)


def load_tirem_lib():
    """Loads tirem DLL"""
    global tirem_dll
    return

    try:
        tirem_dll = CDLL("./TIREM320DLL.dll")
    except OSError:
        try:
            tirem_dll = WinDLL("./TIREM320DLL.dll")
        except OSError:
            print('ERROR! Failed to load TIREM DLL')


def call_tirem_loss(d_array, e_array, params):
    """Sets up data for Tirem DLL call."""
    # Load DLL
    if tirem_dll is None:
        load_tirem_lib()

    # initialize the pointer and data for each argument
    # inputs just set to some number in their valid range
    TANTHT = pointer(c_float(params.txheight))  # 0 - 30, 000m
    RANTHT = pointer(c_float(params.rxheight))
    PROPFQ = pointer(c_float(params.freq))  # 1 to 20, 000 MHz

    # next three values characterize the shape of terrain
    NPRFL = pointer(c_int32(d_array.shape[0]))  # number of points in array MAYBE TGUS

    HPRFL = e_array.astype(np.float32).ctypes.data_as(POINTER(c_float))  # array of above (mean) sea level heights
    XPRFL = d_array.astype(np.float32).ctypes.data_as(POINTER(c_float))  # array of great circles distances between
    # points and start

    EXTNSN = pointer(c_int32(params.extsn))  # boolean, 0 is false
    # anything else is true. False = new profile, true = extension of last profile terrain
    # Haven't been able to figure out what the extension flag actually does

    REFRAC = pointer(c_float(params.refrac))  # Surface refractivity  200 to 450.0 "N-Units"
    CONDUC = pointer(c_float(params.conduc))  # 0.00001 to 100.0 S/m
    PERMIT = pointer(c_float(params.permit))  # Relative permittivity of earth surface  1 to 1000
    HUMID = pointer(c_float(params.humid))  # Units g/m^3   0 to 110.0
    polar_ascii = np.array([ord(c) for c in params.polarz])
    POLARZ = polar_ascii.astype(np.uint8).ctypes.data_as(POINTER(c_void_p))

    # output starts here, I just intialize them all to 0
    VRSION = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8).ctypes.data_as(POINTER(c_void_p))
    MODE = np.array([0, 0, 0, 0], dtype=np.uint8).ctypes.data_as(POINTER(c_void_p))
    LOSS = pointer(c_float(0))
    FSPLSS = pointer(c_float(0))

    tirem_dll.CalcTiremLoss(TANTHT, RANTHT, PROPFQ, NPRFL, HPRFL, XPRFL, EXTNSN, REFRAC, CONDUC,
                            PERMIT, HUMID, POLARZ, VRSION, MODE, LOSS, FSPLSS)
    return LOSS.contents.value


def get_tirem_loss(d_array, e_array, params):
    """Returns TIREM loss.

    Usage: loss = get_tirem_loss(d_array, e_array, params)

    inputs: d_array - distance array
            e_array - elevation array
            params - transmitter parameters

    output:   loss - estimated propagation loss
    """
    return call_tirem_loss(d_array, e_array, params)


if __name__ == '__main__':
    main()
