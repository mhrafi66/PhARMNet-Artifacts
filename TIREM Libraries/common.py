"""Common Functions for Mobility Tool."""

import numpy as np

#@jit(nopython=True)
def build_arrays(side_len, sampling_interval, tx, rx, elev_map):
    """ Returns distance and elevation arrays for use in TIREM

    Notes:
    # side_len           :  physical distance of a tile edge
    # sampling_interval  :  in units of tile edges
    # tx, rx             :  location of antennas as [col;row]
    # elev_map                :  the elevation map

    # Lower left tile is (1,1)
    """
    # base_row = 596
    # base_col = 874
    # tx = (base_col, base_row)
    #sampling_interval = 1
    nrows, ncols = elev_map.shape
    dir_ = (rx - tx).astype(np.float64)
    dist = np.linalg.norm(dir_, ord=2)  # Note, this in units of grid cell side_length
    unit_dir = dir_ / dist
    num_samples = int(np.rint(dist / sampling_interval)) + 1
    d_array = np.linspace(0.0, dist * side_len, num_samples)  # Didn't do any great-circle corrections yet
    loc = np.rint(tx[:,None] + unit_dir[:,None] * sampling_interval * np.arange(num_samples)).astype(np.int32)
    loc[0, loc[0] >= ncols] = ncols
    loc[1, loc[1] >= nrows] = nrows
    loc[0, loc[0] < 1] = 1
    loc[1, loc[1] < 1] = 1
    e_array = elev_map[loc[1]-1, loc[0]-1]
    return d_array, e_array


def get_utm_letter(lat):
    """ References the UTM letter designator for each latitude """
    ch = 'XWVUTSRQPNMLKJHGFEDC'[::-1]
    idx = int(np.floor((lat + 80) / 8)) + 1
    size = lat.flatten().shape[0]
    zone_letter = np.reshape(np.array(['Z' for _ in range(size)]), lat.shape)
    valid = 0 < idx <= len(ch) and not np.isnan(idx)
    if valid:
        zone_letter = ch[idx]
    return zone_letter


def wgs84_to_utm(long, lat, zone):
    """ function utm_easting, utm_northing, zone_number, zone_letter = wgs84_to_utm(long, lat, zone)

    Usage: utm_easting, utm_northing, zone_number, zone_letter = wgs84_to_utm(long, lat)

    zone > 0  all eastings projected into specified zone, may be out of range
    zone < 0  all eastings projected into minimum zone, output will be positive
    zone == 0 (default) returns correct zone number for all values

    Converts WGS84 lat/lon to UTM coordinates using code adapted from:
        Chuck Gantz- chuck.gantz@globalstar.com
        Equations from USGS Bulletin 1532

    R. Martin, 11/2001

    08/08/02  forced all output to be minimum zone unless 'zone' is specified
    09/11/02  zone>0, zone<0, zone==0 (default)
    """

    # CHECK THIS
    if long is None:
        return np.array([]), np.array([]), np.array([]), np.array([])

    force_zone = [] if zone is None else 0

    # constants for WGS84
    a = 6378137  # equatorial radius
    ecc_squared = 0.00669438  # eccentricity squared
    k0 = 0.9996
    pi_fourth = np.pi / 4
    deg2rad = np.pi / 180
    rad2deg = 180.0 / np.pi

    # long_origin
    # ecc_prime_squared
    # N, T, C, A, M

    # Make sure the longitude is between - 180.00 and 179.9
    long_temp = long  # (Long + 180) - fix((Long + 180) / 360) * 360 - 180;
    lat_rad = lat * deg2rad
    long_rad = long_temp * deg2rad

    # compute the UTM Zone from the latitude and longitude
    zone_number = np.floor((long_temp + 180) / 6) + 1
    if 30 < zone_number < 38:
        # compute special zones
        sz32 = (56.0 <= lat < 64.0 and 3.0 <= long_temp < 12.0)
        sz = (72.0 <= lat < 84.0)
        sz31 = (0.0 <= long_temp < 9.0)
        sz33 = (9.0 <= long_temp < 21.0)
        sz35 = (21.0 <= long_temp < 33.0)
        sz37 = (33.0 <= long_temp < 42.0)

        zone_number[sz32] = 32
        zone_number[sz and sz31] = 31
        zone_number[sz and sz33] = 33
        zone_number[sz and sz35] = 35
        zone_number[sz and sz37] = 37

    # force all zone numbers to be identical
    # zone_number = zone_number * 0 + np.floor(np.median(zone_number));
    if force_zone > 0:
        zone_number = np.ones(long_temp.shape) * force_zone
    elif force_zone < 0:
        use_zone = min(zone_number(zone_number > 0))
        zone_number = np.ones(long_temp.shape) * use_zone

    # projection origin
    long_origin = (zone_number - 1) * 6 - 180 + 3  # +3 puts origin in middle of zone
    long_origin_rad = long_origin * deg2rad
    zone_letter = get_utm_letter(lat)

    # here it is ...
    ecc_prime_squared = ecc_squared / (1 - ecc_squared)
    N = a / np.sqrt(1 - ecc_squared * (np.sin(lat_rad) ** 2))
    T = np.tan(lat_rad) * np.tan(lat_rad)
    C = ecc_prime_squared * np.cos(lat_rad) ** 2
    A = np.cos(lat_rad) * (long_rad - long_origin_rad)
    M = a * ((1 - ecc_squared / 4 - 3 * ecc_squared * ecc_squared / 64 - 5 * ecc_squared * ecc_squared * ecc_squared /
              256) * lat_rad - (3 * ecc_squared / 8 + 3 * ecc_squared * ecc_squared / 32 + 45 * ecc_squared *
                                ecc_squared * ecc_squared / 1024) * np.sin(2 * lat_rad) + (15 * ecc_squared *
                                                                                           ecc_squared / 256 + 45 *
                                                                                           ecc_squared * ecc_squared *
                                                                                           ecc_squared / 1024) *
             np.sin(4 * lat_rad) - (35 * ecc_squared * ecc_squared * ecc_squared / 3072) * np.sin(6 * lat_rad))

    utm_easting = (k0 * N * (A + (1 - T + C) * A * A * A / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ecc_prime_squared) *
                             A * A * A * A * A / 120) + 500000.0)

    utm_northing = (k0 * (M + N * np.tan(lat_rad) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A * A * A * A / 24 +
                                                     (61 - 58 * T + T * T + 600 * C - 330 * ecc_prime_squared) * A * A
                                                     * A * A * A * A / 720)))

    # 10000000 meter offset for southern hemisphere
    if lat < 0:
        utm_northing += 10000000.0

    # round to nearest cm
    utm_easting = np.rint(utm_easting * 100) / 100
    utm_northing = np.rint(utm_northing * 100) / 100

    return utm_easting, utm_northing, zone_number, zone_letter


def utm_to_wgs84(utm_easting, utm_northing, zone_number, zone_letter):
    """ Function converts UTM coordinates to WGS84 Lat/Lon

    Usage: long, lat = utm_to_wgs84(utm_easting, utm_northing, zone_number, zone_letter)

    Adapted from: Chuck Gantz- chuck.gantz@globalstar.com, Equations from USGS Bulletin 1532

    R. Martin, 11/2001
    """

    if utm_easting == '':
        return np.array([]), np.array([])

    # Default is northern hemisphere
    zone_letter = 'N' if zone_letter == '' else zone_letter
    # if ~exist('ZoneLetter', 'var')
    #    zone_letter = 'N'

    # Constants
    a = 6378137  # equatorial radius
    ecc_squared = 0.00669438  # eccentricity Squared
    k0 = 0.9996
    e1 = (1 - np.sqrt(1 - ecc_squared)) / (1 + np.sqrt(1 - ecc_squared))

    pi_fourth = np.pi / 4
    deg2rad = np.pi / 180
    rad2deg = 180.0 / np.pi

    x = utm_easting - 500000.0  # remove 500,000 meter offset for longitude
    y = utm_northing

    # check for values in southern hemisphere and remove false northing
    if zone_letter[1] - 'N' >= 0:
        northern_hemisphere = 1
    else:
        northern_hemisphere = 0
        y = y - 10000000.0

    # here is the equation CHECK THIS MAKE SURE MULTIPLICATION WORKS
    long_origin = (zone_number - 1) * 6 - 180 + 3  # +3 puts origin in middle of zone
    ecc_prime_squared = ecc_squared / (1 - ecc_squared)
    m = y / k0
    mu = m / (a * (
            1 - ecc_squared / 4 - 3 * ecc_squared * ecc_squared / 64 - 5 * ecc_squared * ecc_squared * ecc_squared / 256))
    phi1Rad = mu + (3 * e1 / 2 - 27 * e1 * e1 * e1 / 32) * np.sin(2 * mu) + (
            21 * e1 * e1 / 16 - 55 * e1 * e1 * e1 * e1 / 32) * np.sin(4 * mu) + \
              (151 * e1 * e1 * e1 / 96) * np.sin(6 * mu)
    phi1 = phi1Rad * rad2deg
    n1 = a / np.sqrt(1 - ecc_squared * np.sin(phi1Rad) ** 2)
    t1 = np.tan(phi1Rad) ** 2
    c1 = ecc_prime_squared * np.cos(phi1Rad) ** 2
    r1 = a * (1 - ecc_squared) / (1 - ecc_squared * np.sin(phi1Rad) ** 2) ** 1.5
    d = x / (n1 * k0)
    lat = phi1Rad - (n1 * np.tan(phi1Rad) / r1) * (
            d * d / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * ecc_prime_squared) * d * d * d * d / 24 +
            (
                    61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * ecc_prime_squared - 3 * c1 * c1) * d * d * d * d * d * d / 720)
    lat = lat * rad2deg
    long = (d - (1 + 2 * t1 + c1) * d * d * d / 6 + (
            5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * ecc_prime_squared + 24 * t1 * t1) * d * d * d * d * d / 120) / np.cos(
        phi1Rad)
    long = long_origin + long * rad2deg
    lat[np.isnan(long)] = np.NaN


def lon_lat_to_grid_xy(lon, lat, img, column_map):
    """Function to calculate the local column and row coordinates for lon and lat on a given utm image.

    Usage: x, y, idx = lon_lat_to_grid_xy(lon, lat, img)

    inputs: lon - longitude in decimal degrees (WGS84)
            lat - latitude in decimal degrees
            img - image specification structure from create_geoimage
                ncols: 205
                nrows: 230
                xllcorner: 595410
                yllcorner: 4127730
                     axis: [595410 601560 4127730 4134630]
             NODATA_value: NaN
                 cellsize: 30
               utmZoneNum: 10
               utmZoneLtr: 'S'
            column_map - column names to column index within img

    output:   x - pixel column address on given image
              y - pixel row address on given image
              indx - pixel subscript address eg: sub2ind([nrows ncols],y x)
    indx will be NaN for all x and y not on the image

R. Martin 11/2001
    """
    # CHECK THIS
    # my_img = Path('img')
    # if not my_img.is_file():
    #     img = np.array([])

    # Image must be matrix
    # if isinstance(img, list):
    #     img = np.array(img)
    # if not isinstance(img, (pd.Series, np.ndarray)):
    #     print('ERROR! lon_lat_to_grid_xy: must provide coordinate system reference')
    #     return np.array([]), np.array([]), np.array([])

    # Project image coordinates
    # projection = img.projection if img.hasattr('projection') else 'UTM'
    projection = img[column_map['projection']] if 'projection' in column_map else 'UTM'
    if projection == 'UTM':
        # ue, un, zn, zl = wgs84_to_utm(lon, lat, img.utmZoneNum)
        ue, un, zn, zl = wgs84_to_utm(lon, lat, img[column_map['utmZoneNum']])
    elif projection == 'OSGB36':
        pass
        # ue, un = wgs84_to_osgb36(lon, lat)
    else:
        print('ERROR! lon_lat_to_grid_xy: unrecognized projection')
        return np.array([]), np.array([]), np.array([])

    cell_size = img[column_map['cellsize']]
    nrows = img[column_map['nrows']]
    ncols = img[column_map['ncols']]

    if 'xllcorner' in column_map and 'yllcorner' in column_map:
        xOrigin = img[column_map['xllcorner']] - cell_size / 2
        yOrigin = img[column_map['yllcorner']] - cell_size / 2
    elif 'xllcenter' in column_map and 'yllcenter' in column_map:
        xOrigin = img[column_map['xllcenter']] - cell_size
        yOrigin = img[column_map['yllcenter']] - cell_size
    else:
        xOrigin = 0
        yOrigin = 0

    xLocal = ue - xOrigin
    yLocal = un - yOrigin

    x = int(np.rint(xLocal / cell_size))
    y = int(np.rint(yLocal / cell_size))

    # CHECK THIS Convert to subscript
    # is_good = 1 <= x <= ncols and 1 <= y <= nrows
    # idx = np.zeros((len(is_good), 1)) * np.NaN
    # if np.any(is_good):
    #     idx[is_good] = sub2ind([nrows, ncols], y[is_good], x[is_good])
    idx = 0  # NOT SURE

    return x, y, idx
