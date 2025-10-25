class Params:
    """Represents TIREM Input Parameters"""
    #In this clss, we will declare all the input parameters of TIREM

    def __init__(self, bs_endpoint_name, bs_is_tx, txheight, rxheight, bs_lon, bs_lat, bs_x, bs_y,
                 freq, polarz, generate_features, map_type, map_filedir, gain, first_call, extsn,
                 refrac, conduc, permit, humid, side_len, sampling_interval):

        """
        :param bs_endpoint_name: endpoint (basestation) name #that means, give a name for the basestation
        :param bs_is_tx: Is the basestation a transmitter (1) or a receiver (0) #wheather the bs is a transmitter or receiver
        :param txheight: transmitter height #height of transmitter
        :param rxheight: receiver height #height of receiver
        :param bs_lon: longitude of the base station #longitude of the base station
        :param bs_lat: latitude of the base station #latitude of the base station
        :param bs_x: x coordinate of the base station in the grid #x coordinate of the base station in the grid
        :param bs_y: y coordinate of the base station in the grid #y coordinate of the base station in the grid
        :param freq: transmit frequency in MHz #transmission frequency in MHz
        :param polarz: 'H' = horizontal polarization of Tx antenna and 'V' = vertical polarization
        :param generate_features: boolean, Generate only TIREM predictions (0) / also generate features (1) #whether to generate features or predictions
        :param map_type: Fusion map ("fusion") or lidar DSM map ("lidar"). The fusion map struct has to have the fields
        "dem" for digital elevation map and "hybrid_bldg" for building heights. The lidar map has to have the field
        "data" having combined information of elevations and building heights.
        :param map_filedir: map file directory including the filename and extension #File directory and its extension
        :param gain: antenna gain in dB #antenna gain
        :param first_call:boolean, 1 loads the TIREM library for people using it for the first time
        :param extsn: boolean, 0 is false; anything else is true. False = new profile, true = extension of last profile
        terrain
        :param refrac: surface refractivity, range: (200 to 450.0) "N-Units"
        :param conduc: conductivity, range: (0.00001 to 100.0) S/m
        :param permit: relative permittivity of earth surface, range: (1 to 1000)
        :param humid: humidity, units g/m^3, range: (0 to 110.0)
        :param side_len: grid_cell side length (m)
        :param sampling_interval: sampling interval along the tx-rx link. e.g. 0.5 means a given grid cell is sampled
        twice. Side length of 0.5 and sampling interval of 0.5 mean that the Tx-Rx horizontal array step size
        0.5*0.5 = 0.25 m.
        """

        self.bs_endpoint_name = bs_endpoint_name
        self.bs_is_tx = bs_is_tx
        self.txheight = txheight
        self.rxheight = rxheight
        self.bs_lon = bs_lon
        self.bs_lat = bs_lat
        self.bs_x = bs_x
        self.bs_y = bs_y
        self.freq = freq
        self.polarz = polarz
        self.generate_features = generate_features
        self.map_type = map_type
        self.map_filedir = map_filedir
        self.gain = gain
        self.first_call = first_call
        self.extsn = extsn
        self.refrac = refrac
        self.conduc = conduc
        self.permit = permit
        self.humid = humid
        self.side_len = side_len
        self.sampling_interval = sampling_interval

    def __str__(self):
        return f'BS ENDPOINT NAME: {self.bs_endpoint_name}\nBS_IS_TX: {self.bs_is_tx}\nTX HEIGHT: {self.txheight}\n' \
               f'RX HEIGHT: {self.rxheight}\nBS LONGITUDE: {self.bs_lon}\nBS LATITUDE: {self.bs_lat}\nBS X: {self.bs_x}\n'\
               f'BS Y: {self.bs_y}\nFREQUENCY: {self.freq}\nPOLARIZATION: {self.freq}\nGENERATE FEATURES: ' \
               f'{self.generate_features}\nMAP TYPE: {self.map_type}\nMAP FILEDIR: {self.map_filedir}\nGAIN: {self.gain}\n' \
               f'FIRST CALL: {self.first_call}\nEXTENSION: {self.extsn}\nREFRACTIVITY: {self.refrac}\nCONDUCTIVITY:' \
               f'{self.conduc}\nPERMITTIVITY: {self.permit}\nHUMIDITY: {self.humid}\nSIDE LENGTH: {self.side_len}\n' \
               f'SAMPLING INTERVAL: {self.sampling_interval}'


    def __unicode__(self):
        s = f'BS ENDPOINT NAME: {self.bs_endpoint_name}\nBS_IS_TX: {self.bs_is_tx}\nTX HEIGHT: {self.txheight}\n' \
               f'RX HEIGHT: {self.rxheight}\nBS LONGITUDE: {self.bs_lon}\nBS LATITUDE: {self.bs_lat}\nBS X: {self.bs_x}\n'\
               f'BS Y: {self.bs_y}\nFREQUENCY: {self.freq}\nPOLARIZATION: {self.freq}\nGENERATE FEATURES: ' \
               f'{self.generate_features}\nMAP TYPE: {self.map_type}\nMAP FILEDIR: {self.map_filedir}\nGAIN: {self.gain}\n' \
               f'FIRST CALL: {self.first_call}\nEXTENSION: {self.extsn}\nREFRACTIVITY: {self.refrac}\nCONDUCTIVITY:' \
               f'{self.conduc}\nPERMITTIVITY: {self.permit}\nHUMIDITY: {self.humid}\nSIDE LENGTH: {self.side_len}\n' \
               f'SAMPLING INTERVAL: {self.sampling_interval}'
        return s.encode('utf-8')
