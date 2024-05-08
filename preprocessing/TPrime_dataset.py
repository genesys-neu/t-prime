import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import argparse
from glob import glob
import os
from tqdm import tqdm
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))
import sys
sys.path.append(proj_root_dir)

class FileCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = {}
        self.access_queue = []

    def _evict_oldest(self):
        oldest_file = self.access_queue.pop(0)
        self.cache.pop(oldest_file)

    def get(self, file_path):
        if file_path in self.cache:
            self.access_queue.remove(file_path)
            self.access_queue.append(file_path)
            return self.cache[file_path]
        else:
            return None

    def put(self, file_path, content):
        if len(self.cache) == self.max_size:
            self._evict_oldest()
        if file_path in self.cache:
            self.access_queue.remove(file_path)
        self.cache[file_path] = content
        self.access_queue.append(file_path)


class TPrimeDataset(Dataset):
    def __init__(self,
                 protocols: list,  # this list will contain the folder/protocols names to include in the dataset
                 slice_len: int,    # this will define the slice size of each signal that will be input to the model
                 ds_type='train',  # either 'train' or 'test'
                 slice_overlap_ratio=0.5,   # this is the overlap ratio for each slice generated from a signal
                                            # this value will affect the number of slices that is possible to create from each signal
                 raw_data_ratio=1.0,        # ratio of the whole raw signal dataset to consider for generation
                 test_ratio=0.2,
                 testing_mode='random_sampling',
                 ds_path='../data/DATASET1_1',
                 file_postfix='',
                 noise_model='AWGN', snr_dbs=[30], seed=4389,
                 apply_noise=True, apply_wchannel=None,
                 override_gen_map=False,
                 normalize=False,
                 ota=False, # OVER THE AIR
                 transform=None, target_transform=None,
                 double_class_label=False,
                 out_mode='real'):

        self.protocols = protocols
        self.slice_len = slice_len
        self.slice_overlap_ratio = slice_overlap_ratio
        self.overlap = self.slice_len - int(self.slice_len * self.slice_overlap_ratio)
        self.raw_data_ratio = raw_data_ratio
        self.n_sig_per_class = {} # this will be filled in the generate_ds_map() function
        self.testing_mode = testing_mode
        self.noise_model = noise_model
        self.ota = ota
        self.snr_dbs = snr_dbs
        self.seed = seed
        self.ds_path = ds_path
        self.test_ratio = test_ratio
        self.double_class_label = double_class_label    # this is used for overlapping case (it outputs two labels)
        self.out_mode = out_mode

        if not (self.seed is None):
            np.random.seed(self.seed)

        if ota:
            assert(apply_wchannel is None)

        assert(ds_type in ['train', 'test'])
        self.ds_type = ds_type
        if file_postfix != '' and file_postfix[-1] != '_':
            file_postfix += '__'
        if not ota:
            info_filename = 'ds_info__'+file_postfix+'slice'+str(slice_len)+'_'+str(len(protocols))+testing_mode+'class.pkl'
        else: # over the air
            info_filename = 'ds_info__'+file_postfix+'slice'+str(slice_len)+'_'+str(len(protocols))+testing_mode+'class_ota.pkl'
        ds_info_path = os.path.join(self.ds_path, info_filename)
        do_gen_info = True
        if os.path.exists(ds_info_path) and (not override_gen_map):
            #ans = input('File '+info_filename+' already exists. Do you wanna create a new one? [y/n]')
            #if ans.lower() in ['n', 'no']:
            do_gen_info = False

        if do_gen_info:
            self.ds_info = self.generate_ds_map(ds_path, info_filename, test_ratio)
        else:
            self.ds_info = pickle.load(open(ds_info_path, 'rb'))

        if normalize:
            #Todo
            pass

        # let's assign inputs and labels
        self.transform = transform
        self.target_transform = target_transform
        self.apply_wchannel = apply_wchannel
        self.apply_noise = apply_noise

        self.signal_cache = FileCache(max_size=20e3)

        # let's initialize Matlab engine
        if not (self.apply_wchannel is None):
            self.possible_channels = ['TGn', 'TGax', 'Rayleigh', 'relative', 'random']
            assert(self.apply_wchannel in self.possible_channels)
            self.channel_map = {}
            from preprocessing.matutils import matutils
            self.mateng = matutils.MatlabEngine()  # todo check if we need any custom paths
            # initialize each channel object for each protocol used
            self.chan_models = {}
            for ix, p in enumerate(protocols[0:4]):
                if p == '802_11n':
                    tgn = self.mateng.eng.wlanTGnChannel('SampleRate', float(20e6), 'DelayProfile', 'Model-B', 'LargeScaleFadingEffect', 'Pathloss', 'PathGainsOutputPort', True)
                    self.chan_models[ix] = tgn
                    self.channel_map['TGn'] = ix
                elif p == '802_11ax':
                    tgax = self.mateng.eng.wlanTGaxChannel('SampleRate', float(20e6), 'ChannelBandwidth', 'CBW20', 'DelayProfile', 'Model-B', 'LargeScaleFadingEffect', 'Pathloss', 'PathGainsOutputPort', True)
                    self.chan_models[ix] = tgax
                    self.channel_map['TGax'] = ix
                elif p == '802_11b':
                    rayleighB = self.mateng.eng.comm.RayleighChannel('SampleRate', float(11e6), 'PathDelays', float(1.5e-9), 'AveragePathGains', float(-3), 'PathGainsOutputPort', True)
                    self.chan_models[ix] = rayleighB
                    self.channel_map['RayleighB'] = ix
                elif (p == '802_11b_upsampled') or (p == '802_11g'):
                    rayleigh = self.mateng.eng.comm.RayleighChannel('SampleRate', float(20e6), 'PathDelays', float(1.5e-9), 'AveragePathGains', float(-3), 'PathGainsOutputPort', True)
                    self.chan_models[ix] = rayleigh
                    self.channel_map['Rayleigh'] = ix
    
    def generate_windows(self, len_sig):
        return list(range(0, len_sig-self.slice_len, self.overlap))

    def generate_ds_map(self, ds_path, filename, test_ratio=0.2):
        examples_map = {}
        class_map = dict(zip(self.protocols, range(len(self.protocols))))
        # check if folders are names with _ or .
        if os.path.isdir(os.path.join(ds_path, self.protocols[0])):
            protocols = self.protocols
        elif (os.path.isdir(os.path.join(ds_path, '802.11ax')) or os.path.isdir(os.path.join(ds_path, '802.11ax_0'))): #ds_path[-3:] == 'dBs'): 
            protocols = ['802.11ax', '802.11b', '802.11n', '802.11g']
            if len(self.protocols) == 5:
                protocols.append('noise') 
        else:
            sys.exit('[TPrimeDataset] protocols folder names  not found at ' + ds_path + '. Aborting...')
        # retrieve the list of signals (.mat) from every folder/protocol specified
        for i, p in enumerate(protocols):
            path = os.path.join(ds_path, p)
            if (ds_path.split('_')[-1] == 'power' or ds_path[-3:] == 'dBs'):
                paths = [path + pfix for pfix in ['_0', '_10', '_20', '_30']]
            else:
                paths = [path]
            mat_list_path = []
            num_mat = 0
            for power_path in paths:     
                if os.path.isdir(power_path):
                    mat_list = sorted(glob(os.path.join(power_path, '*.bin'))) if self.ota else sorted(glob(os.path.join(power_path, '*.mat')))
                    self.n_sig_per_class[self.protocols[i]] = int(
                        len(mat_list) * self.raw_data_ratio)  # for each protocol, we save the amount of raw signals to retain

                    mat_list = mat_list[:self.n_sig_per_class[self.protocols[i]]] # then we just clip the list
                    num_mat += len(mat_list)                     # and store the new list value length
                    mat_list_path.extend(mat_list)
                else:
                    if p != 'noise' and not ds_path[-3:] == 'dBs': # not all datasets contain noise 
                        sys.exit('[TPrimeDataset] folder ' + power_path + ' not found. Aborting...')
            examples_map[self.protocols[i]] = dict(
                    zip(
                        list(range(num_mat)),
                        mat_list_path
                    )
                )

        # now let's go through each class examples and assign a global sample index
        # based on the slice len and overlap configuration

        # also, we need to assign a unique slice index to each possible slice to be used with __getitem__ function
        # and assign the corresponding label
        # TRAINING DATA
        data_ixs = {}
        labels_ixs = {}
        ixs_count = 0
        # TEST DATA
        test_data_ixs = {}
        test_labels_ixs = {}
        test_ixs_count = 0
        for c in tqdm(examples_map.keys(), desc='Analyzing signal dataset...'):
            nsamples = len(examples_map[c].items())
            if self.testing_mode == 'future':
                test_threshold = int(nsamples *(1-self.test_ratio))
            else:
                test_threshold = nsamples
            for ix, path in examples_map[c].items():
                if not self.ota: # simulation files
                    sig = sio.loadmat(path)
                    len_sig = sig['waveform'].shape[0]
                else:
                    # read binary files and get length
                    sig = np.fromfile(path, dtype=np.complex128)
                    len_sig = sig.shape[0]
                window_ixs = self.generate_windows(len_sig)
                n_windows = len(window_ixs)
                examples_map[c][ix] = {'sample': window_ixs, # sample is slice for CNN or sequence for Transformer
                                       'path': examples_map[c][ix]}  # here we modify the original content
                if ix <= test_threshold:
                    for w in window_ixs:
                        data_ixs[ixs_count] = {'path': path, 'sample_ix': w}
                        labels_ixs[ixs_count] = class_map[c]
                        ixs_count += 1
                else:
                    for w in window_ixs:
                        test_data_ixs[test_ixs_count] = {'path': path, 'sample_ix': w}
                        test_labels_ixs[test_ixs_count] = class_map[c]
                        test_ixs_count += 1


        # lastly, we separate the training and testing dataset by randomly sampling a certain % of data samples indexes
        if self.testing_mode == 'random_sampling':
            test_ixs = np.random.choice(ixs_count, size=int(ixs_count*test_ratio), replace=False)
            for i in test_ixs.tolist():
                i_data = data_ixs.pop(i)
                test_data_ixs[i] = i_data
                i_label = labels_ixs.pop(i)
                test_labels_ixs[i] = i_label
        # we also need to store a map from linear indexes (used by ray when retrieving data from dataset)
        train_ixs = dict(zip(range(len(data_ixs.keys())), list(data_ixs.keys())))
        test_ixs = dict(zip(range(len(test_data_ixs.keys())), list(test_data_ixs.keys())))


        # after creating class map and examples map, let's store this information for future use
        ds_info_path = os.path.join(ds_path, filename)
        ds_info = {
            'class_map': class_map, 'examples_map': examples_map,
            'ds_indexes':
                {
                    'train': {'data': data_ixs, 'labels': labels_ixs},
                    'test': {'data': test_data_ixs, 'labels': test_labels_ixs}
                },
            'ixs_maps': {'train': train_ixs, 'test': test_ixs}  # linear indexes maps (used by ray/torch)
            }
        pickle.dump(ds_info, open(ds_info_path, 'wb'))

        return ds_info

    def info(self):
        ds_info = {
            'slice_len': self.slice_len,
            'numsamps': {'train' : len(self.ds_info['ds_indexes']['train']['data'].keys()),
                         'test': len(self.ds_info['ds_indexes']['test']['data'].keys())},
            'nclasses': len(self.protocols),
            'nchans': 2,    # real and imag components are separated on different channels
        }
        return ds_info

    def __len__(self):
        return len(self.ds_info['ds_indexes'][self.ds_type]['data'].keys())
    
    def retrieve_obs(self, noisy_sig, obs_info):
        obs = noisy_sig[obs_info['sample_ix']:obs_info['sample_ix']+self.slice_len, 0]
        if self.out_mode == 'complex':
            obs = obs   # return complex version
        elif self.out_mode == 'real_invdim':
            obs = np.stack((obs.real, obs.imag))
            obs = np.swapaxes(obs, 1, 0)        # shape = [N, 2]
        elif self.out_mode == 'real_ampphase':
            angle = np.angle(obs)   # phase in radians
            n_angle = (angle-np.min(angle))/(np.max(angle)-np.min(angle))   # normalize within 0,1 first
            n_angle = 2 * n_angle - 1  # then we convert -1, 1 (from the paper)
            obs = np.stack((np.abs(obs), n_angle))    # compute phase and amplitude
            obs = np.swapaxes(obs, 1, 0)                    # shape = [N, 2]
        else:
            obs = np.stack((obs.real, obs.imag))  # shape = [2, N] with real and imag separated

        if self.transform:
            obs = self.transform(obs)
        return obs

    def __getitem__(self, idx):
        dataset = self.ds_info['ds_indexes'][self.ds_type]
        # first retrieve the internal sample index relative to the linear index idx
        s_idx = self.ds_info['ixs_maps'][self.ds_type][idx]
        # retrieve the info relative to this input sample
        obs_info = dataset['data'][s_idx]
        # let's first retrieve the signal from the cache (or load it in if not present)
        sig_dict = self.signal_cache.get(obs_info['path'])
        self.last_file_loaded = obs_info['path']
        if sig_dict is None:
            if not os.path.exists(dataset['data'][s_idx]['path']):
                # substitute the original path with the new raw source dir
                fullpath, filename = os.path.split(obs_info['path'])
                orig_raw_source, dirname = os.path.split(fullpath)
                # update the entry in the dictionary
                dataset['data'][s_idx]['path'] = os.path.join(self.ds_path, dirname, filename)
                # re-obtain the observation info dict
                obs_info = dataset['data'][s_idx]
            if not self.ota:
                mat_dict = sio.loadmat(obs_info['path'])
                self.signal_cache.put(obs_info['path'], {'np': mat_dict['waveform'], \
                            'mat': self.mateng.py2mat_array(mat_dict['waveform']) if not (self.apply_wchannel is None) else ''})
            else:
                mat_dict = np.fromfile(obs_info['path'], dtype=np.complex128)
                mat_dict = np.expand_dims(mat_dict, axis=1)
                self.signal_cache.put(obs_info['path'], {'np': mat_dict, 'mat': ''}) # WCHANNEL SHOULD ALWAYS BE NONE WITH OTA SAMPLES

            sig_dict = self.signal_cache.get(obs_info['path'])

        label = dataset['labels'][s_idx]
        # apply wireless channel and noise if required
        chan_sig = self.apply_wchan(sig_dict['mat'], label) if not (self.apply_wchannel is None) else sig_dict['np']
        noisy_sig = self.apply_AWGN(chan_sig) if self.apply_noise else chan_sig

        # then, retrieve the relative slice of the requested dataset sample
        obs = self.retrieve_obs(noisy_sig, obs_info)
        if self.target_transform:
            label = self.target_transform(label)
        if self.double_class_label and (type(label) is not list):
            label = [label, label]
        return obs, label

    # Function to change the shape of obs
    # the input is obs with shape (channel, slice)
    def chan2sequence(self, chan):
        seq = np.empty((chan.size))
        seq[0::2] = chan[0]
        seq[1::2] = chan[1]
        return seq

    def apply_AWGN(self, sig):
        # apply AWGN noise based on the levels specified when instantiating the dataset object
        # if more than one is specified, a random level is picked among the one specified
        # first, let's compute signal power (in Watts) as rms(sig)**2
        rms = np.sqrt(np.mean(np.abs(sig) ** 2))
        sig_W = rms ** 2  # power of signal in Watts
        # convert signal power in dBW
        assert(type(self.snr_dbs) is list)
        if self.snr_dbs[0] == 'range':
            SNR = np.random.uniform(-20.0, 30.0)
        else:
            SNR = np.random.choice(self.snr_dbs)
        sig_dbW = 10 * np.log10(sig_W / 1)
        # now compute the relative noise power based on the specified SNR level
        noise_dbW = sig_dbW - float(SNR)
        # noise variance = noise power
        noise_var = 10 ** (noise_dbW / 10)
        # obtain noise standard deviation
        noise_std = np.sqrt(noise_var)
        # now generate the noise samples.
        # NOTE: since we are generating ** complex ** noise samples, the nominal variance for a
        # normal complex distribution is equal to 1/2 instead of 1.
        # https://en.wikipedia.org/wiki/Complex_normal_distribution
        # var = std**2 = 1/2 ---> std = sqrt(1/2) = 1/sqrt(2)
        complex_std = noise_std * 1 / np.sqrt(2)
        noise_samples = np.random.normal(0, complex_std, size=sig.shape) + 1j * np.random.normal(0, complex_std,
                                                                                                 size=sig.shape)
        # now apply noise to the signal
        noisy_sig = sig + noise_samples
        return noisy_sig

    def apply_wchan(self, mat_sig, label):
        if self.apply_wchannel == 'relative':
            # in this case we apply the channel relative to the protocol used
            channel = self.chan_models[label]
        else:
            # in other cases, we have a fixed channel we want to apply to all the signals
            # in this case, 802.11b is not supported in its native sampling rate (11 MHz) due to consistency with
            # sampling rate of other standards (20 MHz)
            assert(not ('802_11b' in self.protocols))
            if self.apply_wchannel == 'random':
                # It is important that the protocols are in the following order 802_11ax, 802_11b_upsampled, 802_11n, 802_11g
                chan_ix = np.random.randint(4)
                if chan_ix == 3: # no channel applied
                    return np.array(mat_sig)
            else:
                chan_ix = self.channel_map[self.apply_wchannel]
            channel = self.chan_models[chan_ix]

        proc_sig = self.mateng.eng.step(channel, mat_sig, nargout=1)
        return np.array(proc_sig)

class TPrimeDataset_Transformer(TPrimeDataset):
    
    def __init__(self, seq_len: int, **kwargs):
        self.seq_len = seq_len
        super().__init__(**kwargs)
        

    def generate_windows(self, len_sig):
        return list(range(0, len_sig-(self.slice_len*self.seq_len), \
                    self.overlap*(self.seq_len-1) + self.slice_len)) # window is now seq_len*slice_len
    
    def info(self):
        ds_info = {
            'seq_len': self.seq_len,
            'slice_len': self.slice_len,
            'numsamps': {'train' : len(self.ds_info['ds_indexes']['train']['data'].keys()),
                         'test': len(self.ds_info['ds_indexes']['test']['data'].keys())},
            'nclasses': len(self.protocols),
            'nchans': 2,    # real and imag components are separated on different channels
        }
        return ds_info

    def retrieve_obs(self, noisy_sig, obs_info):
        obs = noisy_sig[obs_info['sample_ix']:obs_info['sample_ix'] + self.overlap*(self.seq_len-1) + self.slice_len, 0]
        obs = np.stack((obs.real, obs.imag))

        if self.transform:
            obs = self.transform(obs)

        slice_ixs = list(range(0, obs.size-self.slice_len*2+1, self.overlap*2))
        obs = [obs[i:i+self.slice_len*2] for i in slice_ixs]
        return np.asarray(obs)
    

class TPrimeDataset_Transformer_overlap(TPrimeDataset_Transformer):
    def __init__(self, **kwargs):
        self.mixes = ['b-ax', 'b-g', 'b-n', 'g-ax', 'g-n', 'n-ax']
        super().__init__(**kwargs)

    def mix_to_labels(self, classes):
        if classes == 'NOISE':
            return [4, 4]
        # signal belongs to two classes 
        classes = classes.split('-')
        final_labels = []
        if 'ax' in classes:
            final_labels.append(0)
        if 'b' in classes:
            final_labels.append(1)
        if 'n' in classes:
            final_labels.append(2)
        if 'g' in classes:
            final_labels.append(3)
        return final_labels

    def generate_ds_map(self, ds_path, filename, test_ratio=0.2):
        examples_map = {}
        class_map = dict(zip(self.protocols, range(len(self.protocols))))
        directories = self.mixes + ['NOISE']
        # retrieve the list of signals (.mat) from every folder/mix specified
        for i, p in enumerate(directories):
            if p == 'NOISE':
                paths = [os.path.join(ds_path, 'NOISE')]
            else:
                paths = [os.path.join(ds_path, p.replace('-', ''))]
            mat_list_path = []
            num_mat = 0
            for path in paths:
                if os.path.isdir(path):
                    mat_list = sorted(glob(os.path.join(path, '*.bin')))
                    self.n_sig_per_class[directories[i]] = int(
                        len(mat_list) * self.raw_data_ratio)  # for each mix, we save the amount of raw signals to retain

                    mat_list = mat_list[:self.n_sig_per_class[directories[i]]] # then we just clip the list
                    num_mat += len(mat_list)                     # and store the new list value length
                    mat_list_path.extend(mat_list)
                else:
                    if p != 'NOISE':
                        sys.exit('[TPrimeDataset] folder ' + path + ' not found. Aborting...')
            if num_mat != 0:
                examples_map[directories[i]] = dict(
                        zip(
                            list(range(num_mat)),
                            mat_list_path
                        )
                    )

        # now let's go through each class examples and assign a global sample index
        # based on the slice len and overlap configuration

        # also, we need to assign a unique slice index to each possible slice to be used with __getitem__ function
        # and assign the corresponding label
        # TRAINING DATA
        data_ixs = {}
        labels_ixs = {}
        ixs_count = 0
        # TEST DATA
        test_data_ixs = {}
        test_labels_ixs = {}
        test_ixs_count = 0
        for c in tqdm(examples_map.keys(), desc='Analyzing signal dataset...'):
            nsamples = len(examples_map[c].items())
            if self.testing_mode == 'future':
                test_threshold = int(nsamples *(1-self.test_ratio))
            else:
                test_threshold = nsamples
            for ix, path in examples_map[c].items():
                # read binary files and get length
                sig = np.fromfile(path, dtype=np.complex128)
                len_sig = sig.shape[0]
                window_ixs = self.generate_windows(len_sig)
                n_windows = len(window_ixs)
                examples_map[c][ix] = {'sample': window_ixs, # sample is slice for CNN or sequence for Transformer
                                       'path': examples_map[c][ix]}  # here we modify the original content
                if ix <= test_threshold:
                    for w in window_ixs:
                        data_ixs[ixs_count] = {'path': path, 'sample_ix': w}
                        labels_ixs[ixs_count] = self.mix_to_labels(c)
                        ixs_count += 1
                else:
                    for w in window_ixs:
                        test_data_ixs[test_ixs_count] = {'path': path, 'sample_ix': w}
                        test_labels_ixs[test_ixs_count] = self.mix_to_labels(c)
                        test_ixs_count += 1


        # lastly, we separate the training and testing dataset by randomly sampling a certain % of data samples indexes
        if self.testing_mode == 'random_sampling':
            test_ixs = np.random.choice(ixs_count, size=int(ixs_count*test_ratio), replace=False)
            for i in test_ixs.tolist():
                i_data = data_ixs.pop(i)
                test_data_ixs[i] = i_data
                i_label = labels_ixs.pop(i)
                test_labels_ixs[i] = i_label
        # we also need to store a map from linear indexes (used by ray when retrieving data from dataset)
        train_ixs = dict(zip(range(len(data_ixs.keys())), list(data_ixs.keys())))
        test_ixs = dict(zip(range(len(test_data_ixs.keys())), list(test_data_ixs.keys())))


        # after creating class map and examples map, let's store this information for future use
        ds_info_path = os.path.join(ds_path, filename)
        ds_info = {
            'class_map': class_map, 'examples_map': examples_map,
            'overlapping_samples': True,
            'ds_indexes':
                {
                    'train': {'data': data_ixs, 'labels': labels_ixs},
                    'test': {'data': test_data_ixs, 'labels': test_labels_ixs}
                },
            'ixs_maps': {'train': train_ixs, 'test': test_ixs}  # linear indexes maps (used by ray/torch)
            }
        pickle.dump(ds_info, open(ds_info_path, 'wb'))

        return ds_info

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--protocols", nargs='+', default=['802_11ax', '802_11b_upsampled', '802_11n', '802_11g'],
                        choices=['802_11ax', '802_11b', '802_11b_upsampled', '802_11n', '802_11g'],
                        help="Specify the protocols/classes to be included in the dataset")
    parser.add_argument('--raw_path', default='../data/DSTL_DATASET_1_0', help='Path where raw signals are stored.')
    parser.add_argument('--postfix', default='', help='Postfix to append to dataset file.')
    parser.add_argument('--slicelen', default=128, type=int, help='Signal slice size')
    parser.add_argument('--overlap_ratio', default=0.5, help='Overlap ratio for slices generation')
    parser.add_argument('--raw_data_ratio', default=1.0, type=float, help='Specify the ratio of examples per class to consider while training/testing')
    args, _ = parser.parse_known_args()

    myds = TPrimeDataset(protocols=args.protocols, ds_path=args.raw_path, slice_len=args.slicelen, slice_overlap_ratio=args.overlap_ratio, raw_data_ratio=args.raw_data_ratio,
                       apply_wchannel='TGn', file_postfix=args.postfix)    # this case has consistent sampling rates (20 MHz) and applies a specific channel to all signals

    import matplotlib.pyplot as plt
    classes_slice_count = {}
    # compute an histogram that shows the distribution of labels in the dataset
    for k in ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']:
        classes_slice_count[k] = 0
        for key_idx, content in myds.ds_info['examples_map'][k].items():
             classes_slice_count[k] += len(content['slices'])

    plt.bar(list(classes_slice_count.keys()), list(classes_slice_count.values()))
    plt.show()

    for _ in range(10):
        index = np.random.choice(list(myds.ds_info['ixs_maps']['train'].keys()))
        obs, lbl = myds[index]
