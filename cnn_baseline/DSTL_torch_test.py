import numpy as np
import torch
from dstl.preprocessing.DSTL_dataset import DSTLDataset
from model_cnn1d import Baseline_CNN1D



Nclass = 4
num_channels = 2
num_feats = 1
slice_len = 128
snr_dBs = [30]

PATH = '/home/mauro/Research/DSTL/dstl/cnn_baseline/results/SNR30/model.best.pt'

protocols = ['802_11ax', '802_11b', '802_11n', '802_11g']
ds_test = DSTLDataset(protocols, ds_type='test', snr_dbs=snr_dBs, slice_len=slice_len, slice_overlap_ratio=0.5, override_gen_map=True)

model = Baseline_CNN1D(classes=Nclass, numChannels=num_channels, slice_len=slice_len)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(model.device.type) # reload the model on the appropriate device
model.eval()    # set the evaluation mode

# obtain a random sample from the test dataset. This will be needed in order to populate the cache initially
rand_sixs = np.random.choice(len(ds_test))
_, true_label = ds_test[rand_sixs]
# now let's retrieve the whole signal from the signal cache
signal_path = list(ds_test.signal_cache.cache.keys())[0]
# let's obtain the whole signal and run inference on each slice
sig = ds_test.signal_cache.cache[signal_path]
# apply AWGN
noisy_sig = ds_test.apply_AWGN(sig)
len_sig = noisy_sig.shape[0]
window_ixs = list(range(0, len_sig-ds_test.slice_len, ds_test.overlap))

for w in window_ixs:
    complex_slice = noisy_sig[w : w + ds_test.slice_len, 0]
    slice = np.stack((complex_slice.real, complex_slice.imag))
    slice = slice[np.newaxis, :, :]
    slice_t = torch.Tensor(slice)
    slice_t = slice_t.to(model.device.type)
    out = model(slice_t.float())
    _, predicted = torch.max(out, 1)