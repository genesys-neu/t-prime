import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

def rms(X):
    if np.iscomplexobj(X):
        return np.sqrt(np.mean(X.real * X.real + X.imag * X.imag))
    else:
        return (np.sqrt(np.mean(X * X))) # root-mean-square

if __name__ == "__main__":
    ds_path = "../data/DATASET3_0/OTA_dataset/"
    exp_folder = "802.11g_z"
    noise_folder = "noise"
    file_list = glob(os.path.join(os.path.join(ds_path, exp_folder), '*.bin'))

    noise_list = glob(os.path.join(os.path.join(ds_path, noise_folder), '*.bin'))

    data = []
    for f in file_list:
        data.append(np.fromfile(f, dtype=np.complex128))

    n_data = []
    for n in noise_list:
        n_data.append(np.fromfile(n, dtype=np.complex128))

    manual_mag_thresh = 0.01

    min_zeros = 10

    for filename, long_sig in list(zip(file_list, data)):  # for each signal in the folder
        print("Working on ", filename)
        plt.clf()
        plt.plot(np.abs(long_sig))
        plt.show()

        run_val, run_st, run_len = find_runs((np.abs(
            long_sig) > manual_mag_thresh))  # find 1s (True) and 0s (False) runs (i.e. signals and noise portions in the capture)
        pauses_ixs = []
        vsl_list = list(zip(run_val, run_st, run_len))
        for v, s, l in vsl_list:
            if v == False and l > min_zeros:
                pauses_ixs.append((v, s, l))  # collect the "runs" info about the silent moments

        # init two lists, one to store each signal and the other for its relative pause after it
        signals = []
        noise_parts = []
        start_ix = 0
        for _, p_ix, p_l in pauses_ixs:
            if len(long_sig[start_ix:p_ix]) > 0:
                signals.append(long_sig[start_ix:p_ix])
                noise_parts.append(long_sig[p_ix:p_ix + p_l])
                start_ix = p_ix + p_l
            else:
                continue

        # if there are still transmissions, let's include the last transmissions in the list (without any pause after it)
        if start_ix != long_sig.shape[0] - 1:
            signals.append(long_sig[start_ix:])
            start_ix = long_sig.shape[0]

        # compute individual powers in Watts (TODO: some of them won't be accurate because we don't have the whole signal, so we should discard those)
        pwrs_sig = []
        pwrs_noise = []
        for s in signals:
            pwrs_sig.append(rms(s) ** 2)
        for s in noise_parts:
            pwrs_noise.append(rms(s) ** 2)

        count = 0
        for ps, pn in list(zip(pwrs_sig, pwrs_noise)):
            ps_dBW = 10 * np.log10(ps)
            pn_dBW = 10 * np.log10(pn)
            #print("SNR (dB)", ps_dBW - pn_dBW, "check", 10 * np.log10(ps / pn))
            print("SNR (dB)", ps_dBW - pn_dBW, "Sig. Pow (dBW)", ps_dBW, "Noise Pow (dBW)", pn_dBW)
            sig_norm = signals[count] / np.sqrt(ps)
            sig_norm_dBW = 10 * np.log10( rms(sig_norm) ** 2 )
            print("Normalized Sig. Pow (dBW)", sig_norm_dBW)
            plt.clf()
            plt.plot(np.abs(sig_norm))
            plt.plot(np.abs(signals[count]))
            plt.show()
            count += 1

        print("-------------------------")
