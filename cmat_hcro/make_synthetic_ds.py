"""Build a T-PRIME-format synthetic dataset from the clean transmitted waveforms.

Each class waveform (complex64, baseband) is treated as cyclic (it was looped on TX). Waveforms not stored at
the 30.72 MS/s receive rate (see SRC_RATE_HZ) are first resampled to it.
We draw random 1024-sample crops, reject near-silent crops (gaps between bursts), and save each crop
as <out>/<class>/<i>.mat with key 'waveform' of shape (1024, 1), complex128 -- the format
preprocessing/TPrime_dataset.py expects.
"""
import argparse, os
import numpy as np
import scipy.io as sio
from fractions import Fraction
from scipy.signal import resample_poly

CLASS_FILES = {  # class folder name -> source waveform
    '5GNR': '5GNR_peaknorm.bin', 'LoRa': 'LoRa_peaknorm.bin', 'Nbiot': 'NBIoT_peaknorm.bin',
    'OFDM': 'OFDM_peaknorm.bin', 'ssm': 'SSM_peaknorm.bin', 'Zigbee': 'Zigbee_peaknorm.bin',
}
WIN = 1024
FS = 30.72e6
SRC_RATE_HZ = {'ssm': 1e6}  # SSM_peaknorm.bin is 1 MS/s (16 samples/symbol); others are already at FS


def resample_cyclic(x, fs_in, fs_out=FS):
    """Resample one period of a looped waveform, avoiding filter edge effects at the wrap point."""
    r = Fraction(fs_out / fs_in).limit_denominator(1000)
    y = resample_poly(np.concatenate([x, x, x]), r.numerator, r.denominator)
    n = len(y) // 3
    return y[n:2 * n]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/mnt/ssd/normalized_waveforms')
    ap.add_argument('--out', default='/home/sat3737/t-prime-ext/data/HCRO_SYNTH')
    ap.add_argument('--n_per_class', type=int, default=10000)
    ap.add_argument('--min_rel_power', type=float, default=0.1, help='reject crops below this x median crop power')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--classes', nargs='+', default=list(CLASS_FILES), help='classes to (re)generate')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for cls in a.classes:
        x = np.fromfile(os.path.join(a.src, CLASS_FILES[cls]), dtype=np.complex64).astype(np.complex128)
        if cls in SRC_RATE_HZ:
            x = resample_cyclic(x, SRC_RATE_HZ[cls])
        N = len(x)
        xx = np.concatenate([x, x[:WIN]])
        c = np.concatenate([[0.0], np.cumsum(np.abs(xx) ** 2)])
        p = (c[WIN:WIN + N] - c[:N]) / WIN            # power of crop starting at each index
        valid = np.flatnonzero(p >= a.min_rel_power * np.median(p))
        starts = rng.choice(valid, size=a.n_per_class, replace=len(valid) < a.n_per_class)
        os.makedirs(os.path.join(a.out, cls), exist_ok=True)
        for i, s in enumerate(starts):
            sio.savemat(os.path.join(a.out, cls, f'{i:05d}.mat'), {'waveform': xx[s:s + WIN].reshape(WIN, 1)})
        print(f'{cls}: {a.n_per_class} crops from {N} samples, {len(valid)}/{N} valid start offsets', flush=True)
