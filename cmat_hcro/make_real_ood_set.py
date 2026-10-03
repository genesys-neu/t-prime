"""Sample 1024-sample windows from the cleaned X410 captures to build the real-world (OOD) test set.

Windows are drawn uniformly at random from each file (budget split evenly across a class's files).
For each window we store a wideband (full 30.72 MHz) SNR estimate against the receiver noise floor.
The floor is the 1st-percentile window power, minimised over each parent capture's files; the receiver
floor is consistent across captures, so we use the median of the per-capture estimates (some captures,
e.g. 0845, never drop to the floor).
Output: <out>/<class>.npz with x (n,1024) complex64, fname, start, snr_db.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

CLASSES = ['5GNR', 'LoRa', 'Nbiot', 'OFDM', 'ssm', 'Zigbee', 'hoppedssm']
WIN = 1024
EXCLUDE = {
    # the capture the CMAT channel estimates were extracted from
    '6.3.s5.0937.OFDM2min.OFDM.17.290-115.402.bin',
    # mostly hopping smart-meter interference 20-35 dB above the Zigbee signal (Zigbee dominates only in gaps)
    '6.2.s2.CUTx.1340.Zigbee.154.200-176.367.bin',
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/ssd/x410_recordings/CleanedData')
    ap.add_argument('--out', default='/home/sat3737/t-prime-ext/data/HCRO_REAL_OOD')
    ap.add_argument('--n_per_class', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    files = {c: sorted(e['file_name'] for e in json.load(open(f'{a.root}/{c}/metadata.json'))
                       if e['file_name'] not in EXCLUDE) for c in CLASSES}
    # pass 1: noise floor per parent capture
    p1 = {}
    for c, fl in files.items():
        for fn in fl:
            m = np.memmap(f'{a.root}/{c}/{fn}', dtype=np.complex64, mode='r')
            st = rng.integers(0, len(m) - WIN, 2000)
            p1[fn] = np.percentile([np.mean(np.abs(m[s:s + WIN]) ** 2) for s in st], 1)
    parent = {e['file_name']: e['parent_file_name'] for c in CLASSES
              for e in json.load(open(f'{a.root}/{c}/metadata.json'))}
    floors = defaultdict(lambda: np.inf)
    for fn, v in p1.items():
        floors[parent[fn]] = min(floors[parent[fn]], v)
    for k, v in sorted(floors.items()):
        print(f'noise floor {k:32s} {10*np.log10(v):7.2f} dB')
    floor = np.median(list(floors.values()))
    print(f'using global noise floor {10*np.log10(floor):.2f} dB')

    # pass 2: sample windows
    for c, fl in files.items():
        per_file = int(np.ceil(a.n_per_class / len(fl)))
        X, F, S, SNR = [], [], [], []
        for fn in fl:
            m = np.memmap(f'{a.root}/{c}/{fn}', dtype=np.complex64, mode='r')
            st = np.sort(rng.integers(0, len(m) - WIN, per_file))
            w = np.stack([np.asarray(m[s:s + WIN]) for s in st])
            p = np.mean(np.abs(w) ** 2, axis=1)
            snr = 10 * np.log10(np.maximum(p / floor - 1, 1e-3))
            X.append(w); F += [fn] * len(st); S.append(st); SNR.append(snr)
        X = np.concatenate(X); S = np.concatenate(S); SNR = np.concatenate(SNR)
        np.savez(f'{a.out}/{c}.npz', x=X, fname=np.array(F), start=S, snr_db=SNR)
        print(f'{c:10s} {len(fl):2d} files  {len(X)} windows  wideband SNR pct10/50/90 = '
              f'{np.percentile(SNR,10):6.1f} {np.percentile(SNR,50):6.1f} {np.percentile(SNR,90):6.1f}  '
              f'frac>=5dB {np.mean(SNR>=5):.2f}', flush=True)
