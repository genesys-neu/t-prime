"""Evaluate trained T-PRIME checkpoints on (1) the synthetic validation split and (2) the real X410 OOD set.

Real windows go through exactly the preprocessing used in training (TPrime_dataset.py): rms-normalise the
1024-sample window, cut the transformer observations at the dataset's window offsets, interleave I/Q
(chan2sequence) and split into overlapping slices.
"""
import argparse, json, os, sys, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'TPrime_transformer'))
from preprocessing.TPrime_dataset import TPrimeDataset_Transformer
from model_transformer import TransformerModel

PROTOCOLS = ['5GNR', 'LoRa', 'Nbiot', 'OFDM', 'ssm', 'Zigbee']
CHANS = os.path.join(ROOT, 'data/HCRO_CHANNELS/0937_OFDM')  # local copy of CMAT_ChannelExtraction/0937_OFDM/estimated_chans
SLICE, SEQ, OVERLAP_RATIO = 64, 16, 0.5


def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_val_ds(syn_dir, postfix, channel_aug):
    return TPrimeDataset_Transformer(
        protocols=PROTOCOLS, ds_type='test', file_postfix=postfix, ds_path=syn_dir,
        snr_dbs=['30'], seq_len=SEQ, slice_len=SLICE, slice_overlap_ratio=OVERLAP_RATIO, raw_data_ratio=1.0,
        override_gen_map=False, apply_wchannel=None, transform=chan2sequence, channel_path=CHANS,
        use_channel_aug=channel_aug, use_sota=True, sota_type='awgn', use_cfo=False, max_cfo=2000)


def real_obs(windows, ds):
    """windows: (n, 1024) complex -> (n * n_offsets, SEQ, 2*SLICE) using the dataset's own retrieve_obs."""
    offsets = ds.generate_windows(windows.shape[1])
    out = []
    for w in windows:
        w = ds.rms_normalize(w.astype(np.complex128).reshape(-1, 1))
        out.extend(ds.retrieve_obs(w, {'sample_ix': s}) for s in offsets)
    return np.asarray(out, dtype=np.float32), len(offsets)


@torch.no_grad()
def predict(model, X, dev, bs=1024):
    preds = []
    for i in range(0, len(X), bs):
        preds.append(model(torch.from_numpy(X[i:i + bs]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def eval_val(model, ds, dev, repeats=1):
    ys, ps = [], []
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=8, worker_init_fn=seed_worker)
    for _ in range(repeats):
        for X, y in dl:
            ps.append(model(X.float().to(dev)).argmax(1).cpu().numpy()); ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return ys, ps


def summarize(y, p):
    cm = confusion_matrix(y, p, labels=list(range(len(PROTOCOLS))))
    per = {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(PROTOCOLS)}
    return {'n': int(len(y)), 'acc': float((y == p).mean()), 'balanced_acc': float(np.mean(list(per.values()))),
            'per_class': per, 'confusion': cm.tolist()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['baseline', 'cmat'])
    ap.add_argument('--real_dir', default=os.path.join(ROOT, 'data/HCRO_REAL_OOD'))
    ap.add_argument('--syn_dir', default=os.path.join(ROOT, 'data/HCRO_SYNTH'), help='synthetic dataset (validation split)')
    ap.add_argument('--val_tag', default='', help='run-name prefix of the training runs whose split pickles to use')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results/hcro'))
    ap.add_argument('--snr_bins', nargs='+', type=float, default=[-100, -5, 0, 5, 10, 15, 20, 30, 100])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # validation sets: both use the baseline split pickle's indices (splits are identical: same seed & files)
    val_sets = {'val_awgn': make_val_ds(a.syn_dir, a.val_tag + 'baseline', False),
                'val_cmat': make_val_ds(a.syn_dir, a.val_tag + 'cmat', True)}

    # self-check: our real-data path == dataset path when no augmentation is applied
    chk = make_val_ds(a.syn_dir, a.val_tag + 'baseline', False); chk.use_sota = False; chk.apply_noise = False
    x_ds, _ = chk[0]
    info = chk.ds_info['ds_indexes']['test']['data'][chk.ds_info['ixs_maps']['test'][0]]
    import scipy.io as sio
    raw = sio.loadmat(info['path'])['waveform'][:, 0]
    xr, _ = real_obs(raw[None], chk)
    k = chk.generate_windows(1024).index(info['sample_ix'])
    assert np.allclose(xr[k], x_ds, atol=1e-5), 'real-data preprocessing does not match dataset preprocessing'
    print('preprocessing self-check passed')

    real = {}
    for c in PROTOCOLS + ['hoppedssm']:
        d = np.load(os.path.join(a.real_dir, f'{c}.npz'))
        X, nrep = real_obs(d['x'], chk)
        real[c] = dict(X=X, snr=np.repeat(d['snr_db'], nrep), fname=np.repeat(d['fname'], nrep))
        print(f'real {c}: {len(X)} observations', flush=True)

    results = {}
    for m in a.models:
        ck = torch.load(os.path.join(ROOT, f'checkpoints/hcro/{m}/modelNone_30_sm.pt'), map_location=dev)
        model = TransformerModel(classes=len(PROTOCOLS), d_model=2 * SLICE, seq_len=SEQ, nlayers=6, use_pos=True)
        model.load_state_dict(ck['model_state_dict']); model.to(dev).eval()
        R = {'best_val_loss_during_training': float(ck['loss']), 'best_epoch': ck.get('epoch')}
        R_val = {}
        for vn, ds in val_sets.items():
            y, p = eval_val(model, ds, dev)
            R_val[vn] = (y, p)
            R[vn] = summarize(y, p)
            print(f'[{m}] {vn}: acc {R[vn]["acc"]:.4f}', flush=True)

        y_all = np.concatenate([np.full(len(real[c]['X']), i) for i, c in enumerate(PROTOCOLS)])
        p_all = np.concatenate([predict(model, real[c]['X'], dev) for c in PROTOCOLS])
        s_all = np.concatenate([real[c]['snr'] for c in PROTOCOLS])
        f_all = np.concatenate([real[c]['fname'] for c in PROTOCOLS])
        R['real_all'] = summarize(y_all, p_all)
        R['real_snr_ge_5dB'] = summarize(y_all[s_all >= 5], p_all[s_all >= 5])
        R['real_snr_ge_m5dB'] = summarize(y_all[s_all >= -5], p_all[s_all >= -5])
        R['real_by_snr'] = {}
        for lo, hi in zip(a.snr_bins[:-1], a.snr_bins[1:]):
            mk = (s_all >= lo) & (s_all < hi)
            if mk.sum():
                R['real_by_snr'][f'[{lo:g},{hi:g})'] = {'n': int(mk.sum()), 'acc': float((y_all[mk] == p_all[mk]).mean())}
        R['real_by_file'] = {}
        for fn in np.unique(f_all):
            mk = f_all == fn
            R['real_by_file'][str(fn)] = {'n': int(mk.sum()), 'acc': float((y_all[mk] == p_all[mk]).mean()),
                                          'median_snr_db': float(np.median(s_all[mk]))}
        ph = predict(model, real['hoppedssm']['X'], dev)
        np.savez(os.path.join(a.out, f'preds_{m}.npz'), y=y_all, p=p_all, snr_db=s_all, fname=f_all,
                 hop_p=ph, hop_snr_db=real['hoppedssm']['snr'], hop_fname=real['hoppedssm']['fname'],
                 val_awgn_y=R_val['val_awgn'][0], val_awgn_p=R_val['val_awgn'][1],
                 val_cmat_y=R_val['val_cmat'][0], val_cmat_p=R_val['val_cmat'][1], classes=np.array(PROTOCOLS))
        hm = real['hoppedssm']['snr'] >= 5
        R['hoppedssm_snr_ge_5dB_pred_dist'] = {c: float((ph[hm] == i).mean()) for i, c in enumerate(PROTOCOLS)}
        print(f'[{m}] real: acc {R["real_all"]["acc"]:.4f} | SNR>=5dB acc {R["real_snr_ge_5dB"]["acc"]:.4f} '
              f'bal {R["real_snr_ge_5dB"]["balanced_acc"]:.4f}', flush=True)
        results[m] = R

    json.dump(results, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    print('wrote', os.path.join(a.out, 'results.json'))
