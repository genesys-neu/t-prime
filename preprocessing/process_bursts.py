#!/usr/bin/env python3
from pathlib import Path
import re, json
import numpy as np

# ========== CONFIG ==========
RAW_ROOT   = Path("/mnt/dm-3/DroneDetect/DroneDetectV2_byModel")  # where .dat live
OUT_ROOT   = Path("/mnt/dm-3/DroneDetect/processed_bursts")       # where .npy go

FS        = 60e6            # raw sample rate (Hz)
DS_FS     = 20e6            # output sample rate (Hz), ≈ 3x decim
DTYPE     = "float32"       # 'int16' or 'float32' input
ENDIAN    = "little"        # 'little' | 'big'
IQ_ORDER  = "iq"            # 'iq' | 'qi'
SCALE_I16 = True            # scale int16 to [-1,1)

# Burst scan (time spikes via short FFT slices)
SLICE_NFFT   = 2048
SLICE_HOP    = 1024         # 50% overlap
ABS_THR_DB   = -110.0       # fixed PSD threshold (dB/Hz)
DC_CF_TOL_HZ = 1.0e5        # |CF| < 100 kHz considered near-DC
MIN_BW_NEAR_DC_HZ = 2.0e5   # reject near-DC if BW < 200 kHz
MIN_LEN_SLICES = 2          # need >=2 consecutive “on” slices
GAP_SLICES     = 1          # allow brief dropouts inside a burst

# Filtering after centering
LP_MARGIN  = 1.2            # LP cutoff ≈ 0.5*bw*LP_MARGIN
FILT_ORDER = 6              # IIR order (if SciPy available)

# Splitting policy
MAX_RAW_PER_BURST   = 112_320                                   # ~1.872 ms @60MS/s
MAX_OUT_PER_PART    = int(round(MAX_RAW_PER_BURST*(DS_FS/FS)))  # ≈ 37,440 @20MS/s

# Streaming read
READ_CHUNK_IQ = 112_320

# Try SciPy (optional)
try:
    from scipy.signal import butter, sosfiltfilt, resample_poly
    from scipy.io import savemat
    SCIPY_AVAILABLE = True
    SAVEMAT_AVAILABLE = True
except Exception:
    butter = sosfiltfilt = resample_poly = None
    savemat = None
    SCIPY_AVAILABLE = False
    SAVEMAT_AVAILABLE = False

# ========== IO & DSP HELPERS ==========
def complex_iq_reader(path: Path, dtype: str, endianness: str, chunk_iq: int,
                      iq_order: str = "iq", scale: bool = True):
    if dtype not in ("int16","float32"): raise ValueError("dtype must be 'int16' or 'float32'")
    if endianness not in ("little","big"): raise ValueError("endianness must be 'little' or 'big'")
    if iq_order not in ("iq","qi"): raise ValueError("iq_order must be 'iq' or 'qi'")
    dmap = {("int16","little"):"<i2", ("int16","big"):">i2", ("float32","little"):"<f4", ("float32","big"):">f4"}
    dt = np.dtype(dmap[(dtype, endianness)])
    bps = dt.itemsize
    scalars_per_chunk = chunk_iq*2
    with path.open("rb") as f:
        while True:
            buf = f.read(scalars_per_chunk*bps)
            if not buf: break
            arr = np.frombuffer(buf, dtype=dt)
            if arr.size < 2: break
            if arr.size % 2: arr = arr[:-1]
            I = arr[0::2] if iq_order=="iq" else arr[1::2]
            Q = arr[1::2] if iq_order=="iq" else arr[0::2]
            z = I.astype(np.float32) + 1j*Q.astype(np.float32)
            if dtype == "int16" and scale: z /= 32768.0
            yield z.astype(np.complex64, copy=False)

def rational_resample_factors(fs_in: float, fs_out: float, max_den: int = 1000):
    from fractions import Fraction
    if abs((fs_in/fs_out) - round(fs_in/fs_out)) < 1e-9:
        return 1, int(round(fs_in/fs_out))
    frac = Fraction(fs_out/fs_in).limit_denominator(max_den)
    return frac.numerator, frac.denominator

def design_fir_lowpass(cutoff_norm: float, numtaps: int = 129):
    cutoff_norm = float(max(min(cutoff_norm, 0.499), 1e-4))
    M = numtaps - 1
    n = np.arange(numtaps, dtype=np.float64)
    h = 2.0*cutoff_norm*np.sinc(2.0*cutoff_norm*(n - M/2))
    w = np.hamming(numtaps); h *= w; h /= np.sum(h)
    return h.astype(np.float32)

def filtfilt_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    y = np.convolve(x, h, mode="same")
    y2 = np.convolve(y[::-1], h, mode="same")[::-1]
    return y2.astype(np.complex64)

def lowpass_dc(x: np.ndarray, fs: float, half_bw_hz: float, iir_order: int):
    norm = min(max(half_bw_hz/(fs*0.5), 1e-4), 0.99)
    if SCIPY_AVAILABLE:
        sos = butter(N=iir_order, Wn=norm, btype="low", output="sos")
        return sosfiltfilt(sos, x).astype(np.complex64)
    taps = 129 if iir_order <= 6 else 255
    return filtfilt_fir(x, design_fir_lowpass(norm*0.98, numtaps=taps))

def resample_poly_fallback(x: np.ndarray, fs_in: float, fs_out: float):
    up, down = rational_resample_factors(fs_in, fs_out)
    if up != 1:
        raise RuntimeError("Install SciPy for non-integer resampling; fallback supports pure decimation only.")
    cutoff = 0.45 / down
    taps = 127 if down <= 8 else 255
    y = np.convolve(x, design_fir_lowpass(cutoff, numtaps=taps), mode="same")
    return y[::down].astype(np.complex64)

def mix_frequency(x: np.ndarray, fs: float, f_shift: float, start_idx: int):
    n = np.arange(x.size, dtype=np.float64) + float(start_idx)
    phase = np.exp(1j*2.0*np.pi*f_shift*(n/fs))
    return (x * phase.astype(np.complex64)).astype(np.complex64)

def slice_psd_detect(sig: np.ndarray, fs: float, threshold_db: float = ABS_THR_DB, nfft: int = SLICE_NFFT):
    x = np.asarray(sig)
    w = np.hanning(nfft).astype(np.float32)
    X = np.fft.fft(w * x[:nfft], n=nfft)
    P = (np.abs(X)**2) / (np.sum(w**2) * fs)
    f = np.fft.fftfreq(nfft, d=1/fs)
    P = np.fft.fftshift(P); f = np.fft.fftshift(f)
    P_db = 10.0*np.log10(P + 1e-20)
    valid = (P_db >= threshold_db)
    if not np.any(valid):
        return False, 0.0, 0.0
    k0 = int(np.argmax(P_db))
    # contiguous region above threshold around peak → BW
    k_lo = k0
    while k_lo-1 >= 0 and valid[k_lo-1]: k_lo -= 1
    k_hi = k0
    L = len(f)
    while k_hi+1 < L and valid[k_hi+1]: k_hi += 1
    f_lo, f_pk, f_hi = float(f[k_lo]), float(f[k0]), float(f[k_hi])
    cf_hz = 0.5*(f_lo + f_hi)
    bw_hz = max(0.0, f_hi - f_lo)
    return True, cf_hz, bw_hz

# ========== FILENAME PARSER → (model, noise, op) ==========
# Expected like: CLEAN__AIR_FY__AIR_0010_00.dat
NAME_RE = re.compile(r'^(?P<noise>[A-Z]+)__([A-Z]+)_(?P<op>[A-Z]+)__(?P<model>[A-Z]+)_')

def parse_tokens(dat_path: Path):
    base = dat_path.name
    m = NAME_RE.match(base)
    if m:
        return m.group("model"), m.group("noise"), m.group("op")
    # Fallback heuristic
    parts = base.split("__")
    noise = parts[0] if parts else "UNKNOWN"
    op = "UNK"; model = "UNK"
    if len(parts) >= 2:
        mid = parts[1]              # e.g., AIR_FY
        toks = mid.split("_")
        if len(toks) >= 2:
            model, op = toks[0], toks[1]
    if len(parts) >= 3 and model == "UNK":
        tail = parts[2]             # e.g., AIR_0010_00.dat
        model = tail.split("_")[0]
    return model, noise, op

# ========== PER-FILE PROCESSOR (single pass, variable-length bursts) ==========
def process_file_variable_bursts(path: Path, out_root: Path):
    assert path.exists(), f"Not found: {path}"
    model, noise, op = parse_tokens(path)
    out_dir = out_root / model / noise / op
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    rdr = complex_iq_reader(path, DTYPE, ENDIAN, READ_CHUNK_IQ, IQ_ORDER, scale=(DTYPE=="int16"))

    buf = np.zeros(0, dtype=np.complex64)
    file_pos = 0

    active = False
    start_abs = None
    last_ok_end_abs = None
    consec_ok = 0
    gap = 0
    cf_hold = 0.0
    bw_hold = 0.0
    cf_alpha = 0.2

    stem = path.stem
    parts_written = 0
    burst_idx = 0

    def save_burst_segment(seg_raw: np.ndarray, seg_start_abs: int, cf_hz: float, bw_hz: float):
        nonlocal parts_written, burst_idx
        # center → LP → downsample
        mixed = mix_frequency(seg_raw, FS, f_shift=-cf_hz, start_idx=seg_start_abs)
        half_bw = max(0.5*bw_hz*LP_MARGIN, 5e3)
        base = lowpass_dc(mixed, FS, half_bw, iir_order=FILT_ORDER)

        if SCIPY_AVAILABLE:
            up, down = rational_resample_factors(FS, DS_FS)
            y = resample_poly(base, up, down).astype(np.complex64)
        else:
            y = resample_poly_fallback(base, FS, DS_FS)

        # All bursts are smaller than frame length in your data; save each burst as a
        # single file (no splitting into parts). Prefer MATLAB .mat when available.
        if y.size <= 0:
            return

        fname_mat = f"{stem}_burst{burst_idx:04d}_cf{cf_hz/1e6:.3f}MHz_bw{bw_hz/1e6:.3f}MHz_{y.size}S.mat"
        if SAVEMAT_AVAILABLE:
            try:
                savemat(str(out_dir / fname_mat), {"waveform": y.reshape(-1, 1)})
                saved_name = fname_mat
            except Exception:
                saved_name = fname_mat.replace('.mat', '.npy')
                np.save(out_dir / saved_name, y)
        else:
            saved_name = fname_mat.replace('.mat', '.npy')
            np.save(out_dir / saved_name, y)

        meta = {
            "file": saved_name, "input": str(path),
            "model": model, "noise": noise, "op": op,
            "burst_index": burst_idx,
            "raw_start_idx": int(seg_start_abs),
            "raw_end_idx": int(seg_start_abs + seg_raw.size),
            "fs_in": float(FS), "fs_out": float(DS_FS),
            "cf_hz": float(cf_hz), "bw_hz": float(bw_hz),
            "nsamp_out": int(y.size)
        }
        with open(summary_path, "a") as f:
            f.write(json.dumps(meta) + "\n")
        parts_written += 1

    def finish_burst():
        nonlocal active, start_abs, last_ok_end_abs, consec_ok, gap, cf_hold, bw_hold, burst_idx
        if (active and start_abs is not None and last_ok_end_abs is not None
                and consec_ok >= MIN_LEN_SLICES and last_ok_end_abs > start_abs):
            # try to slice from current buffer; else tiny re-read
            s, e = start_abs, last_ok_end_abs
            start_rel = s - file_pos
            end_rel   = e - file_pos
            if 0 <= start_rel < end_rel <= buf.size:
                seg = buf[start_rel:end_rel].copy()
            else:
                seg = np.zeros(e - s, dtype=np.complex64)
                filled = 0; acc = 0
                for z2 in complex_iq_reader(path, DTYPE, ENDIAN, max(8*SLICE_NFFT, 65536), IQ_ORDER, scale=(DTYPE=="int16")):
                    zlen = z2.size
                    if acc + zlen <= s:
                        acc += zlen; continue
                    take_from = max(0, s - acc)
                    take_to   = min(zlen, e - acc)
                    if take_to > take_from:
                        ncopy = take_to - take_from
                        seg[filled:filled+ncopy] = z2[take_from:take_to]
                        filled += ncopy
                        if filled >= seg.size: break
                    acc += zlen
            save_burst_segment(seg, s, cf_hold, bw_hold)
            burst_idx += 1

        # reset
        active = False
        start_abs = None
        last_ok_end_abs = None
        consec_ok = 0
        gap = 0
        cf_hold = 0.0
        bw_hold = 0.0

    # streaming scan
    for z in rdr:
        buf = z if buf.size == 0 else np.concatenate((buf, z), axis=0)

        i = 0
        while i + SLICE_NFFT <= buf.size:
            sl = buf[i:i+SLICE_NFFT]
            ok, cf_hz, bw_hz = slice_psd_detect(sl, FS, threshold_db=ABS_THR_DB, nfft=SLICE_NFFT)
            # ignore near-DC skinny junk
            if ok and (abs(cf_hz) < DC_CF_TOL_HZ and bw_hz < MIN_BW_NEAR_DC_HZ):
                ok = False

            slice_start_abs = file_pos + i
            slice_end_abs   = file_pos + i + SLICE_NFFT

            if ok:
                gap = 0
                consec_ok += 1
                last_ok_end_abs = slice_end_abs
                if not active:
                    active = True
                    start_abs = slice_start_abs
                    cf_hold = cf_hz
                    bw_hold = bw_hz
                else:
                    cf_hold = 0.8*cf_hold + 0.2*cf_hz
                    bw_hold = max(bw_hold, bw_hz)
            else:
                if active:
                    gap += 1
                    if gap > GAP_SLICES:
                        finish_burst()

            i += SLICE_HOP

        # keep overlap
        keep_from = max(0, buf.size - (SLICE_NFFT - SLICE_HOP))
        file_pos += (buf.size - keep_from)
        buf = buf[keep_from:]

    finish_burst()
    return parts_written, out_dir

# ========== DRIVER ==========
def main():
    dat_files = sorted(RAW_ROOT.glob("*/*.dat"))
    if not dat_files:
        print(f"No .dat files found under {RAW_ROOT}")
        return
    print(f"Found {len(dat_files)} .dat files")
    total_parts = 0
    for k, f in enumerate(dat_files, 1):
        try:
            parts_written, out_dir = process_file_variable_bursts(f, OUT_ROOT)
            total_parts += parts_written
            print(f"[{k}/{len(dat_files)}] {f.name} → {parts_written} parts → {out_dir}")
        except Exception as e:
            print(f"[{k}/{len(dat_files)}] ERROR {f}: {e}")
    print(f"Done. Total parts written: {total_parts}")

if __name__ == "__main__":
    main()
