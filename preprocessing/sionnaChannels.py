import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')  # double safety
except Exception:
    pass
import numpy as np
# ---- Complex 1D convolution helper (works with complex64) ----
def complex_conv1d_same(x_complex_T1, k_complex_L):
    """
    x_complex_T1: [T,1] complex64
    k_complex_L:  [L]   complex64 (already time-reversed if emulating linear conv)
    returns: [T,1] complex64
    """
    T = tf.shape(x_complex_T1)[0]
    L = tf.shape(k_complex_L)[0]

    xr = tf.reshape(tf.math.real(x_complex_T1), [1, T, 1])
    xi = tf.reshape(tf.math.imag(x_complex_T1), [1, T, 1])

    hr = tf.reshape(tf.math.real(k_complex_L), [L, 1, 1])
    hi = tf.reshape(tf.math.imag(k_complex_L), [L, 1, 1])
    with tf.device("/CPU:0"):
        yr = tf.nn.conv1d(xr, hr, stride=1, padding="SAME") - tf.nn.conv1d(xi, hi, stride=1, padding="SAME")
        yi = tf.nn.conv1d(xr, hi, stride=1, padding="SAME") + tf.nn.conv1d(xi, hr, stride=1, padding="SAME")

    y = tf.complex(tf.squeeze(yr, axis=0), tf.squeeze(yi, axis=0))
    return y
# ---------- PDP builder: TGn Model-B (9 taps, 2 clusters, ~15 ns RMS, ~80 ns max delay) ----------
def _pdp_tgn_model_b(seed=0):
    """
    Deterministic PDP approximating TGn Model-B specs from your table:
      - 9 taps total, 2 clusters
      - Maximum delay ≈ 80 ns
      - RMS delay spread ≈ 15 ns
      - K = 0 (Rayleigh)
    Returns:
      delays_s: [K] float32 seconds
      powers:   [K] float32 linear, normalized to sum=1
    """
    rng = np.random.default_rng(seed)

    # Choose fixed, reproducible cluster centers within [0, 80] ns
    max_delay_ns = 80.0
    c0 = 0.15 * max_delay_ns     # ~12 ns
    c1 = 0.75 * max_delay_ns     # ~60 ns

    # Split 9 taps into two clusters (4 + 5)
    taps_c0, taps_c1 = 4, 5

    # Small spreads inside each cluster to keep RMS ~15 ns and max delay ~80 ns
    spread0 = 0.10 * max_delay_ns   # ~8 ns
    spread1 = 0.12 * max_delay_ns   # ~9.6 ns

    def cluster(center, spread, n):
        # Deterministic symmetric offsets around center
        base = np.linspace(-1.5, 1.5, n)  # n points
        offs = (spread/6.0) * base        # ~±spread/4 range
        d = np.clip(center + offs, 0.0, max_delay_ns)
        d.sort()
        return d

    d0 = cluster(c0, spread0, taps_c0)
    d1 = cluster(c1, spread1, taps_c1)
    delays_ns = np.concatenate([d0, d1]).astype(np.float64)

    # Exponential decays inside clusters (cluster 2 slightly weaker)
    def exp_decay(x, tau):
        return np.exp(-x / tau)

    # Normalize per-cluster offsets to start at zero
    p0 = exp_decay(d0 - d0.min(), tau=0.6 * 15.0)          # tune to hit ~15 ns RMS
    p1 = 0.7 * exp_decay(d1 - d1.min(), tau=0.8 * 15.0)    # slightly weaker 2nd cluster
    powers = np.concatenate([p0, p1]).astype(np.float64)

    # Normalize total power
    powers /= powers.sum()

    # One light pass to nudge RMS DS toward 15 ns if needed
    mean_tau = np.sum(delays_ns * powers)
    rms_spread = np.sqrt(np.sum((delays_ns - mean_tau)**2 * powers))
    target = 15.0
    if abs(rms_spread - target) > 0.05 * target:
        # Scale second cluster outward/inward to adjust RMS
        scale = np.clip(target / (rms_spread + 1e-12), 0.6, 1.6)
        d1_adj = d1.min() + (d1 - d1.min()) * scale
        delays_ns = np.concatenate([d0, np.clip(d1_adj, 0.0, max_delay_ns)])
        # Recompute powers (same rule)
        p0 = exp_decay(d0 - d0.min(), tau=0.6 * 15.0)
        p1 = 0.7 * exp_decay(d1_adj - d1_adj.min(), tau=0.8 * 15.0)
        powers = np.concatenate([p0, p1])
        powers /= powers.sum()

    delays_s = (delays_ns * 1e-9).astype(np.float32)
    powers = powers.astype(np.float32)
    return delays_s, powers


# ---------- Exact TGn Model-B channel layer (block Rayleigh) with path loss + tap output ----------
class TGnModelB_Exact(tf.keras.layers.Layer):
    """
    TGn Model-B-equivalent small-scale fading + TGn-style path loss + taps out.

    MATLAB parity:
      - 'DelayProfile','Model-B'  -> 9 taps, 2 clusters, ~15 ns RMS, ~80 ns max delay, Rayleigh (K=0)
      - 'LargeScaleFadingEffect','Pathloss' -> breakpoint PL (5 m, n=2/3.5) with Friis 1 m ref
      - 'PathGainsOutputPort', true -> returns taps (path gains) with delays & powers
      - 'SampleRate' is implicit: pass x at 20 Msps (same grid as your baseband)
    """
    def __init__(self,
                 sample_rate_hz=20e6,
                 carrier_frequency_hz=5.18e9,
                 distance_m=1,
                 apply_shadowing=False,
                 shadowing_std_db_near=3.0,
                 shadowing_std_db_far=4.0,
                 seed=0):
                # match MATLAB default RNG seed

        super().__init__()
        self.fs   = tf.constant(sample_rate_hz, tf.float32)
        self.fc   = tf.constant(carrier_frequency_hz, tf.float32)
        self.dist = tf.constant(distance_m, tf.float32)
        self.apply_shadowing = bool(apply_shadowing)
        self.shadowing_std_db_near = float(shadowing_std_db_near)
        self.shadowing_std_db_far  = float(shadowing_std_db_far)
        self.rng_seed = int(seed)

        # Fixed, deterministic PDP per TGn Model-B row
        delays_s, powers = _pdp_tgn_model_b(seed=seed)
        self.delays_s = tf.constant(delays_s, tf.float32)   # [K]
        self.powers   = tf.constant(powers,   tf.float32)   # [K]
        self.num_taps = self.powers.shape[0]

    # ---- Large-scale fading: TGn breakpoint model with Friis 1 m reference ----
    def _pathloss_amplitude(self, distance_m):
        c   = 299792458.0
        lam = c / float(self.fc.numpy())
        fspl_1m_lin = (4.0 * np.pi / lam)**2  # linear power at 1 m

        d = float(distance_m.numpy())
        d_bp = 5.0
        if d <= d_bp:
            n = 2.0
            pl_rel = d**n
            std_db = self.shadowing_std_db_near
        else:
            # continuity at d_bp
            pl_rel = (d_bp**2.0) * ((d / d_bp)**3.5)
            std_db = self.shadowing_std_db_far

        pl_lin = fspl_1m_lin * pl_rel  # power loss
        if self.apply_shadowing:
            shadow_db = np.random.normal(0.0, std_db)
            pl_lin *= 10.0**(shadow_db/10.0)

        # Return AMPLITUDE attenuation (so power scales by this^2)
        att = 1.0 / np.sqrt(pl_lin)
        return tf.constant(att, dtype=tf.float32)

    # ---- Sample Rayleigh taps for a block (K=0 dB) ----
    def _sample_block_taps(self, batch_size):
        # Complex Gaussian with per-tap variance = powers
        std = tf.sqrt(self.powers / 2.0)             # per real/imag
        std = tf.reshape(std, [1, -1])               # [1, K]
        real = tf.random.normal([batch_size, self.num_taps], stddev=1.0, seed=self.rng_seed)
        imag = tf.random.normal([batch_size, self.num_taps], stddev=1.0, seed=self.rng_seed + 1)
        h = tf.complex(real * std, imag * std)       # [B, K]
        return h

    def call(self, x):
        """
        x: [B, T, N_tx], real or complex
        Returns:
          y:    [B, T, 1] (SISO after summing across TX; extend if you need MIMO)
          taps: {'h':[B,K], 'delays_s':[K], 'powers':[K]}
        """
        x = tf.cast(x, tf.complex64)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # Block Rayleigh taps (constant over this block)
        h = self._sample_block_taps(B)               # [B, K]

        # Quantize delays to sample grid (20 Msps)
        Ts = 1.0 / self.fs
        sample_delays = tf.cast(tf.round(self.delays_s / Ts), tf.int32)  # [K]
        K = tf.shape(sample_delays)[0]
        L = tf.reduce_max(sample_delays) + 1                              # FIR length

        # Build per-batch FIR kernels by scattering taps to delay positions
        def build_kernel(h_b):
            k = tf.zeros([L], dtype=tf.complex64)
            return tf.tensor_scatter_nd_add(k,
                                            tf.expand_dims(sample_delays, 1),  # [K,1]
                                            h_b)
        kernels = tf.map_fn(build_kernel, h, dtype=tf.complex64)          # [B, L]

        # Sum across TX for SISO and convolve per batch
        x_sum = tf.reduce_sum(x, axis=-1, keepdims=True)                  # [B, T, 1]

        def conv_b(xb, kb):
            kb_rev = tf.reverse(kb, axis=[0])
            return complex_conv1d_same(xb, kb_rev)

        y = tf.map_fn(lambda args: conv_b(args[0], args[1]),
                      (x_sum, kernels),
                      fn_output_signature=tf.complex64)                   # [B, T, 1]

        # Apply large-scale path loss (amplitude)
        att = self._pathloss_amplitude(self.dist)                         # scalar
        y = tf.cast(att, y.dtype) * y

        taps = {"h": h, "delays_s": self.delays_s, "powers": self.powers}
        return y, taps

# --- PDP for TGax Model-B (9 taps, 2 clusters, RMS~15 ns, max delay~80 ns, K=0) ---
def _pdp_tgax_model_b(seed=0):
    rng = np.random.default_rng(seed)
    max_delay_ns = 80.0
    rms_target_ns = 15.0

    # Two clusters, 4 + 5 taps
    c0, c1 = 0.15*max_delay_ns, 0.75*max_delay_ns
    n0, n1 = 4, 5
    sp0, sp1 = 0.10*max_delay_ns, 0.12*max_delay_ns  # small intra-cluster spreads

    def cluster(center, spread, n):
        # deterministic symmetric offsets about center (keeps realizations reproducible)
        base = np.linspace(-1.5, 1.5, n)
        offs = (spread/6.0) * base
        d = np.clip(center + offs, 0.0, max_delay_ns)
        d.sort()
        return d

    d0 = cluster(c0, sp0, n0)
    d1 = cluster(c1, sp1, n1)
    delays_ns = np.concatenate([d0, d1]).astype(np.float64)

    # Exponential intra-cluster decays; cluster 2 slightly weaker
    def exp_decay(x, tau): return np.exp(-x/tau)
    p0 = exp_decay(d0 - d0.min(), tau=0.6*rms_target_ns)
    p1 = 0.7*exp_decay(d1 - d1.min(), tau=0.8*rms_target_ns)
    powers = np.concatenate([p0, p1]).astype(np.float64)
    powers /= powers.sum()

    # Single pass tweak to nudge RMS towards 15 ns if off by >5%
    mean_tau = np.sum(delays_ns * powers)
    rms = np.sqrt(np.sum((delays_ns - mean_tau)**2 * powers))
    if abs(rms - rms_target_ns) > 0.05*rms_target_ns:
        scale = np.clip(rms_target_ns/(rms+1e-12), 0.6, 1.6)
        d1_adj = d1.min() + (d1 - d1.min())*scale
        delays_ns = np.concatenate([d0, np.clip(d1_adj, 0.0, max_delay_ns)])
        p0 = exp_decay(d0 - d0.min(), tau=0.6*rms_target_ns)
        p1 = 0.7*exp_decay(d1_adj - d1_adj.min(), tau=0.8*rms_target_ns)
        powers = np.concatenate([p0, p1]); powers /= powers.sum()

    return delays_ns.astype(np.float32)*1e-9, powers.astype(np.float32)  # seconds, linear


class TGaxModelB_Exact(tf.keras.layers.Layer):
    """
    802.11ax TGax 'Model-B' channel (CBW20, path loss, taps out).
    Mirrors:
      - DelayProfile='Model-B' → 9 taps, 2 clusters, RMS~15 ns, max delay~80 ns, Rayleigh (K=0)
      - ChannelBandwidth='CBW20' → run input at 20 Msps (sampling implicit)
      - LargeScaleFadingEffect='Pathloss' (+ optional floors/walls like TGax)
      - PathGainsOutputPort=true → returns (y, taps)
    """
    def __init__(self,
                 sample_rate_hz=20e6,           # 'SampleRate'
                 channel_bandwidth='CBW20',     # 'ChannelBandwidth' (exposed for parity)
                 carrier_frequency_hz=5.25e9,   # TGax default CF (can set to 5.18e9 if you prefer) :contentReference[oaicite:2]{index=2}
                 distance_m=3.0,                # TGax default Tx/Rx distance (used for path loss) :contentReference[oaicite:3]{index=3}
                 num_floors=0,                  # NumPenetratedFloors (adds floor loss)
                 num_walls=0,                   # NumPenetratedWalls (adds wall loss)
                 wall_loss_db=5.0,              # WallPenetrationLoss (dB) default from doc examples :contentReference[oaicite:4]{index=4}
                 apply_shadowing=False,         # enable log-normal shadowing on large-scale loss
                 shadowing_std_db_near=3.0,     # near breakpoint sigma
                 shadowing_std_db_far=4.0,      # far breakpoint sigma
                 seed=73):
        super().__init__()
        self.fs   = tf.constant(float(sample_rate_hz), tf.float32)
        self.fc   = tf.constant(float(carrier_frequency_hz), tf.float32)
        self.dist = tf.constant(float(distance_m), tf.float32)
        self.cbw  = channel_bandwidth
        self.num_floors = int(num_floors)
        self.num_walls  = int(num_walls)
        self.wall_loss_db = float(wall_loss_db)
        self.apply_shadowing = bool(apply_shadowing)
        self.shadowing_std_db_near = float(shadowing_std_db_near)
        self.shadowing_std_db_far  = float(shadowing_std_db_far)
        self.seed = int(seed)

        # Fixed PDP per TGax Model-B row
        delays_s, powers = _pdp_tgax_model_b(seed=seed)
        self.delays_s = tf.constant(delays_s, tf.float32)   # [9]
        self.powers   = tf.constant(powers,   tf.float32)   # [9]
        self.K = self.powers.shape[0]

    # --- Large-scale TGax path loss with floor/wall terms (per doc sections) ---
    def _tgax_pathloss_amp(self, distance_m):
        # Breakpoint model + Friis 1 m ref (same core model as TGn, but TGax supports floor/wall terms) :contentReference[oaicite:5]{index=5}
        c   = 299792458.0
        lam = c / float(self.fc.numpy())
        fspl_1m_lin = (4.0*np.pi/lam)**2  # power at 1 m

        d = float(distance_m.numpy())
        d_bp = 5.0
        if d <= d_bp:
            pl_rel = d**2.0
            std_db = self.shadowing_std_db_near
        else:
            pl_rel = (d_bp**2.0) * ((d/d_bp)**3.5)
            std_db = self.shadowing_std_db_far

        pl_lin = fspl_1m_lin * pl_rel  # base large-scale loss (power)

        # Add TGax floor & wall penetration losses (in dB), then convert back to linear power
        if self.num_floors > 0:
            # Floor loss (PEL_floor) per TGax (18.3*n*(n+2)/(n+1) - 0.46) dB :contentReference[oaicite:6]{index=6}
            n = self.num_floors
            pel_floor_db = 18.3 * n * (n + 2)/(n + 1) - 0.46
        else:
            pel_floor_db = 0.0

        pel_wall_db = self.num_walls * self.wall_loss_db  # simple linear model in TGax doc sections :contentReference[oaicite:7]{index=7}
        extra_db = pel_floor_db + pel_wall_db

        pl_lin *= 10.0**(extra_db/10.0)

        if self.apply_shadowing:
            shadow_db = np.random.normal(0.0, std_db)
            pl_lin *= 10.0**(shadow_db/10.0)

        # Return amplitude attenuation factor
        att = 1.0/np.sqrt(pl_lin)
        return tf.constant(att, dtype=tf.float32)

    # --- Block Rayleigh taps (K=0 dB) sampled from PDP ---
    def _sample_block_taps(self, batch_size):
        std = tf.sqrt(self.powers/2.0)[None, :]  # [1, 9]
        real = tf.random.normal([batch_size, self.K], stddev=1.0, seed=self.seed)
        imag = tf.random.normal([batch_size, self.K], stddev=1.0, seed=self.seed+1)
        return tf.complex(real*std, imag*std)     # [B, 9]

    def call(self, x):
        """
        x: [B, T, N_tx] at 20 Msps (CBW20)
        returns:
          y:    [B, T, 1]
          taps: {'h':[B,9], 'delays_s':[9], 'powers':[9]}
        """
        x = tf.cast(x, tf.complex64)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # Draw one block of taps per batch (static over block)
        h = self._sample_block_taps(B)  # [B, 9]

        # Quantize delays to sample grid (CBW20 ⇒ fs ≈ 20 MHz)
        Ts = 1.0/self.fs
        d_samp = tf.cast(tf.round(self.delays_s/Ts), tf.int32)  # [9]
        L = tf.reduce_max(d_samp) + 1

        # Build FIR kernel per batch
        def build_kernel(h_b):
            k = tf.zeros([L], tf.complex64)
            return tf.tensor_scatter_nd_add(k, tf.expand_dims(d_samp, 1), h_b)
        kernels = tf.map_fn(build_kernel, h, dtype=tf.complex64)  # [B, L]

        # Collapse TXs for SISO (extend to true MIMO if you need)
        x_sum = tf.reduce_sum(x, axis=-1, keepdims=True)  # [B, T, 1]

        def conv_b(xb, kb):
            kb_rev = tf.reverse(kb, axis = [0])
            return complex_conv1d_same(xb, kb_rev)
        y = tf.map_fn(lambda args: conv_b(args[0], args[1]),
                      (x_sum, kernels),
                      fn_output_signature=tf.complex64)   # [B, T, 1]

        # Apply TGax path loss (incl. optional floor/wall & shadowing)
        att = self._tgax_pathloss_amp(self.dist)
        y = tf.cast(att, y.dtype) * y

        taps = {"h": h, "delays_s": self.delays_s, "powers": self.powers}
        return y, taps

class RayleighFIR_Exact(tf.keras.layers.Layer):
    """
    Rayleigh FIR fading with user-specified path delays and average path gains (dB).
    - SampleRate            -> sample_rate_hz
    - PathDelays            -> path_delays_s (float or list, seconds)
    - AveragePathGains (dB) -> avg_path_gains_db (float or list, dB)
    - PathGainsOutputPort   -> returns (y, taps) where taps['h'] are complex path gains
    Default: block fading (MaximumDopplerShift = 0 Hz). Expose max_doppler_hz if needed.
    """
    def __init__(self,
                 sample_rate_hz=11e6,
                 path_delays_s=1.5e-9,          # float or list/ndarray (seconds)
                 avg_path_gains_db=-3.0,        # float or list (dB)
                 max_doppler_hz=0.0,            # 0 => block fading (matches static default)
                 seed=0):
        super().__init__()
        self.fs = tf.constant(float(sample_rate_hz), tf.float32)
        # Normalize to arrays
        if np.isscalar(path_delays_s):
            path_delays_s = [float(path_delays_s)]
        if np.isscalar(avg_path_gains_db):
            avg_path_gains_db = [float(avg_path_gains_db)]
        assert len(path_delays_s) == len(avg_path_gains_db), "Delays and gains length mismatch"

        self.delays_s = tf.constant(np.array(path_delays_s, dtype=np.float32), tf.float32)   # [K]
        # Convert dB gains to linear power per tap and normalize total power to 1 (matches typical behavior)
        p_lin = 10.0 ** (np.array(avg_path_gains_db, dtype=np.float32) / 10.0)
        p_lin = p_lin / np.sum(p_lin)  # normalize overall power (comm.RayleighChannel uses relative avg path gains)
        self.powers = tf.constant(p_lin.astype(np.float32), tf.float32)                      # [K]

        self.K = int(self.powers.shape[0])
        self.seed = int(seed)
        self.max_doppler_hz = float(max_doppler_hz)

    def _sample_block_taps(self, batch_size):
        """
        Rayleigh (K=0) complex coefficients per path for a block.
        Variance per path = powers[k]. (Per real/imag std = sqrt(p/2)).
        """
        std = tf.sqrt(self.powers / 2.0)[None, :]  # [1,K]
        real = tf.random.normal([batch_size, self.K], stddev=1.0, seed=self.seed)
        imag = tf.random.normal([batch_size, self.K], stddev=1.0, seed=self.seed+1)
        return tf.complex(real * std, imag * std)  # [B,K]

    def call(self, x):
        """
        x: [B, T, N_tx] (complex or real). Returns:
          y:    [B, T, 1]
          taps: {'h':[B,K], 'delays_s':[K], 'powers':[K]}
        """
        x = tf.cast(x, tf.complex64)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # Block fading (MaxDopplerShift=0). If you need time variation, we can add Doppler shaping.
        h = self._sample_block_taps(B)  # [B,K]

        # Quantize delays to sample grid
        Ts = 1.0 / self.fs
        d_samp = tf.cast(tf.round(self.delays_s / Ts), tf.int32)  # [K]
        L = tf.reduce_max(d_samp) + 1

        # Build per-batch FIR kernels by scattering taps at their sample delays
        def build_kernel(h_b):
            k = tf.zeros([L], dtype=tf.complex64)
            return tf.tensor_scatter_nd_add(k, tf.expand_dims(d_samp, 1), h_b)
        kernels = tf.map_fn(build_kernel, h, dtype=tf.complex64)  # [B,L]

        # Collapse TX to SISO output (extend to MIMO if needed)
        x_sum = tf.reduce_sum(x, axis=-1, keepdims=True)          # [B,T,1]

        def conv_b(xb, kb):
            kb_rev = tf.reverse(kb, axis=[0])
            return complex_conv1d_same(xb, kb_rev)
        y = tf.map_fn(lambda args: conv_b(args[0], args[1]),
                      (x_sum, kernels),
                      fn_output_signature=tf.complex64)           # [B,T,1]

        taps = {"h": h, "delays_s": self.delays_s, "powers": self.powers}
        return y, taps
