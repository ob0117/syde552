import numpy as np

class LIFCollection:
    def __init__(self, n=1, dim=1, tau_rc=0.02, tau_ref=0.002, v_th=1,
                 max_rates=(200, 400), intercept_range=(-1, 1),
                 t_step=0.01, v_init=0.0):
        self.n = n
        self.dim = dim
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.v_th = np.ones(n) * v_th
        self.t_step = t_step
        self.voltage = np.ones(n) * v_init
        self.refractory_time = np.zeros(n)
        self.output = np.zeros(n)
        max_rates_tensor = np.random.uniform(*max_rates, n)
        intercepts_tensor = np.random.uniform(*intercept_range, n)
        self.gain = (
            self.v_th *
            (1 - 1 / (1 - np.exp((self.tau_ref - 1/max_rates_tensor) /
                                 self.tau_rc))) /
            (intercepts_tensor - 1)
        )
        self.bias = np.expand_dims(
            self.v_th - self.gain * intercepts_tensor, axis=1
        )
        self.encoders = np.random.randn(n, self.dim)
        self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, None]
        
    def reset(self):
        self.voltage[:] = 0.0
        self.refractory_time[:] = 0.0
        self.output[:] = 0.0
        
    def step(self, inputs):
        dt = self.t_step
        self.refractory_time -= dt
        delta_t = np.clip(dt - self.refractory_time, 0.0, dt)
        I = np.sum(self.bias + inputs * self.encoders * self.gain[:, None],
                   axis=1)
        self.voltage = I + (self.voltage - I) * np.exp(-delta_t / self.tau_rc)
        spike_mask = self.voltage > self.v_th
        self.output[:] = spike_mask / dt
        t_spike = (
            self.tau_rc *
            np.log((self.voltage[spike_mask] - I[spike_mask]) /
                   (self.v_th[spike_mask] - I[spike_mask])) +
            dt
        )
        self.voltage[spike_mask] = 0.0
        self.refractory_time[spike_mask] = self.tau_ref + t_spike
        return self.output

def compute_firing_rates(Q, K, n_steps=100, t_step=1e-2, lif_kwargs=None, seed=None):
    if Q.shape != K.shape:
        raise ValueError("Q and K must have the same shape (n,d)")
    n, d = Q.shape
    if seed is not None:
        np.random.seed(seed)
    lif_kwargs = lif_kwargs or {}
    lif = LIFCollection(n=n, dim=d, t_step=t_step, **lif_kwargs)
    enc = K.astype(float).copy()
    enc /= np.linalg.norm(enc, axis=1, keepdims=True) + 1e-12
    lif.encoders = enc
    spikes = np.zeros(n)
    for _ in range(n_steps):
        spikes += lif.step(Q) * t_step
    duration = n_steps * t_step
    return spikes / duration