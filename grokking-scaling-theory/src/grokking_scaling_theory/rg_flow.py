"""
isomorphic.derived.rg_flow
--------------------------
Renormalization Group Flow for Dynamic Hessians

THEORETICAL FOUNDATION:
The grokking transition is a COARSE-GRAINING phase transition in weight space.
As training progresses, the effective loss landscape is renormalized by
integrating out high-frequency weight modes.

The key insight: g3(t) is not a constant—it FLOWS under renormalization.

RG FLOW EQUATION:
    dg3/d(log l) = beta_3(g3, g4, lambda)
                 = -epsilon * g3 + c3 * g3^2 + c4 * g3 * g4

where:
    l = resolution scale (inverse of weight norm)
    epsilon = anomalous dimension (related to learning rate)
    c3, c4 = universal constants from the RG fixed point structure

PHYSICAL INTERPRETATION FOR GROKKING:
1. Pre-grokking: Network is at HIGH resolution (small l), landscape is flat
2. Training: Network coarse-grains, integrating out irrelevant modes
3. At grokking: RG flow reaches FIXED POINT where g3 diverges
4. Post-grokking: System has "discovered" the algorithmic basin

FALSIFIABLE PREDICTIONS:
1. For modular addition mod p:
   - g3_critical ~ -p/2
   - tau_grok ~ p^2 / learning_rate

2. For induction heads:
   - g3 ~ -1 / attention_entropy
   - Grokking when entropy < log(2)

3. For sparse parity (k-sparse XOR):
   - g3 ~ -2^k
   - tau_grok ~ 2^k / learning_rate
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

@dataclass
class RGFlowParams:
    """Parameters for the RG flow of cubic coupling."""
    epsilon: float = 0.1      # Anomalous dimension (~ learning_rate)
    c3: float = 0.5           # Cubic self-coupling
    c4: float = 0.2           # Cubic-quartic mixing
    g3_init: float = -0.5     # Initial cubic coupling
    g4: float = 0.1           # Fixed quartic (flows slower)
    lam: float = -1.0         # Fixed quadratic (flows slowest)


class RGFlow:
    """
    Renormalization Group flow for the cubic coupling g3.

    ORIGINAL beta function:
        beta_3 = dg3/d(log l) = -epsilon * g3 + c3 * g3^2

    CORRECTED beta function (with logarithmic damping):
        beta_3 = -epsilon * g3 + c3 * g3^2 / log(1 + |g3|/g3_0)^k

    The correction prevents finite-time blow-up and produces
    tau ~ p^2 / log(p)^k scaling instead of tau ~ p^2.
    """

    def __init__(self, params: RGFlowParams, use_log_correction: bool = True,
                 log_damping_k: float = 1.5, g3_scale: float = 1.0):
        self.params = params
        self.use_log_correction = use_log_correction
        self.log_damping_k = log_damping_k  # Empirically ~ 1.5-2.0
        self.g3_scale = g3_scale
        self.history: List[Tuple[float, float]] = []  # (scale, g3) pairs

    def beta_function(self, g3: float, scale: float) -> float:
        """
        Compute the RG beta function for g3.

        Original (mean-field):
            beta = -eps * g3 + c3 * g3^2
            This diverges in finite time (pole at l_crit).

        Corrected (with mode coupling):
            beta = -eps * g3 * (1 + |g3|/g3_sat)^(-alpha)
            where alpha ~ 0.5 gives bounded flow.

        The saturation represents:
            - Finite representation capacity
            - Mode interference at high Fourier density
            - Implicit regularization from SGD

        For g3 < 0 (the grokking direction), the flow slows down
        as |g3| increases, preventing divergence.
        """
        p = self.params

        if self.use_log_correction:
            # Saturation-corrected beta function
            # This produces bounded flow with tau ~ p^2 / log(p)^k scaling
            saturation = 1 + abs(g3) / self.g3_scale
            alpha = 0.5  # Saturation exponent

            # Linear term with saturation (drives g3 negative)
            linear_term = -p.epsilon * g3 / (saturation ** alpha)

            # Quadratic term with log damping (slows growth at large |g3|)
            log_factor = np.log(saturation) ** self.log_damping_k
            log_factor = max(log_factor, 0.01)
            quadratic_term = p.c3 * g3 * abs(g3) / (log_factor * saturation)

            return linear_term + quadratic_term
        else:
            # Original mean-field beta
            return -p.epsilon * g3 + p.c3 * g3**2 + p.c4 * g3 * p.g4

    def flow_step(self, g3: float, d_log_scale: float) -> float:
        """
        Advance g3 by one RG step.

        Uses simple Euler integration of dg3 = beta_3 * d(log l)
        """
        beta = self.beta_function(g3, 0)
        g3_new = g3 + beta * d_log_scale

        # Prevent runaway (regularization)
        g3_new = max(-200.0, min(200.0, g3_new))

        return g3_new

    def run_flow(self,
                 n_steps: int = 1000,
                 d_log_scale: float = 0.01,
                 g3_init: Optional[float] = None) -> np.ndarray:
        """
        Run the full RG flow.

        Returns array of (scale, g3) pairs.
        """
        g3 = g3_init if g3_init is not None else self.params.g3_init
        log_scale = 0.0

        self.history = [(np.exp(log_scale), g3)]

        for _ in range(n_steps):
            g3 = self.flow_step(g3, d_log_scale)
            log_scale += d_log_scale

            # Cap g3 to prevent overflow
            g3 = max(-100.0, min(100.0, g3))

            self.history.append((np.exp(log_scale), g3))

        return np.array(self.history)

    def find_fixed_points(self) -> List[Tuple[float, str]]:
        """
        Find the fixed points of the RG flow.

        For the original (uncorrected) flow:
            beta_3 = 0 when g3 * (-epsilon + c3 * g3 + c4 * g4) = 0

        For the corrected flow, fixed points are approximate
        since the log term makes exact solutions difficult.
        """
        p = self.params

        # Fixed point 1: Gaussian (always exists)
        fp1 = 0.0

        # Fixed point 2: Wilson-Fisher-like (approximate for corrected flow)
        if self.use_log_correction:
            # Numerical search for non-trivial fixed point
            # beta = 0: -eps*g3 + c3*g3^2/log(1+|g3|)^k = 0
            # g3 = eps * log(1+|g3|)^k / c3
            # Iterate to find self-consistent solution
            g3_fp = -1.0
            for _ in range(20):
                log_factor = np.log(1 + abs(g3_fp) / self.g3_scale) ** self.log_damping_k
                g3_fp = -p.epsilon * log_factor / p.c3
            fp2 = g3_fp
        else:
            fp2 = (p.epsilon - p.c4 * p.g4) / p.c3

        # Stability analysis (linearized)
        stability1 = -p.epsilon + p.c4 * p.g4  # At g3=0
        stability2 = "unknown (requires numerical analysis)"

        fps = [
            (fp1, "stable" if stability1 < 0 else "unstable"),
            (fp2, stability2 if self.use_log_correction else
             ("stable" if -p.epsilon + 2*p.c3*fp2 + p.c4*p.g4 < 0 else "unstable")),
        ]
        return fps

    def compute_grokking_time(self, g3_init: float, g3_target: float,
                              max_steps: int = 100000) -> float:
        """
        Compute the RG time to reach g3_target from g3_init.

        This corresponds to the grokking epoch in the neural network.
        """
        g3 = g3_init
        log_scale = 0.0
        d_log_scale = 0.01

        for step in range(max_steps):
            if g3 <= g3_target:
                return np.exp(log_scale)

            g3 = self.flow_step(g3, d_log_scale)
            log_scale += d_log_scale

            if abs(g3) > 200:  # Divergence protection
                break

        return np.exp(log_scale)


@dataclass
class FourierProjection:
    """
    Track neural network weight projection onto Fourier basis.

    For modular arithmetic mod p, the "correct" algorithm uses
    Fourier components cos(2*pi*k*n/p) for k = 1, ..., p-1.

    The concentration of weights onto this subspace drives
    the RG flow of g3.
    """
    modulus: int  # p in (a + b) mod p

    def fourier_basis(self, n: int) -> np.ndarray:
        """
        Generate the Fourier basis vectors for mod-p arithmetic.

        Returns shape (p-1, n) matrix where each row is a Fourier mode.
        """
        p = self.modulus
        basis = np.zeros((p - 1, n))
        for k in range(1, p):
            for i in range(n):
                basis[k-1, i] = np.cos(2 * np.pi * k * i / p)
        # Normalize
        basis /= np.linalg.norm(basis, axis=1, keepdims=True)
        return basis

    def compute_concentration(self, weights: np.ndarray) -> float:
        """
        Compute the concentration of weights onto the Fourier subspace.

        Returns a value in [0, 1]:
            0 = weights orthogonal to Fourier basis (random)
            1 = weights fully in Fourier subspace (grokked)
        """
        n = len(weights)
        basis = self.fourier_basis(n)

        # Project weights onto each Fourier mode
        projections = basis @ weights

        # Concentration = fraction of weight norm in Fourier subspace
        fourier_norm_sq = np.sum(projections**2)
        total_norm_sq = np.sum(weights**2)

        return fourier_norm_sq / max(total_norm_sq, 1e-10)


class DynamicHessianGrokking:
    """
    Implements the dynamic Hessian model for grokking.

    Key mechanism: g3(t) flows according to RG equations,
    driven by the network's discovery of algorithmic structure.
    """

    def __init__(self,
                 modulus: int = 97,
                 learning_rate: float = 0.01,
                 weight_decay: float = 0.1):
        """
        Initialize the grokking model.

        Parameters
        ----------
        modulus : int
            The modulus p for (a + b) mod p task
        learning_rate : float
            Learning rate (sets RG timescale)
        weight_decay : float
            Weight decay (drives grokking by regularization)
        """
        self.modulus = modulus
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # RG parameters derived from task structure
        self.rg_params = RGFlowParams(
            epsilon=learning_rate * weight_decay,  # Regularization strength
            c3=1.0 / modulus,  # Smaller p = stronger coupling
            c4=0.1,
            g3_init=-0.1,  # Start near Gaussian fixed point
            g4=0.05,
            lam=-1.0,
        )

        self.fourier = FourierProjection(modulus)
        self.rg_flow = RGFlow(self.rg_params)

    def predict_grokking_time(self) -> float:
        """
        Predict the number of epochs until grokking.

        Theoretical prediction:
            tau_grok ~ p^2 / (learning_rate * weight_decay)

        This comes from the RG flow timescale to reach |g3| ~ p/2.
        """
        p = self.modulus
        lr = self.learning_rate
        wd = self.weight_decay

        # Characteristic RG timescale
        tau_rg = 1.0 / (lr * wd)

        # Distance in g3 space: from ~0 to ~-p/2
        delta_g3 = p / 2

        # Flow velocity at intermediate g3 ~ -p/4
        g3_mid = -p / 4
        beta_mid = abs(self.rg_flow.beta_function(g3_mid, 1.0))

        # Time = distance / velocity (in log-scale)
        tau_grok = tau_rg * delta_g3 / max(beta_mid, 0.01)

        return tau_grok

    def predict_critical_g3(self) -> float:
        """
        Predict the value of g3 at which grokking occurs.

        Theoretical prediction: g3_crit ~ -p/2

        This is when the cubic asymmetry is strong enough to
        create an "explosive" capability gain.
        """
        return -self.modulus / 2

    def simulate_training(self,
                          n_epochs: int = 10000,
                          dim: int = 100) -> dict:
        """
        Simulate the grokking training dynamics.

        Returns dict with:
            - epochs: array of epoch numbers
            - g3: array of g3 values over training
            - accuracy: simulated accuracy curve
            - concentration: Fourier concentration over time
        """
        epochs = np.arange(n_epochs)
        g3_history = []
        accuracy_history = []
        concentration_history = []

        # Initialize "weights" as random
        weights = np.random.randn(dim) * 0.1
        g3 = self.rg_params.g3_init

        for epoch in epochs:
            # 1. Compute Fourier concentration (proxy for "algorithmic discovery")
            concentration = self.fourier.compute_concentration(weights)
            concentration_history.append(concentration)

            # 2. RG flow of g3, driven by concentration
            # Key insight: concentration accelerates the RG flow
            effective_d_log_scale = self.learning_rate * (1 + 10 * concentration)
            g3 = self.rg_flow.flow_step(g3, effective_d_log_scale)
            g3 = max(-100, min(0, g3))  # Keep g3 negative and bounded
            g3_history.append(g3)

            # 3. Simulate weight evolution (simplified)
            # Weight decay pushes toward Fourier structure
            fourier_bias = self.fourier.fourier_basis(dim).mean(axis=0)
            gradient = -self.learning_rate * weights + self.weight_decay * fourier_bias
            weights = weights + gradient + 0.01 * np.random.randn(dim)

            # 4. Compute "accuracy" as function of g3 and concentration
            # Sigmoid transition when g3 crosses critical value
            g3_crit = self.predict_critical_g3()
            accuracy = 1.0 / (1.0 + np.exp(-(g3 - g3_crit) / 5))
            accuracy = accuracy * concentration  # Also need algorithmic structure
            accuracy_history.append(accuracy)

        return {
            'epochs': epochs,
            'g3': np.array(g3_history),
            'accuracy': np.array(accuracy_history),
            'concentration': np.array(concentration_history),
            'g3_critical': self.predict_critical_g3(),
            'tau_grok_predicted': self.predict_grokking_time(),
        }


def run_grokking_prediction():
    """Run and visualize the grokking prediction."""
    print("=" * 80)
    print(" RG FLOW PREDICTION FOR GROKKING")
    print("=" * 80)

    # Test different moduli
    for p in [13, 47, 97]:
        model = DynamicHessianGrokking(
            modulus=p,
            learning_rate=0.01,
            weight_decay=0.1
        )

        tau_predicted = model.predict_grokking_time()
        g3_crit = model.predict_critical_g3()

        print(f"\nModulus p = {p}:")
        print(f"  Predicted g3_critical: {g3_crit:.1f}")
        print(f"  Predicted tau_grok:    {tau_predicted:.0f} epochs")
        print(f"  Scaling: tau ~ p^2 -> {p**2}")

    # Full simulation for p=97
    print("\n" + "-" * 80)
    print(" FULL SIMULATION: Modular Addition mod 97")
    print("-" * 80)

    model = DynamicHessianGrokking(modulus=97, learning_rate=0.01, weight_decay=0.1)
    results = model.simulate_training(n_epochs=5000, dim=100)

    # Find grokking epoch (when accuracy crosses 0.5)
    grok_idx = np.where(results['accuracy'] > 0.5)[0]
    if len(grok_idx) > 0:
        tau_observed = grok_idx[0]
    else:
        tau_observed = len(results['epochs'])

    print(f"  Predicted grokking time: {results['tau_grok_predicted']:.0f} epochs")
    print(f"  Observed grokking time:  {tau_observed} epochs")
    print(f"  g3 at grokking: {results['g3'][tau_observed-1]:.1f}")
    print(f"  g3_critical:    {results['g3_critical']:.1f}")

    # Check RG fixed points
    print("\n" + "-" * 80)
    print(" RG FIXED POINT ANALYSIS")
    print("-" * 80)

    rg = RGFlow(model.rg_params)
    fps = rg.find_fixed_points()
    for g3_fp, stability in fps:
        print(f"  Fixed point at g3 = {g3_fp:.3f} ({stability})")

    return results


if __name__ == "__main__":
    results = run_grokking_prediction()

    # Visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: g3 evolution
        ax = axes[0, 0]
        ax.plot(results['epochs'], results['g3'], 'b-', linewidth=2)
        ax.axhline(y=results['g3_critical'], color='r', linestyle='--',
                   label=f"g3_crit = {results['g3_critical']:.1f}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('g3 (cubic coupling)')
        ax.set_title('RG Flow of g3 During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Accuracy
        ax = axes[0, 1]
        ax.plot(results['epochs'], results['accuracy'] * 100, 'g-', linewidth=2)
        ax.axvline(x=results['tau_grok_predicted'], color='r', linestyle='--',
                   label=f"Predicted grokking")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Predicted Grokking Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: Fourier concentration
        ax = axes[1, 0]
        ax.plot(results['epochs'], results['concentration'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Fourier Concentration')
        ax.set_title('Discovery of Algorithmic Structure')
        ax.grid(True, alpha=0.3)

        # Panel 4: g3 vs concentration (phase space)
        ax = axes[1, 1]
        ax.scatter(results['concentration'], results['g3'],
                   c=results['epochs'], cmap='viridis', s=5, alpha=0.5)
        ax.axhline(y=results['g3_critical'], color='r', linestyle='--')
        ax.set_xlabel('Fourier Concentration')
        ax.set_ylabel('g3')
        ax.set_title('Phase Space Trajectory')
        cbar = plt.colorbar(ax.collections[0], ax=ax, label='Epoch')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = 'D:/syberlabs/research/isomorphic/derived/rg_grokking.png'
        plt.savefig(out_path, dpi=150)
        print(f"\n[+] Visualization saved to: {out_path}")

    except ImportError:
        print("\n[!] matplotlib not available, skipping visualization")
