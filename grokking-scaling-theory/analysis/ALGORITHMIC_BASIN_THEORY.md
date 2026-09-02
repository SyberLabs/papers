# Algorithmic Basin Theory for Grokking Scaling

## Executive Summary

The log-exponent q in the scaling law `tau ~ p^2 / (log(p)^q * wd^beta)` is not a universal constant but a **signature of the algorithmic family** the network converges to. This document formalizes:

1. **Basin selection criteria** - what determines which algorithm a network learns
2. **Algorithm-to-exponent mapping** - the theoretical prediction of q for each family
3. **gamma_R(p) derivations** - the effective rule-formation rate for each algorithm
4. **Testable predictions** - falsifiable experiments to validate the theory

---

## 1. Algorithmic Families in Modular Arithmetic

### 1.1 The Fourier Family (q = 2)

**Definition**: Networks that learn modular arithmetic via Fourier modes on Z_p.

**Mechanism**: The network discovers that modular addition corresponds to phase rotation in Fourier space:
```
(a + b) mod p  <-->  e^{2pi i k a/p} * e^{2pi i k b/p} = e^{2pi i k (a+b)/p}
```

**Coordination structure**:
- The network must align O(p^2) input pairs to the same Fourier phase
- Weak modes with amplitude a_m ~ 1/m require harmonic accumulation
- The coherent recruitment gain is H(p) = sum_{m=1}^{p-1} 1/m ~ log(p)
- Coordination susceptibility (how easily modes lock) scales as chi(p) ~ log(p)

**Rule-formation rate**:
```
gamma_R^{Fourier}(p) = H(p) * chi(p) / p^2 = log(p)^2 / p^2
```

**Predicted scaling**: tau ~ p^2 / log(p)^2, hence **q = 2**

**Empirical support**: Published MLP regime shows q = 1.95 with CV = 0.056

---

### 1.2 The Direct Position Encoding Family (q = 0)

**Definition**: Networks that learn circular position embeddings without explicit Fourier decomposition.

**Mechanism**: The network learns a 2D circular embedding:
```
a  -->  (cos(2pi*a/p), sin(2pi*a/p))
```
Then uses vector operations to compute the sum.

**Coordination structure**:
- Each input position maps to a fixed embedding
- No weak-mode accumulation - the circular structure is learned directly
- Coordination burden scales as O(p) (embed each position) not O(p^2)

**Rule-formation rate**:
```
gamma_R^{Position}(p) = 1 / p^2
```
(The p^2 remains because there are still p^2 input pairs, but no log enhancement)

**Predicted scaling**: tau ~ p^2, hence **q = 0**

**When expected**: Architectures with explicit positional encodings, or when initialization biases toward direct geometric representations.

---

### 1.3 The Lookup Table Family (q < 0 or non-power-law)

**Definition**: Networks that memorize input-output pairs without discovering algorithmic structure.

**Mechanism**: The network learns a separate output for each (a, b) pair:
```
(a, b)  -->  f(a, b) = (a + b) mod p  [as a lookup]
```

**Coordination structure**:
- O(p^2) independent entries to memorize
- No coordination benefit from structure
- Generalization only occurs when all p^2 entries are stored

**Rule-formation rate**:
```
gamma_R^{Lookup}(p) = 1 / p^4  [or worse]
```

**Predicted scaling**: tau ~ p^4 (or exponential in pathological cases)

**When expected**:
- Very small p (lookup is cheaper than Fourier discovery)
- Insufficient weight decay (no pressure to compress)
- Random seeds that initialize far from Fourier basin

---

### 1.4 The Hybrid Family (variable q)

**Definition**: Networks that use Fourier modes for some residue classes and memorization for others.

**Mechanism**: Partial Fourier learning:
- Easy residue classes (small k) learned via Fourier
- Hard residue classes (large k or irregular structure) memorized

**Coordination structure**:
- Mixed: O(p^2) coordination for Fourier portion, O(p^2) memory for lookup portion
- The ratio determines effective q

**Rule-formation rate**:
```
gamma_R^{Hybrid}(p) ~ alpha * log(p)^2 / p^2 + (1 - alpha) / p^4
```
where alpha is the fraction of Fourier-learned classes.

**Predicted scaling**: Intermediate q in (0, 2), or non-power-law behavior

**When expected**: Intermediate training budgets, mixed architectures, or boundary conditions between regimes.

---

## 2. Basin Selection Criteria

### 2.1 The Basin Landscape

At initialization, the network occupies a high-dimensional weight space. Multiple algorithmic basins exist:
```
                    [Weight Space]
                         |
         +---------+-----+-----+---------+
         |         |           |         |
     [Fourier] [Position] [Hybrid]  [Lookup]
       q=2       q=0       q~1       q<0
```

**Key insight**: The basin a network converges to depends on:
1. **Architecture** - determines which basins are accessible
2. **Initialization** - determines starting position relative to basins
3. **Training dynamics** - determines which basin gradient descent finds first
4. **Modulus p** - determines relative basin sizes and depths

### 2.2 Architecture Dependence

| Architecture | Fourier Basin | Position Basin | Lookup Basin |
|--------------|---------------|----------------|--------------|
| Shallow MLP | Large, deep | Small | Large, shallow |
| Deep MLP | Medium | Medium | Large |
| Residual | Small | Large | Medium |
| Transformer | Large (attention helps) | Medium | Small |

**Rationale**:
- **Shallow MLPs**: Limited depth forces efficient representations. Fourier basis is the most efficient for modular arithmetic (O(log p) neurons suffice). Lookup basin is accessible but shallow (gradient descent prefers compression).

- **Residual networks**: Skip connections allow position information to propagate directly. This enlarges the position encoding basin. Early layers can pass raw positions to later layers, reducing pressure to discover Fourier structure.

- **Transformers**: Attention patterns can implement Fourier-like operations naturally. The softmax-weighted sum over positions resembles discrete Fourier transform. Large Fourier basin.

### 2.3 Initialization Dependence

**Hypothesis**: Initialization scale affects basin selection.

- **Small init**: Weights start near origin. Gradient descent explores slowly. First basin encountered wins. For modular arithmetic, this is typically Fourier (most efficient gradient direction).

- **Large init**: Weights start far from origin. May be closer to lookup basin (random function approximation). Takes longer to fall into any structured basin.

**Testable prediction**: Networks with smaller initialization should show q closer to 2 more reliably.

### 2.4 Modulus Dependence

**Critical observation**: Small-p anomaly in local sweep data.

At p = 31, the network takes 20,000 epochs (max epochs) - this may reflect:
1. **Censoring**: Network never actually grokked
2. **Basin misidentification**: Network converged to lookup instead of Fourier
3. **Finite-size effect**: Fourier basin shrinks at small p

**Hypothesis**: There exists a critical modulus p_c below which the Fourier basin becomes subdominant.

**Mechanism**: At small p, lookup table has only p^2 entries. For p = 31, that's ~961 entries. A network with 256-width hidden layer has enough capacity to memorize this. The Fourier solution requires discovering subtle frequency relationships, which may be harder than brute memorization when p is small enough.

**Predicted critical modulus**:
```
p_c ~ sqrt(H)
```
where H is the hidden layer width. For H = 256, p_c ~ 16.

This suggests p = 31 is *above* p_c for the published regime but may be *at* a boundary for different architectures.

### 2.5 Weight Decay Dependence

**Hypothesis**: Strong weight decay enlarges the Fourier basin.

**Mechanism**:
- Weight decay penalizes large weights
- Lookup tables require large weights (sharp decision boundaries for each entry)
- Fourier representations are smooth (continuous phase rotation) and can use smaller weights
- Therefore, weight decay preferentially shrinks the lookup basin

**Testable prediction**: At fixed p and architecture:
- High wd --> q closer to 2 (Fourier dominates)
- Low wd --> q varies more (multiple basins accessible)

---

## 3. Formalized gamma_R(p) for Each Family

### 3.1 General Form

The effective rule-formation rate has the general structure:
```
gamma_R(p) = [coordination_gain(p)] / [coordination_burden(p)]
```

For modular arithmetic:
- **Burden** is always O(p^2) because there are p^2 input pairs
- **Gain** depends on the algorithm's coordination structure

### 3.2 Fourier Family Derivation

**Step 1: Harmonic accumulation**

The Fourier basis for Z_p consists of characters chi_k(a) = exp(2pi i k a / p) for k = 0, 1, ..., p-1.

Modular addition corresponds to:
```
chi_k(a + b) = chi_k(a) * chi_k(b)
```

If the network recruits modes with amplitude a_k ~ 1/k (harmonic decay), the total signal strength is:
```
H(p) = sum_{k=1}^{p-1} a_k ~ sum_{k=1}^{p-1} 1/k ~ log(p)
```

**Step 2: Coordination susceptibility**

For modes to lock coherently, they must coordinate across all p^2 input pairs. The susceptibility measures how a small perturbation in one mode affects others.

Near the Fourier fixed point, the susceptibility diverges logarithmically:
```
chi(p) ~ d(coherence)/d(perturbation) ~ log(p)
```

This is analogous to the susceptibility divergence at a continuous phase transition.

**Step 3: Combined rate**

The rule-formation rate is the product of gain terms divided by burden:
```
gamma_R^{Fourier}(p) = H(p) * chi(p) / p^2 = log(p)^2 / p^2
```

**Step 4: Timescale**

The logistic growth equation dR/dt = a_2 * gamma_R(p) * R(1-R) has saturation time:
```
tau_R ~ 1 / gamma_R(p) = p^2 / log(p)^2
```

Hence q = 2.

### 3.3 Position Encoding Family Derivation

**Step 1: Direct embedding**

The network learns a map a --> (cos(theta_a), sin(theta_a)) where theta_a = 2pi a / p.

This requires learning p embedding positions, not discovering subtle Fourier relationships.

**Step 2: No harmonic enhancement**

There is no sum over modes - the circular structure is learned directly:
```
H^{Position}(p) = O(1)
```

**Step 3: Direct coordination**

Coordination is local in embedding space. No marginal susceptibility enhancement:
```
chi^{Position}(p) = O(1)
```

**Step 4: Combined rate**

```
gamma_R^{Position}(p) = O(1) * O(1) / p^2 = 1 / p^2
```

**Step 5: Timescale**

```
tau_R ~ p^2 / 1 = p^2
```

Hence q = 0.

### 3.4 Lookup Family Derivation

**Step 1: Independent entries**

Each (a, b) pair is memorized independently. The "coordination" is trivial - no structure is exploited.

**Step 2: Negative gain**

The burden is O(p^2) entries, but there's no coordination gain:
```
gamma_R^{Lookup}(p) = 1 / p^4
```

(The extra p^2 in the denominator reflects that memorizing each entry is as hard as learning p positions, done p^2 times.)

**Step 3: Timescale**

```
tau_R ~ p^4
```

This corresponds to q = -2 in the framework tau ~ p^2 / log(p)^q, though the scaling is better described as tau ~ p^4 directly.

---

## 4. Testable Predictions

### 4.1 Fourier Concentration Test

**Prediction**: Networks with q ~ 2 should show strong Fourier mode concentration.

**Test**:
1. Train network until grokking
2. Extract hidden representations h(a) for all inputs a in {0, ..., p-1}
3. Compute DFT: H_k = sum_a h(a) * exp(-2pi i k a / p)
4. Compute concentration ratio: C = max_k |H_k|^2 / sum_k |H_k|^2

**Prediction**:
- q ~ 2 networks: C > 0.5 (strong mode concentration)
- q ~ 0 networks: C ~ 1/p (uniform, no mode structure)
- q < 0 networks: C ~ 1/p^2 (noise-like)

### 4.2 Initialization Sensitivity Test

**Prediction**: Smaller initialization should more reliably yield q ~ 2.

**Test**:
1. Fix architecture, modulus, and weight decay
2. Sweep initialization scale: sigma in {0.01, 0.1, 1.0, 10.0}
3. For each sigma, run 10 seeds and measure grokking time
4. Fit q for each sigma

**Prediction**:
- Small sigma (0.01): q ~ 2 consistently
- Large sigma (10.0): q varies widely, often < 2

### 4.3 Critical Modulus Test

**Prediction**: Below p_c ~ sqrt(H), the Fourier basin is subdominant.

**Test**:
1. Fix H = 256 (predicts p_c ~ 16)
2. Sweep p in {11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
3. For each p, measure grokking time and Fourier concentration
4. Identify transition point where q drops and concentration drops

**Prediction**:
- p < 20: Variable behavior, some seeds go to lookup, q << 2
- p > 30: Consistent Fourier learning, q ~ 2

### 4.4 Weight Decay Amplification Test

**Prediction**: Stronger weight decay should stabilize q ~ 2.

**Test**:
1. Fix architecture and modulus at p = 53 (boundary region)
2. Sweep wd in {0.01, 0.1, 0.5, 1.0, 2.0}
3. For each wd, measure grokking time across 10 seeds
4. Compute q and variance

**Prediction**:
- Low wd (0.01): High variance in q, some seeds q << 2
- High wd (2.0): Low variance, q consistently ~ 2

### 4.5 Architecture Transfer Test

**Prediction**: Residual networks should show q < 2 due to enlarged position basin.

**Test**:
1. Train shallow MLP and residual network on same (p, wd) grid
2. Measure grokking times for both
3. Fit q separately for each architecture

**Prediction**:
- MLP: q ~ 2 (Fourier basin dominant)
- Residual: q ~ 1 or variable (Position basin enlarged)

---

## 5. Implications for the (M, R, D) Framework

### 5.1 What Stays Constant

The two-stage hypothesis remains valid across all algorithmic families:
1. **M(t)**: Memorization decays exponentially under weight decay
2. **R(t)**: Rule structure grows logistically
3. **D(t)**: Deployment follows rule structure, suppressed by memorization

The dynamical system structure is **algorithm-independent**.

### 5.2 What Changes

The effective rule-formation rate gamma_R(p) is **algorithm-dependent**:

```
gamma_R(p) = { log(p)^2 / p^2    if Fourier family
             { 1 / p^2           if Position family
             { 1 / p^4           if Lookup family
             { mixed             if Hybrid family
```

### 5.3 Research Program Reframing

Instead of seeking a single universal scaling law, the program becomes:

**Step 1**: Identify the algorithmic family (via Fourier concentration or other diagnostics)

**Step 2**: Predict gamma_R(p) from family membership

**Step 3**: Derive tau from the (M, R, D) dynamical system with the appropriate gamma_R

This is more powerful than a single scaling law because it:
- Explains regime boundaries (basin selection)
- Predicts which experiments will validate q ~ 2
- Identifies conditions where different scaling applies

---

## 6. Open Questions

1. **Basin geometry**: What is the exact shape of algorithmic basins in weight space? Can we visualize the boundaries?

2. **Transition dynamics**: When a network crosses from lookup to Fourier learning, is there a sharp transition or gradual interpolation?

3. **Higher-order corrections**: The log(p)^2 mechanism is a leading-order effect. What are the subleading corrections?

4. **Other tasks**: Does the algorithm-dependent framework extend to other modular arithmetic operations (multiplication, exponentiation)?

5. **Generalization**: What is the analog of "algorithmic families" for other grokking tasks (e.g., group theory, permutations)?

---

## 7. Summary Table

| Family | gamma_R(p) | q | Diagnostic | Conditions |
|--------|------------|---|------------|------------|
| Fourier | log(p)^2 / p^2 | 2 | High Fourier concentration | Standard MLP, moderate p, high wd |
| Position | 1 / p^2 | 0 | Circular embedding structure | Residual, positional encodings |
| Lookup | 1 / p^4 | -2 | Random function-like | Small p, low wd, large init |
| Hybrid | mixed | 0-2 | Partial structure | Boundary conditions |

---

## 8. Conclusions

The failure of the q ~ 2 scaling law across regimes is not a weakness - it is the central prediction of the algorithm-dependent scaling hypothesis. The log-squared correction is a signature of Fourier learning, and its presence or absence tells us which algorithmic basin the network has discovered.

This reframing transforms the empirical program from "finding a universal constant" to "identifying algorithmic families and their scaling signatures." The (M, R, D) framework provides the universal skeleton; gamma_R(p) encodes the algorithmic flesh.

**Key claim**: q ~ 2 is the Fourier fingerprint. Different algorithms leave different fingerprints.
