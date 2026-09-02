# Controlled Asymptotic Derivation of the Grokking Scaling Law

## Goal

Derive the narrowest defensible form of the current empirical law:

`tau ~ p^2 / ((log p)^2 * wd^beta)`

with:

- leading problem-size burden `p^2`
- subleading logarithmic enhancement `log(p)^2`
- separable cleanup factor `wd^{-beta}`

This note is intentionally careful. It distinguishes what is actually derived from what
is still heuristic.

## Step 1: Cleanup Dynamics

Start from the memorization equation:

`dM/dt = -a1 * wd^beta * M`

This solves exactly:

`M(t) = M0 * exp(-a1 * wd^beta * t)`

So the characteristic cleanup time is:

`tau_M ~ wd^{-beta}`

This is the cleanest and strongest part of the derivation.

## Step 2: Why the Naive Rule Equation Is Insufficient

If one writes

`dR/dt = a2 * R * (1-R) / (log p)^k`

then the resulting timescale is only:

`tau_R ~ (log p)^k`

This cannot generate the leading `p^2` dependence.

Therefore the leading size dependence cannot live only in a logarithmically slowed
logistic equation. It must appear in the effective rule-formation rate itself.

## Step 3: Effective Rule-Formation Rate

Write instead:

`dR/dt = a2 * gamma_R(p) * R * (1-R)`

Now the question becomes: what is `gamma_R(p)`?

The current mechanistic claim is:

`gamma_R(p) ~ log(p)^2 / p^2`

which immediately gives

`tau_R ~ p^2 / log(p)^2`

## Step 4: Physical Origin of `gamma_R(p)`

Two ingredients produce the numerator.

### 4.1 Harmonic Mode Accumulation

If task-aligned Fourier contributions obey an effective envelope

`a_m ~ 1/m`

then the coherent recruitment gain is

`H(p) = sum_{m <= p} 1/m ~ log p`

### 4.2 Marginal Coordination Susceptibility

If coordination among weak rule-forming modes is governed by a marginal or nearly
marginal field, its integrated response contributes a second logarithm:

`chi(p) ~ log p`

### 4.3 Combined Rate

With a baseline coordination burden `p^2`, the effective rate becomes

`gamma_R(p) ~ H(p) * chi(p) / p^2 ~ log(p)^2 / p^2`

This is the controlled asymptotic core of the derivation.

## Step 5: Deployment and the Stopping Condition

Deployment obeys

`dD/dt = a3 * R * (1-D) - a4 * M * D`

The full grokking time is therefore

`tau = inf { t : D(t) >= D_crit }`

This means the exact stopping time is a property of a coupled system, not a product of
independent clocks.

## Step 6: Why the Product Form Is Still Reasonable

The observed empirical law is factorized:

`tau ~ p^2 / ((log p)^2 * wd^beta)`

This does not follow as a strict identity from `tau = max(tau_R, tau_M)`, nor from an
arbitrary product ansatz.

The narrow scientific claim is instead:

- modulus primarily controls `gamma_R(p)` and therefore `tau_R`
- weight decay primarily controls cleanup through `M(t)` and therefore `tau_M`
- over the currently observed regime, these two effects appear approximately separable

So the factorized law is justified as a controlled effective approximation in the
observed regime, not as a theorem of the full nonlinear system.

## What Is Derived

- exact cleanup time `tau_M ~ wd^{-beta}` from the `M` equation
- necessity of placing the `p^2` dependence in the effective rule-formation rate
- rule-formation timescale `tau_R ~ p^2 / log(p)^2` once `gamma_R(p)` is specified

## What Remains Heuristic

- the microscopic origin of the `a_m ~ 1/m` spectrum
- the derivation of `chi(p) ~ log p` from a closed RG flow
- the precise conditions under which the full stopping time factorizes

## Publication-Safe Conclusion

The current scaling law can be justified as a controlled asymptotic approximation of a
coupled latent-variable system. Cleanup contributes the factor `wd^{-beta}` through an
exact exponential decay of memorization burden, while rule formation contributes the
factor `p^2 / log(p)^2` through a problem-size-dependent effective rate
`gamma_R(p) ~ log(p)^2 / p^2`. The remaining open work is to derive the harmonic
spectrum and marginal susceptibility from a more microscopic training dynamics.
