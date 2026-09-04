# Coverage artifacts in computational ethnobotany: decoupled reward, controlled coupling, and search benchmarks in high-dimensional plant landscapes

*Short title:* Searching the Green Hypercube

**[Author]**, **[Affiliation]**
[corresponding author / ORCID: TODO]

---

> **Draft status.** Full first draft assembled from the hypothesis ledger and outline. All quantitative results are as reported in the project ledger; all citations are from the audited bibliography (Part IX of the ledger). Items marked `[CONFIRM]` require the author to verify a specific detail; items marked `[TODO]` are placeholders (figures, affiliations, a small number of unsourced cells). Nothing in this draft introduces a statistic or citation not already established.

---

## Abstract

Computational ethnobotany merges biodiversity occurrence, interaction, phytochemistry, and bioactivity databases to prioritise plants for screening, but these sources sample research *attention* as much as latent ecological structure (Hortal et al., 2015). We introduce Green Hypercube, a reproducible search framework with decoupled reward provenance (ChEMBL measured potency; NAEB documented medicinal use), four cue channels, six sequential search strategies, and symmetric confound control applied to both cues and rewards. Across four live landscapes (a curated chemistry-seeded pool, an unfiltered Amazonian regional pool, a NAEB-documented pool, and a Moerman full-flora pool of 349 North American GBIF species, 35% NAEB-labelled), raw cue–reward coupling is strong but collapses under coverage control and reward-depth residualisation, with null multivariate partial R² on the keystone full-flora estimand. Sparsity sweeps show that apparent strategy advantage is strongly reward-density dependent: matched-density NAEB benchmarks have nearly overlapping raw sensory advantage, while unmatched enriched-to-full comparisons confound pool composition with reward sparsity. The M2 ladder indicates that surviving strategy wins track the effort surface rather than effort-independent cue structure; at matched density, enriched sensory genuine is practically equivalent to zero, while full-flora sensory genuine remains non-equivalent negative under severe chemistry-coverage sparsity. A phylogenetic re-analysis on the Saslis-Lagoudakis estimand shows that borderline clade-level clustering of NAEB use is explained by documentation effort, while apparent hot-node enrichment is multiplicity noise present before any effort adjustment. We argue that raw and effort-controlled coupling should be reported side by side as a standard for in silico ethnobotany, and that strategy benchmarks should match reward density before cross-pool comparisons are interpreted.

**Keywords:** computational ethnobotany; bioprospecting; sampling bias; phylogenetic signal; search; reproducibility; NAEB; ChEMBL

---

## 1. Introduction

The search for medicinally useful plants is often framed as drawing on accumulated traditional knowledge to shorten an otherwise astronomical search over chemical and botanical diversity. That framing motivates a now-large body of computational work that integrates biodiversity databases: occurrence records, species-interaction graphs, phylogenies, phytochemical inventories, and bioactivity assays: to rank species by predicted medicinal value. The motivating image is the tropical-forest pharmacopoeia; the empirical material in this paper is North American (NAEB) plus one Amazonian regional pool, and we keep the rainforest framing explicitly motivational rather than evidentiary.

The computational problem has the structure of sequential search under sparse, costly reward with incomplete observability: formally close to a bandit or foraging problem (Srinivas et al., 2010 `[CONFIRM]`; Pirolli & Card, 1999 `[CONFIRM]`). An algorithm encounters candidate species, spends a limited budget testing them, and accumulates reward (measured bioactivity, or documented use). The intuition behind ethnobotanically informed search is that cues observable before testing (taxonomy, chemistry, ecology, animal associations) carry exploitable information about which species will be rewarding.

This intuition sits in tension with two literatures. First, phylogenetic bioprospecting reports both success stories (related taxa share documented uses; cross-cultural "hot nodes" recur; Saslis-Lagoudakis et al., 2012) and weak or absent links between phylogeny and *measured bioactivity* (Rønsted et al., 2012; Grimbs et al., 2017). Second, the biodiversity-informatics literature warns that integrated databases are pervasively shaped by where and what researchers have studied: spatial and taxonomic sampling bias (Beck et al., 2014), the family of "shortfalls" that bound large-scale biodiversity knowledge (Hortal et al., 2015), and the disproportionate study of already-prominent species (Souza et al., 2018). If both cues and rewards are drawn from such databases, an apparent cue–reward association may reflect shared study effort rather than ecology.

We ask three questions:

1. When is measured cue–reward coupling a real ecological signal versus an artifact of research coverage?
2. Which search strategies exploit genuine versus spurious coupling?
3. How do reward provenance and reward sparsity modulate the value of structured search?

**Contributions.**

1. A reproducible **simulation framework** (Green Hypercube) with decoupled reward provenance and a negative-control harness.
2. A **coverage-controlled coupling** instrument applied, to our knowledge for the first time, to integrated GBIF/GloBI/chemistry cues against ChEMBL and NAEB rewards, with symmetric control on the reward side as well as the cue side.
3. **Four live landscapes** that contrast seed bias: curated, unfiltered regional, NAEB-enriched, and a Moerman full-flora pool that reconstructs the classical used-versus-available estimand.
4. **Empirical identification** of (i) a phylogenetic null for measured potency at the species scale and (ii) a study-effort mirage in raw coupling that collapses under symmetric control.
5. A **sparsity × provenance** interaction for structured search, consistent with the apparency literature in which medicinal use is among the weakest availability–use categories (Gonçalves et al., 2016; the literature is mixed: see §6.4 and Part IX).

---

## 2. Related work

### 2.1 Phylogenetic bioprospecting

Saslis-Lagoudakis et al. (2012) showed that traditionally used species cluster phylogenetically across three disparate floras, that cross-cultural hot nodes recur, and that species from those nodes are enriched in known bioactives relative to random draws: a clade-level, use-clustering result. Rønsted et al. (2012) found phylogenetic signal in *use* but not in *bioactivity* for Amaryllidaceae alkaloids, and Grimbs et al. (2017) found no phylogenetic structuring of phytochemistry/bioactivity within Rhododendron (one genus; we cite it as an example, not a universal law). Milliken et al. (2021) report a strong clade-clustering shape for Latin American antimalarials (NRI = 8.66; NTI = 0.33, not significant) without effort control. Souza et al. (2018) show ethnomedicinal use is phylogenetically overdispersed with hot nodes that attract more use reports and screening, yet find FDA-approved/clinical-trial genera disproportionately *outside* hot nodes (2/11): research effort tracking traditional-use structure without translating into forward drug yield.

The key methodological distinction this paper turns on: clade-level clustering of use is a different estimand from pairwise phylogenetic distance predicting continuous reward, and the two need not agree (§6.1).

### 2.2 ML and ethnobotanical prioritisation

Machine-learning prioritisation typically conflates target definitions (use vs. citation vs. activity) and rarely treats the label as missing-at-random. Richard-Bollans et al. (2023) is the closest prior work to ours: it builds classifiers for antiplasmodial plants and explicitly raises sampling bias, motivating label re-weighting. Domingo-Fernández et al. (2023) report non-random taxonomic clustering of therapeutic use at scale and argue use is empirically structured; we cite this as a **contrast**: it is consistent with our *raw* coupling result and becomes the thing to explain once effort is controlled, not an ally for an effort-independent signal.

### 2.3 Ecological encounter and apparency

The apparency / use-value tradition (Phillips & Gentry, 1993) predicts that common, easily encountered plants accumulate more documented uses, fusing the encounter gate with documentation density. Critically, apparency holds best for timber/fuel/construction and is weakest and most mixed for medicinal use specifically (Gonçalves et al., 2016, meta-analysis; Guèze et al., 2014 report a negative use-value–abundance relationship for medicine among the Tsimane'). This is the conceptual ancestor of our encounter gate and of the prediction that medicinal reward is the confound-prone case.

### 2.4 Database bias and shortfalls

Occurrence databases confound species presence with sampling effort (Beck et al., 2014; Hortal et al., 2015). Interaction data (GloBI; Poelen et al., 2014 `[CONFIRM]`) and species→compound bridges (LOTUS; Rutz et al., 2022 `[CONFIRM]`) carry their own coverage gaps, and co-occurrence is a poor proxy for ecological interaction (Blanchet et al., 2020). The standard fix on the occurrence side: modelling and residualising against an effort surface: is the template we extend to the reward side.

### 2.5 Search, transmission, and bandits

Optimal-foraging and bandit models formalise exploration/exploitation under sparse reward (Srinivas et al., 2010 `[CONFIRM]`; Pirolli & Card, 1999 `[CONFIRM]`), and cultural-transmission work situates documented plant knowledge as shaped by social processes and study attention (Reyes-García et al., 2009 `[CONFIRM]`). These frame our gate decomposition and the interpretation of strategy advantage as recovery of an effort surface rather than ecological exploitation.

### 2.6 Datasets and governance

We use the Native American Ethnobotany database (Moerman, 1991) as a documentation-derived reward and Dr. Duke's Phytochemical and Ethnobotanical Database as a chemistry/sensory cue. Because NAEB encodes Indigenous knowledge, we frame data handling against the CARE principles for Indigenous data governance (Carroll et al., 2020); §6.6 states the relevant limitations.

---

## 3. Conceptual framework

We decompose discovery multiplicatively into three terms:

> discovery ≈ P(encounter) × P(coupling | confounds removed) × strategy efficiency

The **encounter gate** is the probability a species enters the search pool at all; in database terms it is inseparable from sampling/observation effort (Beck et al., 2014; the apparency literature, Phillips & Gentry, 1993). The **coupling gate** is the probability a pre-test cue carries exploitable information about reward *after* confounds are removed. The **strategy modulator** is the efficiency with which a search policy converts whatever structure exists into early reward.

The data manifold has several faces: species, phylogeny, co-occurrence, animal association, chemistry/sensory salience, and one hidden face, reward (ChEMBL potency or NAEB use), which is never visible to a strategy during search. Each database field is, for our purposes, a cue, a reward, or a confound, and the central methodological move is to treat research effort symmetrically: effort enters both the cue side and the reward side, and a claim of ecological signal requires removing it from both.

We label every empirical claim by epistemic tier: **AUDIT** (built into the simulator; never evidence for the hypothesis it encodes), **CARTOGRAPHY** (descriptive measurement on one landscape), **EXTRAPOLATION** (suggestive; wide CIs, single landscape, or incomplete control), and **IDENTIFIED** (survives pre-specified falsification, confound control, and literature-consistent interpretation). Two rules follow: an **anti-tautology rule**: mechanisms encoded in data generation or strategy design cannot validate the hypothesis they implement; only measured coupling on independent reward counts, and a **symmetric-confound rule**: cue-side control is necessary but not sufficient, and ecological claims require reward-side residualisation.

*Figure 1 (schematic of the gate decomposition and data provenance) goes here. `[TODO: draw]`*

---

## 4. Methods

### 4.1 Data sources and adapters
GBIF (occurrences, co-occurrence), Open Tree of Life (phylogeny), GloBI (animal–plant interactions), Dr. Duke's (chemistry/sensory, mirror + offline), LOTUS (species→compound bridge), ChEMBL (bioactivity), and NAEB (documented use). All sources are accessed through versioned adapters that write to a frozen on-disk cache with a fixed schema, so every result is reproducible from cache without live re-query. `[CONFIRM cache schema details]`

### 4.2 Landscape seeding modes
Four seeding modes define the four landscapes: `chemistry` (curated Duke mirror), `region` (GBIF Amazonia), `naeb` (NAEB-documented, pre-enriched), and `naeb_full` (a GBIF North American pool labelled with NAEB Drug use: the Moerman used-versus-available estimand). Landscape sizes and composition are given in Table 1.

### 4.3 Reward construction
ChEMBL reward is normalised pChEMBL potency; NAEB reward is log use-intensity. Sparsity is imposed by `reward_top_frac`, which retains the top fraction of species by reward. `[CONFIRM exact pChEMBL normalisation and log-use definition]`

### 4.4 Cue construction
Four cue channels: sensory salience derived from Duke chemistry; phylogenetic similarity from the Open Tree; ecological co-occurrence as a Jaccard graph from GBIF; and animal-association as a Jaccard graph from GloBI. For graph cues, the per-species cue value used in coupling tests is a neighbour-weighted reward (a Moran's-I-style autocorrelation of reward over the relevant graph).

### 4.5 Coverage covariates
Cue-side coverage is captured by `occ_count`, `interaction_count`, and `chem_count` (log1p), used as partialling covariates.

### 4.5b Reward-depth covariates
Reward-side effort is captured by NAEB documentation depth and a `has_chem_record` indicator (and, for ChEMBL, compound-bridge breadth). Reward residualisation regresses reward on these depth covariates; the residual is denoted reward\*.

### 4.5c Sensory missingness
Only 52 of 349 full-flora species have Duke chemistry records; the remaining 297 receive zero-imputed sensory salience (effective n retained at 349). Because `has_chem_record` is a near-binary indicator dominating sensory variance, it is included in reward-depth residualisation so that "has any chemistry record" is controlled, not only "how deep."

### 4.6 Coupling measurement
Univariate coupling is a Spearman statistic judged against a label-permutation null. Coverage control (`--control`) uses partial rank correlation given the coverage covariates; reward residualisation (`--residual-reward`) repeats the test on reward\*. Multivariate coupling is a partial-R² of the four-channel cue vector against reward\* with block-permutation significance.

### 4.7 Search strategies
Six sequential strategies select the next species to test from the untested pool: **random** (uniform); **phylogenetic**, **sensory**, **ecological**, and **animal-association**, each greedily ranking by the corresponding cue (graph cues use neighbour-weighting over discovered reward); and an integrative **cultural/social** strategy that combines cue channels. `[CONFIRM exact ranking and online-update rule per strategy, and the precise composition of the cultural/social strategy: these determine the M2 ladder interpretation and should be stated explicitly.]`

### 4.8 Simulation protocol
Each run fixes a budget below pool size (required for differentiation; H1.2), accumulates reward to compute the area under the discovery curve (AUDC) and regret, and is repeated across 30 search-seed replicates. `[CONFIRM replicate count and budget per landscape.]`

### 4.9 Negative controls
The control harness includes `permute_reward` (uniform label permutation), graph rewiring, and phylo shuffle. The M2 decomposition ladder adds `permute_features` (shuffle cue values) and `permute_reward_within_effort` (permute reward within effort strata defined by an `effort_index` summing `chem_count` and `has_chem_record`).

### 4.10 Software and reproducibility
All landscapes, configs, and frozen caches are released; every figure and table is regenerable from a single CLI. `[TODO: repository link.]`

---

## 5. Results

We present results in hypothesis-ledger order.

### 5.1 Instrument validation
Planted coupling is detected by the coupling instrument; the `permute_reward` control zeroes structured advantage; cue-specific nulls zero the strategies that depend on that cue; and `reward_top_frac` imposes sparsity without destroying planted coupling among survivors (synthetic tests). These pass, so live claims are admissible.

### 5.2 Landscape characterisation
The four landscapes (Table 1) differ sharply in seed bias. The regional GBIF pool overlaps the curated medicinal pool in 1 of 40 species, versus ~100% overlap for the chemistry-seeded pool: confirming that regional seeding does not silently reconstruct a medicinal-enriched flora (H1.1, IDENTIFIED).

**Table 1: Landscape comparison.**

| | Curated | Region | NAEB enriched | NAEB full flora |
|---|---|---|---|---|
| n | 66 | 149 | 198 | 349 |
| Seed | Duke mirror | GBIF Amazonia | NAEB documented | GBIF North America |
| Reward | ChEMBL | ChEMBL (LOTUS) | NAEB Drug use | NAEB Drug use |
| n useful (base) | dense | 43 (29%) | ~99% medicinal | 121 (35%) |
| Phylo coupling | ~null | ~null | ~null | ~null |
| Raw coupling | weak | strong | weak (sensory) | strong (eco/sens/bio) |
| Controlled + residual | - | collapsed | collapsed | eco p = 0.053 (boundary) |
| Multivariate partial R² | - | 0.18 (p = 0.31) | 0.03 (p = 0.97) | 0.030 (p = 0.51) |
| Structured vs. random | modest + | large + | moderate + | moderate + (sensory) |
| M2 genuine component | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| Sensory genuine (TOST, matched @0.2) | - |: | +0.017 (equiv.) | −0.045 (non-eq. −) |
| Sensory genuine (TOST, matched @0.35) | - |: | +0.001 (equiv.) | −0.046 (non-eq. −) |
| Real advantage (matched density) | - |: | see §5.5 | enriched ≈ full at top_frac 0.2–0.35 |
| Real advantage (unmatched M2) | - |: | default top_frac 0.25 | 43–66% lower, density-confounded |
| Keystone stratified FP | 2/16 `[CONFIRM source]` | - |: | raw 3/16; ctrl+resid 1/16 |
| High-doc eco decomposition | - |: | - | depth confound; redundancy bin 3 only |
| Sparsity at top_frac = 0.1 | inverts | survives | survives | survives; sensory +0.144 |

### 5.3–5.4 Measured coupling, raw and controlled
Raw cue–reward coupling is strong on the species-rich landscapes (regional and full flora) and weak on the curated and enriched pools. Phylogenetic coupling is null on every landscape, raw and controlled (H2.1, H2.2). For the remaining channels, raw significance does not survive coverage control on ≥3 of 4 channels on both the regional and NAEB landscapes, supporting the confound hypothesis directly (H2.6, IDENTIFIED): raw database coupling largely reflects study effort.

*Figures 2 (raw coupling, three/four landscapes) and 3 (raw vs. coverage-controlled, paired) go here. Data ready.*

### 5.4a Keystone coupling (Moerman full flora)
Raw coupling on the 349-species GBIF pool was strong for sensory (ρ = 0.30, p < 0.001), co-occurrence (ρ = 0.48, p < 0.001), and animal association (ρ = 0.42, p < 0.001), replicating the study-effort mirage seen on the Amazonian regional landscape. Under symmetric confound control: partial rank correlation on occurrence, interaction, and chemistry counts plus reward residualisation on NAEB documentation depth and chemistry-record coverage: multivariate partial R² was 0.030 (p = 0.51, n = 349). There is no robust, independent co-occurrence signal: global univariate eco coupling sits on a specification boundary (p = 0.037 before adding `has_chem_record`; p = 0.053 after), not in clear null territory, but the association is not stable to covariate choice and vanishes once other cue channels enter a joint model.

Depth-stratified coupling supports this reading and carries its own false-positive statistic. We report it two ways to avoid a shifting denominator. On the cells with raw reward variation (12 cells, bins 2–4 × four channels), raw fires on 3 (P ≈ 0.02 under a binomial null at α = 0.05). On the common 16-cell grid (bins 1–4 × four channels, anchored to the controlled+residual informative set), raw fires on 3 (P ≈ 0.04) and controlled+residual on 1 (P ≈ 0.56). Bin 1 is vacuous at raw (reward range = 0) but informative after residualisation, because residualisation induces reward variance where raw reward was constant; including it in the common grid therefore makes the raw count conservative rather than inflated. Vacuous bin 0 is excluded throughout so guaranteed-null cells do not deflate the false-positive rate. The mirage-versus-null contrast holds under either denominator.

### 5.5 Strategy comparison and the M2 ladder
Structured strategies beat random on the species-rich landscapes (M1). Phylogenetic search is ≤ random everywhere, consistent with the phylogenetic null (M3). The puzzle (M2) is that structured strategies beat random even where single-channel controlled coupling is null.

The M2 ladder resolves this. Advantage collapses under uniform reward permutation (the leakage audit passes) and under feature permutation, but survives reward permutation *within effort strata* (+0.05–0.10 AUDC on the full flora; +0.06–0.14 on the enriched pool). The paired effort-independent ("genuine") component: advantage(real) − advantage(effort-stratified null): is therefore at or near zero. We conclude that structured-search advantage on real landscapes is feature-mediated effort-tracking: strategies recover the effort surface (well-documented, chemically profiled, reward-dense species) rather than exploiting effort-independent cue–reward structure (Arc B: strategy efficiency factors through encounter/effort geometry).

**Reward density, not composition alone, drives apparent benchmark gaps.** Moving from the NAEB-documented seed pool at its default top_frac=0.25 to the full-flora estimand at its natural plateau once suggested a 43–66% shrink in real AUDC advantage over random. The matched-density sweeps show that this comparison stacked pool composition with reward sparsity: full-flora sensory advantage rises from +0.052 at top_frac ≥ 0.35 to +0.098 at 0.2 and +0.144 at 0.1, and enriched versus full-flora raw sensory curves nearly overlap at matched density (+0.099 vs. +0.098 at 0.2; +0.054 vs. +0.052 at 0.35). We therefore treat the 43–66% figure as an unmatched-density cautionary comparison, not as load-bearing evidence that composition alone manufactures strategy wins.

The paired genuine component, gated by a two-one-sided-tests (TOST) equivalence test at a pre-specified SESOI of δ = 0.03 AUDC, resolves a different question. At matched density, enriched sensory genuine collapses to practical equivalence (+0.017 at top_frac=0.2; +0.001 at 0.35), while full-flora sensory genuine remains non-equivalent negative (−0.045 at 0.2; −0.046 at 0.35). The robust result is therefore not a positive-to-negative sign flip; it is a null-versus-negative contrast in which the full-flora sensory channel is slightly anti-predictive at fixed effort. Scope caution: the sensory channel rests on 52/349 chemistry-covered species (85% zero-imputed); the effort stratification for the ladder includes both `chem_count` and `has_chem_record`, matching the coupling residualisation surface. We report this as a composition/chemistry-coverage limitation on sensory-channel M2 inference, not as evidence for effort-independent sensory coupling.

*Figures 4 (strategy AUDC) and 4b (M2 ladder / genuine component) go here. Data ready.*

### 5.6 Sparsity sweeps
On the curated ChEMBL pool, structured advantage *inverts* under potency-tail sparsity (best structured AUDC < random at top_frac ≤ 0.25; phylogenetic search underperforms random; EXTRAPOLATION, n ≈ 66). On the NAEB documented pool the advantage *persists* under sparsity (top_frac = 0.1; CARTOGRAPHY, n = 198). On the Moerman full-flora pool, structured advantage also rises as reward density falls (sensory +0.052 at top_frac ≥ 0.35, +0.098 at 0.2, +0.144 at 0.1). The sign difference between curated potency-tail and use-derived landscapes at matched density identifies the sparsity effect as reward-provenance dependent rather than universal (H3.3). *Figures 5 and 5b. Data ready.*

### 5.7 High-documentation eco and label noise
The full-flora coupling null could be a Type-II artifact of false-negative labels: 228 pool species lack NAEB Drug records, and absence of documentation is not absence of use, with mislabelling effort-correlated. We tested this by restricting depth-stratified coupling to high-documentation bins (3–4), where falsely labelled species are least prevalent. Controlled-plus-residual eco coupling was null in both bins (p = 0.35, p = 0.12), retiring the label-noise objection regardless of mechanism.

Decomposing the layers clarifies the mechanism (Table 2). Univariate eco was significant under coverage control alone (bin 3 p = 0.013, ρ = 0.31; bin 4 p = 0.003, ρ = 0.53) but not after reward-depth residualisation: a depth confound in both bins, eco tracking documented use via within-bin documentation depth rather than independent co-occurrence. Multivariate coverage-controlled R² was small and non-significant at bin 3 (R² = 0.09, p = 0.59), supporting cross-channel redundancy there; at bin 4 (R² = 0.31, p = 0.058) the test is underpowered: a sizable variance fraction missing significance on power, not evidence for redundancy. Marginal eco is therefore specification-sensitive globally; depth confound kills it in both bins; redundancy is supported only where the multivariate test had power.

**Table 2: High-documentation eco decomposition (full flora, bins 3–4).**

| Layer | Bin 3 | Bin 4 | Claim enabled |
|---|---|---|---|
| Controlled (univariate eco) | p = 0.013, ρ = 0.31 | p = 0.003, ρ = 0.53 | Marginal eco under coverage partial only |
| Controlled + reward-residual | p = 0.35 | p = 0.12 | Label-noise-robust null; depth confound |
| Multivariate controlled | p = 0.59, R² = 0.09 | p = 0.058, R² = 0.31 | Redundancy (bin 3); underpowered (bin 4) |

---

## 6. Discussion

### 6.1 Saslis-Lagoudakis, H2.1, and H2.2: an estimand split
Saslis-Lagoudakis et al. (2012) show that used species cluster in phylogenetic clades across floras, that cross-cultural hot nodes recur, and that hot-node species are enriched in known bioactives versus random draws. That is a statement about clade-level non-randomness of *use* and node-level enrichment, not about whether pairwise phylogenetic proximity predicts continuous measured potency or per-species use intensity on a fixed pool after effort control.

Our instrument tests the latter estimand, and on it both H2.1 (phylo → ChEMBL potency) and H2.2 (phylo → NAEB use) fail. This is compatible with Saslis-Lagoudakis rather than a contradiction of it, as the three estimands are distinct:

| Estimand | Statistic | Saslis-Lagoudakis 2012 | Green Hypercube (full flora) |
|---|---|---|---|
| Clade / hot-node clustering of documented use | NRI, NTI (permutation-rank); hot-node FP context | Positive (3 floras) | Global borderline NRI = 1.94 (p = 0.06), effort-null −0.23; local 24/562 ≤ E[FP] ≈ 28, FDR = 0 at raw |
| Pairwise phylo proximity → continuous use intensity | Neighbour-weighted reward + permutation | Not tested | H2.2 fail after control |
| Pairwise phylo proximity → measured potency | Same | Hot-node enrichment only | H2.1 fail after control |

We ran the clade-level estimand directly. All NRI/NTI p-values are two-sided permutation-rank statistics, ((1 + #{null ≥ observed}) / (n_perm + 1)); the NRI/NTI indices are z-like summaries of null MPD/MNTD but the p-values are not normal z→p conversions.

**Global signal is effort-driven.** Raw labels show a borderline, Milliken-shaped pattern (NRI = 1.94, permutation-rank p = 0.06; NTI = 0.88, not significant): a weak directional analogue, not a robust replication. The well-powered test is the effort-matched null over the full set of used species: it drops NRI to −0.23 (p = 0.81), eliminating the global clustering. A second, corroborating but underpowered arm (documentation-residual labels (n = 44 used)) yields NRI = 1.19 (p = 0.23). The borderline deep-clade clustering at raw is thus explained by study effort.

**Local enrichment is multiplicity noise, present before any effort adjustment.** Hot-node tests apply the same false-positive discipline as the keystone grid, over 545 internal clades + 17 genera (562 tests per label). At raw, 24/562 nominally enriched clades sit *below* the chance expectation E[FP] ≈ 28.1, and zero survive Benjamini–Hochberg correction. There was no clade-level enrichment to remove: naive uncorrected counting would report "24 enriched clades," and proper accounting erases them. This is a second instance of the paper's core disease: apparent structure that vanishes under correct accounting: now at the clade-enrichment estimand, paralleling the raw-coupling mirage (3 cells → 1 under control). The residual-label arm yields 12/562, also below chance (and ≈ 3 SD below the null expectation, as expected from conservative, lower-variance permutation on thinned labels; FDR = 0 either way).

Milliken et al. (2021) report a much stronger global shape (NRI = 8.66) without effort control. That magnitude gap is consistent with, but not proof of, confound inflation in uncontrolled NRI estimates, since NRI also scales with flora size and tree resolution; we do not attribute the gap to effort alone. Pairwise H2.2 remains null, and even a clade-targeting strategy could not exploit hot nodes here, because false-positive accounting rules out real hot nodes. There is thus no exploitable NAEB phylogenetic signal at the global (NRI/NTI under effort control), local (FDR), or pairwise (H2.2) estimand.

### 6.2 The study-effort mirage
Our central finding is not that databases contain no biological information, but that integrated coupling measured on database fields largely reflects who was studied, not latent ecology independent of attention. On the keystone landscape, raw depth-stratified coupling fires on 3 of 16 common-grid cells (P ≈ 0.04) while controlled-plus-residual yields 1 of 16 (P ≈ 0.56). The phylogenetic re-analysis shows the same disease at a different estimand (24/562 raw hot nodes vs. E[FP] ≈ 28, FDR = 0). Souza et al. (2018) provide an independent example of the same mechanism, with hot-node clades attracting more use reports and screening while drug-yielding genera sit largely outside them. Practitioners who report only raw coupling or uncorrected clade counts will systematically overstate the exploitable signal available to search algorithms.

### 6.3 The M2 decomposition and pre-enrichment
Structured strategies beat random without passing controlled coupling tests because they recover the effort surface, not because they exploit effort-independent structure. The effort proxy is coarse (record counts, not a latent attention model); for coupling *detection* that risks over- or under-correction, but for the genuine-component estimate it is conservative, since any unmodelled effort dimension would survive within-effort permutation and inflate apparent genuine signal, and apparent genuine is already at or below zero. Sparsity sweeps strengthen the benchmark lesson: structured advantage is strongly reward-density dependent, and enriched versus full-flora NAEB raw benchmarks converge at matched top_frac. The historical 43–66% enriched-to-full shrink is therefore a cautionary result about density-mismatched benchmark design, not evidence that composition alone manufactures wins. P8 shows that the enriched sensory genuine component is practically equivalent to zero at matched density, while the full-flora sensory component remains non-equivalent negative. The negative value is not "no effect" but anti-prediction at fixed effort: salience ranks chemically catalogued plants, not adopted ones, within a severely sparse chemistry-coverage regime.

### 6.3b Eco on the keystone landscape
Co-occurrence is the channel most expected to survive on a Moerman estimand, and global univariate eco after full control sits at p = 0.053: easy to misread as marginal significance. The accurate statement is that there is no robust, independent eco signal: multivariate joint coupling is null (R² = 0.030, p = 0.51); high-documentation stratification shows eco dies under reward-depth residualisation in both bins (label noise retired); depth confound explains the controlled univariate signal; and multivariate redundancy is supported only at bin 3, bin 4 being underpowered rather than null.

### 6.3c Chemistry missingness
Because 85% of full-flora species lack Duke chemistry, the sensory channel is dominated by a coverage indicator. Controlling `has_chem_record` on the reward side and including `chem_count` in the effort stratification means sensory results are interpreted on the ~52 informative species; this is a scope limit on sensory conclusions, not a confound left uncontrolled.

### 6.4 Sparsity × provenance
The curated inversion and the NAEB persistence under sparsity are consistent with the apparency literature in which medicinal use is among the weakest availability–use categories (Gonçalves et al., 2016), though that literature is mixed and we do not claim a universal law. The provenance dependence (curated potency-tail vs. documentation-derived use) is the operative variable.

### 6.5 Implications for bioprospecting
Phylogeny is not justified as a prioritisation cue for measured potency at the species scale on these landscapes, and raw integrated coupling should not be read as ecological signal. A coverage-first discipline: control study effort symmetrically before crediting any cue: is the practical recommendation.

### 6.6 Indigenous knowledge framing
NAEB aggregates Indigenous knowledge compiled from many sources; aggregate analysis does not confer representativeness, and the pool, geography, and provenance limit any claim about Indigenous discovery mechanisms. We frame data handling against the CARE principles (Carroll et al., 2020) and treat sovereignty and representativeness as limitations, not strengths. `[CONFIRM: add explicit CARE compliance paragraph and any community-engagement statement.]`

### 6.7 Limitations
Single regions per landscape; no holdout region or multi-landscape replicate stability yet (M5). ChEMBL reward is near-deterministically correlated with chemistry-study depth (ρ ≈ 0.95), so its controlled/residual results are close to unidentifiable: though the ChEMBL phylogenetic null is protected because it is null even at raw, before residualisation. Partial-rank correlation with collinear effort proxies can over- or under-correct (a composite effort index is planned). Sensory coverage is 15% of the full flora. Matched-density raw comparisons should be reported with uncertainty intervals rather than interpreted from point estimates alone.

### 6.8 Future work
Holdout-region generalisation; PU-learning treatment of the unlabelled class; a composite effort index; multi-landscape replicate stability; uncertainty intervals for matched-density strategy gaps; and Sobol variance decomposition for the gate framework.

---

## 7. Conclusion

Across curated, regional, and Moerman full-flora landscapes, cue–reward coupling in database-integrated ethnobotany is largely a coverage artifact unless study effort is controlled symmetrically on cues and rewards. Raw coupling is statistically detectable (keystone 3/16 common grid) and misleading (1/16 controlled+residual); global multivariate coupling is null (R² = 0.030, p = 0.51). Structured-search advantage is strongly reward-density dependent, and matched-density NAEB sweeps show that raw enriched and full-flora benchmarks largely converge; unmatched benchmark gaps should therefore be treated as density-confounded unless top_frac is aligned. The M2 ladder identifies surviving wins as effort-surface tracking, with enriched sensory genuine practically equivalent to zero and full-flora sensory genuine slightly negative at matched density under severe chemistry missingness. A phylogenetic re-analysis shows borderline clade clustering of use is effort-driven and apparent hot-node enrichment is multiplicity noise. We recommend reporting raw and controlled coupling, and, where reward is documentation-derived, reward-residualised coupling: in every computational ethnobotany prioritisation study, matching reward density before comparing benchmarks, and treating pre-enriched pools as stress tests rather than unbiased estimands. Code, configurations, and frozen caches are available for reproduction.

---

## References

*(From the audited bibliography. Entries marked `[CONFIRM]` need a final field check before submission.)*

- Beck, J. et al. (2014). Spatial bias in the GBIF database and its effect on modeling species' geographic distributions. *Ecological Informatics*. `[CONFIRM vol/pp]`
- Blanchet, F. G., Cazelles, K., & Gravel, D. (2020). Co-occurrence is not evidence of ecological interactions. *Ecology Letters*, 23, 1050–1063.
- Carroll, S. R. et al. (2020). The CARE Principles for Indigenous Data Governance. *Data Science Journal*, 19, 43.
- Domingo-Fernández, D. et al. (2023). [Large-scale ethnobotany informatics]. *iScience*, 26, 107729. *(cited as contrast)*
- Gonçalves, P. H. S. et al. (2016). [Availability and plant use: meta-analysis]. *Ecological Applications*, 26, 2238–2253. doi:10.1002/eap.1364
- Grimbs, A. et al. (2017). [Phytochemistry/bioactivity and phylogeny in Rhododendron]. *Frontiers in Plant Science*, 8, 551.
- Guèze, M. et al. (2014). [Use value and abundance among the Tsimane']. *Economic Botany*, 68, 1–15.
- Hortal, J. et al. (2015). Seven Shortfalls that Beset Large-Scale Knowledge of Biodiversity. *Annual Review of Ecology, Evolution, and Systematics*, 46, 523–549.
- Milliken, W. et al. (2021). [Phylogenetic clustering of Latin American antimalarials]. *Journal of Ethnopharmacology*, 279, 114221. doi:10.1016/j.jep.2021.114221
- Moerman, D. E. (1991). The medicinal flora of native North America: an analysis. *Journal of Ethnopharmacology*, 31, 1–42.
- Phillips, O., & Gentry, A. H. (1993). The useful plants of Tambopata, Peru: statistical hypotheses tests with a new quantitative technique. *Economic Botany*. `[CONFIRM 1993a/b]`
- Pirolli, P., & Card, S. (1999). Information foraging. *Psychological Review*. `[CONFIRM]`
- Poelen, J. H., Simons, J. D., & Mungall, C. J. (2014). Global Biotic Interactions. *Ecological Informatics*, 24, 148–159.
- Reyes-García, V. et al. (2009). [Cultural transmission of ethnobotanical knowledge]. `[CONFIRM venue/year]`
- Richard-Bollans, A. et al. (2023). Machine learning enhances prediction of plants as potential sources of antimalarials. *Frontiers in Plant Science*, 14, 1173328.
- Rønsted, N. et al. (2012). [Phylogenetic signal in Amaryllidaceae use vs. bioactivity]. *BMC Evolutionary Biology*. `[CONFIRM vol]`
- Rutz, A. et al. (2022). The LOTUS initiative for open knowledge management in natural products research. *eLife*, 11, e70780.
- Saslis-Lagoudakis, C. H. et al. (2012). Phylogenies reveal predictive power of traditional medicine in bioprospecting. *PNAS*, 109(39), 15835–15840.
- Souza, E. N. F. et al. (2018). [Hot nodes, screening, and FDA-drug genera]. *Frontiers in Plant Science*, 9, 834.
- Srinivas, N. et al. (2010). Gaussian process optimization in the bandit setting. `[CONFIRM]`

---

## Author-facing notes (delete before submission)

- **Abstract** rewritten to ~250 words, granular statistics moved to Results, and **four landscapes** named consistently (the earlier "three" undercount is fixed).
- **Keystone FP** reported on both the informative set (3/12, P ≈ 0.02) and the common grid (3/16 vs 1/16), with the bin-1 layer-dependent-vacuity note so neither a "shifting denominator" nor a "padded raw" objection lands.
- **Global phylo effort claim** led by the well-powered effort-matched null; the residual-label NRI (n = 44) is flagged as corroborating-but-underpowered.
- **ChEMBL identifiability ceiling** stated in §6.7; the rainforest framing is marked motivational against the North-American keystone.
- **Open `[CONFIRM]`:** the six strategy definitions and the cultural/social composition (§4.7); pChEMBL/log-use formulas (§4.3); replicate/budget counts (§4.8); cache schema (§4.1); the curated 2/16 FP cell source (Table 1); and a handful of reference fields (§References).
- **Open `[TODO]`:** Figures 1 and 6; uncertainty intervals for matched-density raw strategy gaps; explicit CARE/community paragraph; repository link; author/affiliation.
