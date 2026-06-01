# Sheep GPS analysis — questions, answers, and proofs

## Executive summary

Sheep groups of four were tracked through foraging trials in a 12-site
arena. Four test configurations (A–D) baited **fixed** triplets of
three sites — the same positions every trial — allowing spatial memory
to accumulate. Two control configurations (CTRL_FAR, CTRL_BARN) baited
**random** site positions each trial, preventing stable spatial
learning. All results below are from Phase 2 (from 17 Feb 2026).

**Spatial individuality (occupancy).** Individual sheep occupy
distinct sub-regions within a trial rather than moving
interchangeably. Mean per-sheep occupancy entropy is significantly
lower than a sheep-ID shuffle null (p ≈ 0.01 per trial).

**Learning to find sites (path length).** Across assays within a
group, sheep find more reward sites with experience (p = 0.003 vs.
within-group assay-label shuffle), with borderline improvement in
completion time (p = 0.042). Total path length does not decrease
significantly — sheep navigate to food better without necessarily
moving less.

**Flocking (group cohesion).** No detectable change in nearest-
neighbour distance or group spread across assays (Spearman ρ ≈ 0,
p > 0.6). Group cohesion is roughly constant across the learning
period.

**Leadership.** Leadership is non-uniform — one sheep tends to occupy
the frontal position disproportionately within each trial (block-
permutation null, 2000 permutations). Continuous leadership scores
show temporal autocorrelation, indicating persistence over seconds
rather than random fluctuation. Rank stability across assays and
leader–pioneer overlap are quantified in the notebook.

**Site discovery effects.** Sheep slow down significantly after
finding a reward site (speed Δ ≈ −2.1 m/min, p ≈ 0), consistent
with stopping to eat. Group spread changes by only ~3 cm —
statistically detectable at n ≈ 950 events but biologically
negligible. Because control sheep also encounter (randomly-placed)
reward, a non-zero Δ in CTRL would confirm the effect is a general
reward-consumption reflex rather than recognition of a learned
location.

**Random-walk null — the key finding.** The clearest test of spatial
learning uses a *baited preference* metric: the fraction of site-
visit time at the fixed baited triplet vs. all 12 sites (spatial
chance = 3/12 = 0.25). After canonical orientation, the baited
triplet maps to positions {A1, A2, A3} for all test configs. Real
sheep in test trials score 0.49–0.60, compared to a simulated
random-walk baseline of ~0.25. Control sheep score ~0.29 ≈ chance:
because CTRL reward positions change each trial, the canonical
positions {A1, A2, A3} accumulate no learned value — confirming the
effect is driven by spatial memory, not by reward presence alone.

---

## Experimental context

GPS collars track sheep through ~35-minute foraging trials in a flat
arena containing 12 reward-site positions on a square grid
(`SITE_GRID` in `gps_analysis.config`). Four test configurations
(A, B, C, D) each bait a **fixed** triplet of three sites — the same
three positions every trial — so sheep can accumulate spatial memory
across assays. Two control configurations (CTRL_FAR, CTRL_BARN) place
reward at **random** site positions each trial; because the baited
triplet changes, no stable spatial memory can develop. Within each
group of animals, successive trials are numbered by `assay` — this
captures learning across days.

The dataset is split into two phases. Phase 1 (pre-2026-02-17) is
protocol-calibration data with variable group sizes. Phase 2 (date ≥
2026-02-17) is the regular experimental cohort: group size 4, consistent
protocol. **All inferential claims below are restricted to Phase 2.**
Every analytical script enforces this either through a UI default or a
hardcoded `_PHASE2_DATE` filter.

Per-config orientation transforms in `CONFIG_TRANSFORMS` rotate or
mirror each test config's tracks so the baited triplet always lands at
the canonical positions {A1, A2, A3} after `apply_orient=True`. This
makes "baited" a well-defined set of grid positions in the oriented
frame without needing per-trial baiting metadata downstream. Control
configurations have no fixed canonical baited set because their reward
positions change each trial; they are excluded from baited-preference
analyses (§6.1) and used as movement baselines elsewhere.

## AI involvement

The analyses in this document were implemented with substantial help
from an AI coding assistant (Anthropic Claude). The AI drafted code,
proposed permutation schemes and null statistics, identified
methodological issues (e.g. the label-asymmetric ratio statistic in
§5.1), ran code locally to produce the numerical results quoted
throughout, and drafted this document. Scientific decisions — what to
test, how to interpret results, which findings to report — remain with
the researcher; all code is in version control for review, and
methodological caveats are flagged at the end. This note exists so
downstream readers can locate AI involvement at the appropriate point
in the chain of evidence.

## How this document is structured

Each subsection follows the same shape:

- **Question.** The scientific question.
- **Answer.** The null hypothesis (H₀) and the alternative that would
  disprove it.
- **Proof.** The statistical procedure (data slice, statistic, null
  construction, current numerical result).

Shared permutation conventions are listed once at the end.

---

## 1. Occupancy (`scripts/occupancy_heatmap.py`)

### Summary of operation

The script opens with UI controls for date range, bin size, and
config/phase filter (Phase 2 default). It loads all matching trials
and builds per-sheep 2D occupancy histograms on a configurable grid.
The main view shows individual-sheep heatmaps for a selected trial.
Three additional analytical panels follow: (i) a Test vs Control
side-by-side aggregate occupancy comparison; (ii) a sheep-ID-shuffle
null that permutes sheep labels within each trial 1000 times, computes
mean per-sheep Shannon entropy each time, and plots the null
distribution against the observed value with a two-sided empirical p;
(iii) a random-walk occupancy comparison that simulates K = 20
correlated random walks per sheep and renders a three-panel
real / simulated / difference heatmap.

### 1.1 Do individual sheep within a trial occupy distinct sub-regions?

**Question.** Within a single trial, are different sheep statistically
interchangeable in where they spend time?

**Answer.** *H₀:* every sheep's GPS points are a random sample from the
trial's pooled occupancy distribution; sheep-IDs are arbitrary labels.
*Reject* if observed mean per-sheep occupancy entropy differs from the
null produced by permuting sheep-ID labels across GPS points within the
trial.

**Proof.** Per Phase 2 trial: build per-sheep 2-D histograms on the
user-selected bin grid; compute Shannon entropy per sheep; mean across
sheep = observed statistic. Null: 1000 within-trial permutations of
sheep-ID labels (seed = 42), recomputing mean per-sheep entropy each
time. Report observed, null mean ± 95% interval, two-sided empirical p
per config. Aggregate entropy is invariant under this permutation, which
is why mean per-sheep entropy is the statistic. *Result:* p_emp ≈ 0.01
per trial across configs — strong rejection.

### 1.2 Do real sheep concentrate near reward sites more than chance?

**Question.** Where in the arena do real sheep spend more (or less) time
than a movement-matched random walker would?

**Answer.** *H₀:* the aggregate test-pool occupancy distribution matches
the distribution produced by per-sheep correlated random walks with the
same step-length and turn-angle distributions and a reflective arena
boundary. *Reject* if the real–sim difference panel shows systematic
spatial structure (hotspots at reward-site positions).

**Proof.** Per-sheep fit step and turn distributions on Phase 2 tracks;
simulate K = 20 walks per sheep from the real start position; aggregate
real and simulated histograms on the canonical (per-config oriented)
grid; render the signed difference with a divergent colormap. Visual /
descriptive — inferential per-scalar-metric tests live in §6.

---

## 2. Path length and discovery efficiency (`scripts/path_length_analysis.py`)

### Summary of operation

Loads all Phase 2 trials (test configs A/B/C/D plus CTRL_FAR/CTRL_BARN).
For each trial, computes: total path length (cumulative Euclidean
distance), completion time (first moment at which N = 3 distinct
sites have been visited within radius 0.5), and total sites found.
Renders boxplots of all three metrics by configuration, with test and
control configurations visually distinguished. The inferential null
restricts to test trials only (A/B/C/D): Spearman ρ is computed
between assay number and each metric per trial, and assay labels are
shuffled within each group_num across 1000 permutations (seed = 42)
to build the null distribution. Two-sided empirical p is reported for
each metric.

### 2.1 Do sheep find sites faster, with shorter paths, and more completely across assays?

**Question.** Within each group, does completion time decrease, path-to-
completion shorten, or sites-found increase monotonically with assay
number?

**Answer.** *H₀:* assay-label carries no information about per-trial
completion time, path length, or sites-found within a group — Spearman ρ
between assay and each metric is zero. *Reject* if observed ρ falls
outside the null distribution generated by shuffling assay labels within
each `group_num`.

**Proof.** Per-trial table on Phase 2 test trials (A/B/C/D, controls
excluded): `group_num`, `assay`, completion_time, path_length,
sites_found (radius 0.5, N = 3 sites). Observed Spearman ρ per metric vs
assay. Null: 1000 permutations (seed = 42) of `assay` within each
`group_num`. Two-sided empirical p = `mean(|ρ_null| ≥ |ρ_obs|)`.
*Initial result:* sites_found p = 0.003, completion_time p = 0.042,
path_length p = 0.069. Sites-found is the cleanest learning signal —
sheep find sites better with assay but don't necessarily move less.

---

## 3. Flocking (`scripts/flocking_dynamics.py`)

### Summary of operation

Opens with a phase/date filter (Phase 2 default). For each trial,
computes two cohesion metrics at every timestep: nearest-neighbour
distance (NND, minimum pairwise Euclidean distance) and per-sheep
spread from the group centroid (mean distance to centroid). Renders
time-series plots per trial/config. Computes per-trial means and runs
the assay-shuffle Spearman null (1000 perms within group_num, seed = 42)
separately for NND and spread. A separate CTRL comparison panel shows
boxplots of mean NND and spread for test vs. control configurations.
The early-vs-late contrast cell computes Δ = mean(metric, last third)
− mean(metric, first third) per trial, bootstraps the per-config mean
Δ (1000 resamples, seed = 42), and reports 95% CIs. A time-reverse
null overlays the symmetric forward/reverse scatter (points fall on
y = −x by construction).

### 3.1 Does flock cohesion change with assay number?

**Question.** Across assays within a group, does mean nearest-neighbour
distance (NND) or per-sheep spread from centroid trend monotonically?

**Answer.** *H₀:* per-trial mean NND and mean spread are independent of
assay label within each group. *Reject* if Spearman ρ vs assay falls
outside the within-group assay-shuffle null.

**Proof.** Same shuffle scheme as §2.1, applied to per-trial
`mean_nnd_scalar` and `mean_spread_scalar` from the COHESION dict. 1000
perms, seed = 42, two-sided empirical p. *Result:* NND ρ ≈ −0.02
p ≈ 0.85; spread ρ ≈ +0.04 p ≈ 0.68 — no detectable monotonic trend.

### 3.2 Is flocking stationary within a trial?

**Question.** Do sheep cluster more (or less) toward the end of a trial
than the start?

**Answer.** *H₀:* the early-vs-late contrast `Δ = mean(metric, last
third) − mean(metric, first third)` is zero. Mean(NND) and mean(spread)
themselves are trivially invariant under track reversal — they depend
only on the set of positions per timestep — so Δ, not the mean, is the
substantive quantity. *Reject* if per-config bootstrap 95% CI of mean Δ
excludes zero.

**Proof.** Per Phase 2 trial, compute Δ_NND and Δ_spread from the
existing time-series. Bootstrap (1000 resamples, seed = 42) the
per-config mean Δ. The forward / reverse paired scatter is included as
a sanity check — points fall on `y = −x` by construction.

---

## 4. Leadership and influence (`scripts/leader_follower.py`)

### Summary of operation

Phase 2 filter applied (hardcoded `_PHASE2_DATE`). At each timestep,
leadership is assigned to the sheep whose position vector has the
highest projection onto the group centroid-velocity direction (frontal
leader). Per-trial leader-fraction summaries are rendered as bar
charts. Four null tiers test non-uniformity with increasing
autocorrelation awareness (frame χ², run-level χ², per-sheep
binomial, block-permutation — §4.1 reports tier 4). Four further
analytical blocks follow: (a) continuous leadership — per-sheep
cosine similarity to centroid velocity, ACF up to 60 s lag,
persistence time τ; (b) pairwise directional influence — N×N
cross-correlation matrix with circular-shift null (≥5 min rotation,
2000 perms); (c) rank stability — Spearman ρ between per-sheep
leadership fractions across assay pairs within each group, with
sheep-ID shuffle null; (d) leader-vs-pioneer — per-trial Spearman ρ
between leadership fraction rank and pioneer-visit rank, with
per-trial pioneer-rank shuffle null (2000 perms).

### 4.1 Is frontal-position leadership uniform across the group?

**Question.** In a multi-sheep trial, does each sheep take the
frontal-leader role at the chance rate `1/n`, or does one sheep lead
disproportionately?

**Answer.** *H₀:* leader identity at each frame is multinomial(1/n).
*Reject* if observed per-sheep leader-frame counts differ from the null.

**Proof.** Three null tiers with increasing autocorrelation awareness:
(i) frame-level χ² (anti-conservative — ignores autocorrelation);
(ii) collapse leader time-series into runs of constant leader, then
per-sheep binomial test on run counts plus run-level χ²;
(iii) block-permutation empirical p — uniformly shuffle run-level leader
identities preserving the observed run-length distribution (2000 perms,
α = 0.05). Tier (iii) is the reported test.

### 4.2 Is leadership sustained at the individual level?

**Question.** Beyond argmax-of-projection, do specific sheep show
persistent alignment with the group centroid velocity?

**Answer.** *H₀:* per-sheep cos-similarity to centroid velocity is
i.i.d. across timesteps — its autocorrelation function (ACF) should
decay immediately. *Reject* by measuring the lag at which ACF crosses
1/e (persistence time τ) per sheep.

**Proof.** Per Phase 2 sheep-trial, compute cos(velocity, centroid
velocity) with the existing 15-s smoothing. ACF up to 60-s lag,
persistence time τ = first lag with ACF < 1/e. Report distribution of τ
and mean cos by configuration (including CTRL pooled). Visual /
descriptive — a non-zero τ already rejects the i.i.d. null trivially.

### 4.3 Do specific sheep lead specific others?

**Question.** Is there an asymmetric directional influence between pairs
(sheep *i* tends to be followed by sheep *j*) beyond shared group
motion?

**Answer.** *H₀:* time-lagged velocity cross-correlation between any
pair (i, j) is symmetric in sign of lag — no directional influence.
*Reject* if observed `peak xcorr(positive lag) − peak xcorr(negative
lag)` (asymmetry) falls outside the null.

**Proof.** Per trial, build an N × N pairwise influence matrix using
lags from −30 to +30 s. Null: circular-shift each sheep's velocity by a
uniform random lag > 5 minutes before pairing (preserves each sheep's
autocorrelation, destroys alignment). 2000 perms, seed = 42, two-sided
empirical p on per-trial asymmetry magnitude.

### 4.4 Does the same individual lead repeatedly across assays?

**Question.** Within a group, is per-sheep leadership rank correlated
between different assays?

**Answer.** *H₀:* per-sheep leadership rank within (group, assay) is
random — Spearman ρ between any two assays of the same group is zero.
*Reject* if observed mean ρ falls outside the null generated by
sheep-ID shuffles within each (group, assay).

**Proof.** Compute argmax-based leadership fraction per sheep per assay;
build pairwise Spearman ρ matrix per group; aggregate observed mean ρ.
Null: shuffle sheep-IDs within each (group, assay) before re-ranking.
2000 perms.

### 4.5 Are frontal leaders the first to discover sites?

**Question.** Within a trial, does per-sheep frontal-leadership rank
correlate with per-sheep pioneer-visit rank?

**Answer.** *H₀:* leadership rank and pioneer rank are unrelated —
per-trial Spearman ρ = 0. *Reject* if observed ρ distribution falls
outside the per-trial pioneer-rank shuffle null.

**Proof.** Per trial: rank sheep by leadership fraction and by pioneer
visits; observed = Spearman ρ. Null: shuffle the pioneer-rank vector
within each trial. 2000 perms, two-sided empirical p on the
across-trial mean ρ.

---

## 5. Site-discovery effects (`scripts/site_discovery_effects.py`)

### Summary of operation

Two modes. **Single-trial viewer** (interactive, no Phase 2 filter):
user selects any trial; the script renders smooth site-occupancy
probability curves for all 12 sites over time, with first-visit
markers. This is exploratory — any trial may be selected.

**Cross-trial aggregation** (Phase 2, A/B/C/D only): detects
discovery events (dwell ≥ threshold within radius 0.5 of a site
using `detect_site_visits()`); for each event collects 60 s of
speed and group-spread before and after; pools events across all
test trials. A paired coin-flip swap null (1000 perms, seed = 42)
tests whether mean signed Δ = (after − before) is non-zero.
Distribution visualisation renders before/after scatter and Δ
histograms alongside the null band.

**Control placebo baseline**: for each CTRL trial, K random
timestamps are sampled from the empirical distribution of test
discovery times (matching the within-trial time profile). The same
60 s before/after window is applied at those timestamps. This
answers: does a randomly chosen moment in a CTRL trial show the
same Δ as a real discovery event in a test trial? If the Δ in CTRL
is near zero, it confirms the slowing response is specific to actual
site visits, not to any moment in the trial.

### 5.1 Do sheep change behaviour when they find a reward site?

**Question.** Do speed and group spread change in the seconds
immediately after a sheep first dwells at a reward site?

**Answer.** *H₀:* per-event signed difference Δ = (after − before) for
speed and for spread has mean zero. *Reject* if observed mean Δ falls
outside the null produced by coin-flip swap of (before, after) labels
per event.

**Proof.** Pool discovery events across Phase 2 test trials. Window =
60 s before / after each event. Null statistic: signed Δ —
label-symmetric (swap → negation). The original `pct = (a − b)/max(b,
0.01)` statistic was **not** symmetric and produced p = 1 / p = 0
artefacts; that's been fixed. 1000 perms, seed = 42, two-sided
empirical p. *Result:* speed Δ = −2.13 m/min (p ≈ 0; sheep slow to
eat at the site); spread Δ = +0.03 m (statistically significant only
because n = 952 — effect is 3 cm, biologically negligible).

### 5.2 Does the slowing response reflect site-visit behaviour or generic trial dynamics?

**Question.** Is the before/after Δ specific to moments when sheep
actually visit a site, or would a randomly chosen moment in any trial
show the same change?

**Answer.** *H₀:* any moment in a trial produces the same Δ as a
real discovery event — the slowing is driven by time-of-trial
dynamics rather than site-visit behaviour. *Reject* if the observed
Δ in test trials falls outside the distribution of Δ at randomly
sampled timestamps in control trials.

**Proof.** Per CTRL trial, sample K placebo timestamps uniformly from
the empirical distribution of test-trial discovery times (this matches
the within-trial time profile of real events). Apply the same 60 s
before/after window at each placebo time. Plot speed and spread %
change for test (by assay) vs. CTRL placebo side-by-side. Currently
descriptive — no formal inferential comparison. Note: because CTRL
reward positions change each trial and per-trial CTRL baiting maps are
not yet in the data pipeline, we use the placebo approach rather than
detecting real CTRL discovery events.

---

## 6. Random-walk null model (`scripts/random_walk_null.py`)

### Summary of operation

Opens with UI controls for phase filter (Phase 2 default), trial time
window (0–35 min), and K (number of simulated walks, default 50).
Loads Phase 2 tracks; decimates from 10 Hz to 1 Hz for speed.
Applies canonical orientation (`apply_orient=True`) so that each test
config's baited triplet maps to {A1, A2, A3}. For each sheep-trial,
fits empirical step-length and turn-angle distributions from the
real 1 Hz track, then simulates K correlated random walks with a
reflective boundary on the [0, 5]² arena grid.

Two families of outputs follow. **General movement metrics** (coverage,
revisit rate, straightness, time-at-any-site): per-sheep real values
are compared to the K-walk distribution; a diagnostic table reports
median z-score and two-sided empirical p grouped by configuration.
**Baited preference** (test configs A/B/C/D only; CTRL excluded because
it has no fixed canonical baited set): computes
`baited_fraction = time_at_baited / (time_at_baited + time_at_unbaited)`
for each sheep and for K simulated walks; renders per-config violin
(sim) + scatter (real), pooled histogram, and summary table.

The analyses below are grouped by *configuration* (A, B, C, D, CTRL) —
not assay number. (An earlier version of this script labelled the
grouping variable `_assay`, which was misleading; it has been renamed to
`_config_group`.)

### 6.1 Do sheep navigate to baited sites?

**Question.** Do sheep spend a disproportionate share of their
site-time at the **baited** triplet relative to the **unbaited** 9
sites?

**Answer.** *H₀:* the share of site-time at baited sites,
`baited_fraction = time_at_baited / (time_at_baited + time_at_unbaited)`,
matches a movement-matched correlated random walk. In the
canonical-oriented frame this null sits near the spatial chance level
3/12 = 0.25. *Reject* if real `baited_fraction` is systematically
above the simulated distribution.

**Proof.** After `apply_orient=True`, every test config's baited triplet
maps to canonical {A1, A2, A3}. Per Phase 2 sheep-trial, compute
`time_at_baited` (fraction of timesteps within radius 0.5 of any of A1,
A2, A3) and `time_at_unbaited` (any of the other 9). The same metric on
K = 50 simulated walks per sheep. Render a per-config violin (sim) +
scatter (real) plus a pooled histogram across A/B/C/D. CTRL is excluded
because it doesn't share the canonical orientation.

*Phase 2 result:*

| Configuration | Real `b/(b+u)` | Sim `b/(b+u)` | Chance |
|---|---|---|---|
| A | **0.53** | 0.25 | 0.25 |
| B | **0.49** | 0.25 | 0.25 |
| C | **0.58** | 0.25 | 0.25 |
| D | **0.60** | 0.25 | 0.25 |
| CTRL | 0.29 ≈ chance | 0.25 | 0.25 |

Sheep in baited configurations spend ~2× chance of their site-time at
the 3 baited sites. CTRL sits at chance — consistent with no stable
spatial memory: because CTRL reward positions change each trial, the
canonical {A1, A2, A3} positions accumulate no learned value for CTRL
sheep. The simulator independently recovers 0.25 across all configs
(sanity check that the canonical site grid is uniform under a random
walker). Note that CTRL sheep do encounter and consume reward — their
chance-level score on this metric specifically reflects the absence of
spatial learning for fixed locations, not the absence of reward.

### 6.2 Do general movement statistics differ from a random walker?

**Question.** Does each sheep behave like a correlated random walker
with its own empirical step-length and turn-angle distribution on
general movement metrics (coverage, revisit rate, straightness)?

**Answer.** *H₀:* per-sheep observed coverage / revisit rate /
straightness is indistinguishable from the same metrics on K = 50
simulated correlated random walks. *Reject* if real metrics fall
outside the simulated 5–95% envelope.

**Proof.** Per Phase 2 sheep-trial: fit step / turn empirical
distributions from 1 Hz–decimated track; simulate K = 50 walks
(vectorised batched simulator with reflective boundary on [0, 5]²);
compute each metric on real and on all K simulated trajectories. The
diagnostic cell reports median z-score `(real − sim_median)/sim_std`,
% of real points outside sim 5–95%, and a two-sided empirical p,
grouped by configuration.

*Phase 2 result (direction-aware):* sheep cover **less** than null
(memory / focus), revisit **more** (consistent), move **more straight**
(clear). General time-at-any-site is slightly above null. The clean
reward-navigation signal lives in §6.1, not in these general movement
metrics.

---

## Methodological conventions

1. **Phase 2 restriction.** Every analytical script enforces `date ≥
   2026-02-17` either through a UI default (`occupancy_heatmap`,
   `flocking_dynamics`, `random_walk_null`) or a hardcoded
   `_PHASE2_DATE` filter (`path_length_analysis`, `leader_follower`,
   `site_discovery_effects`).
2. **RNG and permutation counts.** `np.random.default_rng(seed = 42)`
   throughout; 1000 permutations for univariate Spearman and paired
   tests, 2000 for leadership tests (matching the existing
   binomial-null reference).
3. **Empirical p reporting.** Each test reports observed statistic,
   null mean ± 95% interval (2.5 / 97.5 percentiles), and a two-sided
   empirical p. Asymptotic KW / MWU p-values are not used as the
   primary test.
4. **Autocorrelation-aware nulls.** Time-series statistics use block,
   run, or circular-shift permutations so the null preserves
   autocorrelation. I.i.d. frame shuffles are included as
   anti-conservative reference points only.

## Caveats

1. **Site-discovery permutation, fixed.** The original null used an
   asymmetric ratio statistic `(a − b)/max(b, ε)` which produced p = 1
   (speed) and p = 0 (spread) artefacts. The current code uses signed
   Δ.
2. **General movement directionality.** The one-sided p in the §6.2
   2×2 panel assumes "real > null"; the substantive picture is the
   two-sided diagnostic table.
3. **CTRL canonical-frame interpretation.** Control configurations
   have reward at random positions per trial, so there is no fixed
   baited triplet to orient. CTRL_BARN additionally does not appear
   in `CONFIG_TRANSFORMS`, so its tracks are not rotated at all.
   Both factors mean CTRL is correctly excluded from §6.1. For the
   general movement metrics in §6.2, CTRL tracks are used without
   canonical orientation; absolute spatial comparisons to test-config
   tracks should be made with caution.
