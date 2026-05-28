# Statistical & Technical Glossary

A reference for all statistical tests, metrics, and technical terms used in the
sheep foraging behaviour presentation.

---

## Statistical Tests

### Kruskal–Wallis H-test (KW)

**What it is.** A non-parametric test for comparing distributions across 3 or
more independent groups. It is the rank-based analogue of one-way ANOVA.

**When to use.** When you want to test whether at least one group differs from
the others, but your data are ordinal, non-normal, or have unequal variances.
We use it to test whether a metric (e.g., completion time) differs significantly
across assay levels 0–7.

**How it works.** All observations are ranked regardless of group. The test
statistic H measures whether the mean ranks differ between groups more than
expected by chance.

**Interpretation.**
- Large H → groups differ (reject H₀ that all groups come from the same distribution).
- Reported as: `KW H = 14.2, p = 0.003`
- KW does *not* tell you *which* groups differ — use pairwise follow-up tests for that.
- In this presentation, KW is used as an omnibus test across assay levels.

**Assumptions.**
- Independent observations (violated here due to repeated groups — see *pseudoreplication*).
- At least 5 observations per group recommended.

---

### Mann–Whitney U test (MWU)

**What it is.** A non-parametric test comparing two independent groups. The
rank-based analogue of the independent-samples t-test.

**When to use.** Pairwise comparisons between two specific conditions (e.g.,
baited vs. unbaited sites, assay 0 vs. assay 7).

**How it works.** Ranks all observations from both groups together. The U
statistic counts how often a value in group A exceeds a value in group B. If U
is extreme (high or low), the groups differ.

**Interpretation.**
- Reported as: `MWU p = 0.012`
- p < 0.05 suggests the two distributions differ.
- Two-sided test used here (no directional hypothesis assumed).

**Note on multiple comparisons.** When multiple MWU tests are performed (e.g.,
all pairwise assay comparisons), the chance of at least one false positive
inflates. This presentation uses uncorrected MWU tests — results are exploratory.

---

### Spearman Rank Correlation (ρ)

**What it is.** A non-parametric measure of the strength and direction of a
*monotonic* relationship between two variables.

**When to use.** To test whether a metric increases (or decreases) consistently
with assay level without assuming linearity.

**How it works.** Ranks both variables independently, then computes Pearson's r
on the ranks.

**Interpretation.**
- ρ = +1: perfect monotonic increase
- ρ = -1: perfect monotonic decrease
- ρ = 0: no monotonic trend
- Reported as: `ρ = 0.72, p = 0.003`
- More robust than Pearson's r to outliers and non-linear monotonic patterns.

**In this presentation.** Used to test whether metrics (sites found, coverage,
straightness, giving-up time) trend monotonically with assay level.

---

## Descriptive Statistics

### Median

**What it is.** The middle value when observations are sorted. 50% of data
falls above, 50% below.

**Why use median over mean.** Medians are resistant to outliers and skew. With
small samples (n = 7–12 per assay), a single extreme trial can distort the mean.
All summary statistics in this presentation use medians.

---

### IQR (Interquartile Range)

**What it is.** The range between the 25th percentile (Q1) and 75th percentile
(Q3): IQR = Q3 − Q1. It captures the middle 50% of the data.

**Interpretation.** Reported as `[Q1, Q3]` in the summary table. A narrow IQR
means the data is tightly clustered; a wide IQR indicates high variability.

**Relation to box plots.** The box in a box plot spans the IQR. The line inside
is the median. Whiskers extend to 1.5×IQR (or data extremes if smaller).

---

### SEM (Standard Error of the Mean)

**What it is.** A measure of how precisely the sample mean estimates the
population mean: SEM = SD / √n.

**Interpretation.** Smaller SEM → more precise estimate. SEM decreases with
larger sample sizes. In the flocking time-series plots, the shaded bands show
±1 SEM around the mean curve.

**SEM vs. SD.**
- SD = spread of individual observations
- SEM = precision of the estimated mean
- SEM is always smaller than SD (by factor √n)
- Use SD to show data variability; use SEM to show estimation uncertainty.

---

### R² (Coefficient of Determination)

**What it is.** The proportion of variance in the dependent variable explained
by the model. Ranges from 0 (model explains nothing) to 1 (perfect fit).

**In this presentation.** Used to evaluate the exponential decay fit to
completion time vs. assay level. R² = 0.93 means 93% of the variation in
median completion time across assays is captured by the exponential model.

**Interpretation.**
- R² > 0.9: excellent fit
- R² = 0.7–0.9: good fit
- R² < 0.5: poor fit
- R² alone does not prove a causal relationship.

---

### n.s. (Not Significant)

**What it means.** The test statistic did not reach the conventional threshold
(typically p > 0.05). We cannot reject the null hypothesis.

**Caution.** n.s. ≠ "no effect." It means insufficient evidence given the
sample size. With n = 12 per group, statistical power is limited.

---

### p-value

**What it is.** The probability of observing a test statistic as extreme as
the one computed, *assuming the null hypothesis is true.*

**Common misinterpretation.** A p-value is NOT the probability that H₀ is true.
It's the probability of the data given H₀.

**Thresholds (conventional):**
- p < 0.05: "significant" (reject H₀)
- p < 0.01: "highly significant"
- p < 0.001: "very highly significant"

**In this presentation.** Reported as exact values (e.g., p = 0.003) rather
than asterisks. Results are exploratory — no formal multiple-comparison correction.

---

## Metric Definitions

### Shannon Entropy (H) / Normalised Entropy (H_norm)

**What it is.** A measure of uncertainty or "evenness" in a distribution.
Originally from information theory.

**Formula.**
```
H = -Σ pᵢ · ln(pᵢ)
```
where pᵢ is the fraction of time sheep i leads the group.

**Normalisation.** Divided by ln(n) to scale 0–1 regardless of group size:
```
H_norm = H / ln(n)
```

**Interpretation.**
- H_norm = 1.0: leadership perfectly egalitarian (all sheep lead equally)
- H_norm = 0.0: one sheep leads 100% of the time (dictator)
- H_norm ≈ 0.7–0.9: mostly shared leadership with slight preferences

**In this presentation.** Measures how evenly frontal-position leadership is
distributed across sheep within a group, computed only during periods when the
group centroid speed exceeds 0.5 m/min.

---

### NND (Nearest-Neighbour Distance)

**What it is.** The minimum distance between any two sheep in the group at a
given time point. Averaged over time to give a trial-level summary.

**Interpretation.**
- Low NND: sheep are close together (tight cohesion)
- High NND: at least one pair is far apart
- Stable NND across assays = group structure doesn't change with learning

**Units.** Reported in metres (grid units × 10).

---

### Straightness Index

**What it is.** The ratio of beeline (straight-line) distance to actual path
length between two points.

**Formula.**
```
SI = beeline distance / cumulative path length
```

**Interpretation.**
- SI = 1.0: perfectly direct path (no detours)
- SI → 0: highly tortuous path (many turns and loops)
- Increasing SI across assays suggests route learning

**In this presentation.** Computed from the trial start position to the first
baited reward site found.

---

### Exponential Decay Model

**What it is.** A model where a quantity decreases rapidly at first, then
levels off at an asymptote.

**Formula.**
```
y = a · exp(-b · x) + c
```
- a = initial drop magnitude
- b = decay rate (higher = faster learning)
- c = asymptotic floor (performance plateau)

**In this presentation.** Fitted to median completion time vs. assay level.
The decay rate b = 0.95 quantifies how rapidly groups learn to find baited
sites. The plateau c ≈ 3 min represents the best achievable performance.

---

### Marginal Value Theorem (MVT)

**What it is.** A classic optimal foraging theory model (Charnov, 1976)
predicting when an animal should leave a patch.

**Prediction.** An optimal forager should leave a patch when the local intake
rate drops below the average environmental rate. In practice: leave depleted
or unrewarding patches sooner.

**In this presentation.** The "giving-up time" at unbaited sites is interpreted
through the MVT lens: experienced sheep who know a site is unrewarding should
leave faster than naïve sheep still assessing it.

---

## Technical Terms

### Affine Transform

**What it is.** A geometric transformation that preserves straight lines and
ratios of distances. Includes translation, rotation, scaling, and shear.

**In this presentation.** Used to project GPS coordinates (latitude/longitude)
onto a 5×5 grid arena coordinate system, using known corner positions (E1–E4)
as control points. A least-squares fit determines the optimal transformation
matrix.

---

### Centroid

**What it is.** The geometric centre (mean position) of all sheep in the
group at a given time point: (mean_x, mean_y).

**In this presentation.** Group velocity and frontal-position leadership are
computed relative to the centroid's movement direction.

---

### Frontal Position (Leadership)

**What it is.** At each time step, each sheep's position is projected onto the
group's direction of travel. The sheep with the highest projection (farthest
ahead in the movement direction) is the "frontal leader."

**Computation.**
1. Compute centroid velocity vector (smoothed over 15 seconds)
2. Normalise to unit direction vector
3. For each sheep: project (sheep_position − centroid) onto direction vector
4. Sheep with maximum projection = leader at that time step

**Threshold.** Only computed when centroid speed > 0.5 m/min to avoid random
assignments during stationary periods.

---

### Pseudoreplication

**What it is.** Treating non-independent data points as independent. Inflates
sample sizes and can produce spuriously significant p-values.

**In this presentation.** The same 12 groups are tested across all 8 assay
levels. Each trial is treated as independent in KW/MWU tests, but within-group
correlation exists (the same sheep appear in multiple trials). This means
p-values are likely too liberal (too small).

**Solution (future work).** Mixed-effects models with group as a random
intercept would properly account for this correlation structure.

---

### Occupancy (Heatmap)

**What it is.** A 2D histogram showing how much time sheep spend in each
spatial bin of the arena. High occupancy = sheep lingered there.

**Log scale.** Applying log1p(counts) compresses the dynamic range, making
low-density regions visible alongside high-density hotspots.

---

### Giving-Up Time

**What it is.** The duration a forager spends at an unrewarding location before
departing. A shorter giving-up time indicates better patch discrimination.

**In this presentation.** Measured as the visit duration at unbaited reward
sites. Analysed across assay levels to test whether experienced groups
recognise and reject unrewarding sites faster.

---

### Multiple Comparisons Problem

**What it is.** When many statistical tests are performed simultaneously, the
probability of at least one false positive increases dramatically.

**Example.** With 8 assay levels, pairwise MWU gives 28 comparisons. At α =
0.05 each, the "family-wise error rate" (probability of at least one false
positive) is 1 − (0.95)²⁸ ≈ 76%.

**Common corrections:**
- **Bonferroni:** Divide α by number of tests (very conservative)
- **Holm–Bonferroni:** Step-down procedure (less conservative)
- **FDR (Benjamini–Hochberg):** Controls *proportion* of false positives

**In this presentation.** No correction applied. Results are explicitly
labelled as exploratory. Significant results should be confirmed with
properly powered, pre-registered follow-up studies.

---

## Summary Table: Tests Used and When

| Symbol | Full Name | Use Case | Null Hypothesis |
|--------|-----------|----------|-----------------|
| KW | Kruskal–Wallis H | ≥3 group comparison | All groups from same distribution |
| MWU | Mann–Whitney U | 2 group comparison | Two groups from same distribution |
| ρ | Spearman rho | Monotonic trend | No monotonic relationship |
| R² | Coefficient of determination | Model fit quality | (descriptive, no test) |
| H_norm | Normalised Shannon entropy | Leadership evenness | (descriptive, no test) |
| SEM | Standard error of mean | Precision of estimate | (descriptive, no test) |

---

## Quick Decision Guide: Which Test to Use?

```
Do you have 2 groups or ≥3?
├── 2 groups → Mann–Whitney U (MWU)
└── ≥3 groups
    ├── Comparing distributions → Kruskal–Wallis (KW)
    └── Testing monotonic trend → Spearman ρ

Is the data normally distributed?
├── Yes (or large n) → Could use parametric (t-test, ANOVA)
└── No / uncertain / small n → Use non-parametric (above)

Do you need to control for repeated measures?
├── Yes → Mixed-effects model (future work)
└── No → KW / MWU are appropriate
```

---

*Generated for the Collective Foraging Behaviour presentation, April 2026.*
