#!/usr/bin/env python3
"""First-order Markov chain of inter-site movement in Phase-2 test trials.

For each Phase-2 test trial we build the ordered sequence of *group-level* site
visits: every (site_label, entry_min) event detected across all sheep, sorted by
entry time. Using canonical oriented labels (after apply_orient=True the baited
triplet is always {A1, A2, A3}) we collapse immediate self-repeats and form a
first-order site -> site transition count matrix pooled across trials. The matrix
is row-normalised to transition probabilities, from which we compute the
stationary distribution and the average (row-weighted) transition entropy.

Two questions are tested with a within-trial visit-order shuffle null
(2000 permutations, rng seed 42):
  1. Is movement between sites structured, i.e. does the observed probability
     mass of transitions INTO baited sites differ from a null that shuffles the
     order of each trial's visit sequence?
  2. Do experienced groups develop preferred baited transitions? We compare the
     early cohort (assay <= 2) against the late cohort (assay >= 5).

Outputs a transition-probability heatmap to analysis/figures/ and prints a
FINDINGS block. Safe to run repeatedly.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from pathlib import Path
from scipy.optimize import minimize_scalar

from gps_analysis import (
    build_trials,
    build_tracks_cache,
    load_trial_tracks,
    detect_site_visits,
    SITE_GRID,
    SITE_LABELS,
    BAITED_CANONICAL,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters (match conventions used elsewhere in analysis/)
# ---------------------------------------------------------------------------
RADIUS = 0.5            # 0.5 grid units = 5 m (generous, for site exploration)
MIN_DWELL_S = 0.0
TEST_CONFIGS = {"A", "B", "C", "D"}
PHASE2_START = "2026-02-17"
CTRL_GROUPS = {9, 14}
N_PERM = 2000
SEED = 42
EARLY_MAX_ASSAY = 2     # early cohort: assay <= 2
LATE_MIN_ASSAY = 5      # late cohort:  assay >= 5

N_SITES = len(SITE_LABELS)
SITE_IDX = {lbl: i for i, lbl in enumerate(SITE_LABELS)}
BAITED_IDX = np.array([SITE_IDX[s] for s in sorted(BAITED_CANONICAL)])
UNBAITED_IDX = np.array([i for i in range(N_SITES) if i not in set(BAITED_IDX)])

# Canonical (oriented) site positions in metres (grid units x 10) and the
# inter-site distance matrix, used to build the geometry / distance-decay null.
POS = np.array([SITE_GRID[lbl] for lbl in SITE_LABELS], dtype=float) * 10.0
_diff = POS[:, None, :] - POS[None, :, :]
DIST = np.sqrt((_diff ** 2).sum(axis=2))           # metres, shape (N, N)
N_BOOT = 2000                                       # bootstrap resamples over trials


# ===========================================================================
# Helpers
# ===========================================================================
def trial_visit_sequence(trial, tracks_cache):
    """Return the ordered list of site labels visited in a trial.

    Pools (site_label, entry_min) events across all sheep, sorts by entry time,
    and collapses immediate self-repeats (consecutive visits to the same site,
    which carry no inter-site transition information). Ties in entry time keep
    a deterministic order (stable sort by time then label).
    """
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks:
        return []
    visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)

    events = []  # (entry_min, site_label)
    for site, vlist in visits.items():
        if site not in SITE_IDX:  # ignore any non-grid labels (e.g. E sites)
            continue
        for v in vlist:
            events.append((v[1], site))
    # Sort by entry time, then label for deterministic tie-breaking.
    events.sort(key=lambda e: (e[0], e[1]))

    seq = []
    for _, site in events:
        if seq and seq[-1] == site:
            continue  # collapse immediate self-repeat
        seq.append(site)
    return seq


def counts_from_sequences(sequences):
    """First-order transition COUNT matrix pooled over a list of label seqs."""
    C = np.zeros((N_SITES, N_SITES), dtype=float)
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            C[SITE_IDX[a], SITE_IDX[b]] += 1.0
    return C


def row_normalise(C):
    """Row-normalise a count matrix; empty rows -> all-zero (no out-transitions)."""
    P = np.zeros_like(C)
    rs = C.sum(axis=1)
    nz = rs > 0
    P[nz] = C[nz] / rs[nz, None]
    return P


def stationary_distribution(P):
    """Stationary distribution via the leading left eigenvector of P.

    Rows with no out-transitions are dropped from the chain (restricted to
    states that emit mass) before solving; if that restricted chain is too
    small we return None.
    """
    active = P.sum(axis=1) > 0
    if active.sum() < 2:
        return None
    Psub = P[np.ix_(active, active)]
    rs = Psub.sum(axis=1)
    nz = rs > 0
    if not nz.all():
        # Re-normalise the restricted matrix so every active row sums to 1.
        Psub = Psub.copy()
        Psub[nz] = Psub[nz] / rs[nz, None]
        Psub[~nz] = 1.0 / Psub.shape[0]
    vals, vecs = np.linalg.eig(Psub.T)
    k = np.argmin(np.abs(vals - 1.0))
    v = np.real(vecs[:, k])
    if v.sum() == 0:
        return None
    v = v / v.sum()
    if (v < -1e-9).any():  # not a valid distribution
        v = np.abs(v) / np.abs(v).sum()
    full = np.zeros(N_SITES)
    full[active] = v
    return full


def transition_entropy(P, C):
    """Mean per-row Shannon entropy (bits), weighted by outgoing-transition count.

    Each from-site's branching entropy is weighted by how often that site is
    actually departed (its row sum in the raw count matrix C), so heavily-used
    sites dominate the average rather than every active row counting equally.
    """
    C_rows = C.sum(axis=1)
    total = C_rows.sum()
    if total == 0:
        return 0.0
    weights = C_rows / total
    ent = 0.0
    for i in range(N_SITES):
        p = P[i][P[i] > 0]
        if p.size:
            ent += weights[i] * (-(p * np.log2(p)).sum())
    return ent


def baited_in_mass(C):
    """Probability mass of transitions INTO a baited site (pooled count matrix).

    Computed as (number of transitions ending at a baited site) /
    (total number of transitions). This is the observed statistic the null
    is compared against.
    """
    total = C.sum()
    if total == 0:
        return np.nan
    return C[:, BAITED_IDX].sum() / total


# ---------------------------------------------------------------------------
# Geometry / distance-decay ("gravity") null.
#
# Baited-in transition mass is confounded by spatial position: central sites
# and sites that lie *between* baited sites collect transitions regardless of
# reward. To isolate a reward preference we compare the data against a gravity
# model where the probability of moving i -> j depends only on inter-site
# distance, P_grav(i->j) ∝ exp(-D_ij / lambda) over j != i. lambda is fit by
# maximum likelihood to the observed transitions, so the null reproduces the
# overall trip-length profile; any *excess* inflow to baited sites beyond this
# null is geometry-controlled evidence of reward-directed routing.
# ---------------------------------------------------------------------------
def gravity_matrix(lam):
    """Row-stochastic distance-decay transition matrix, self-transitions zeroed."""
    W = np.exp(-DIST / lam)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    return W / rs


def fit_gravity_lambda(C):
    """MLE of the distance-decay length scale lambda (metres) from counts C."""
    def neg_ll(lam):
        if lam <= 0:
            return np.inf
        P = gravity_matrix(lam)
        mask = C > 0
        return -float((C[mask] * np.log(P[mask] + 1e-300)).sum())
    res = minimize_scalar(neg_ll, bounds=(1.0, 500.0), method="bounded")
    return float(res.x)


def expected_baited_in(C, P_grav):
    """Geometry-expected transition mass into baited sites for count matrix C.

    Each from-site is weighted by how often it is actually departed (its row sum
    in C), then routed to destinations by the gravity model.
    """
    rs = C.sum(axis=1)
    total = rs.sum()
    if total == 0:
        return np.nan
    w = rs / total
    return float((w[:, None] * P_grav)[:, BAITED_IDX].sum())


def inflow_vectors(C, P_grav):
    """Observed and geometry-expected inflow probability per site (length N)."""
    total = C.sum()
    if total == 0:
        return np.full(N_SITES, np.nan), np.full(N_SITES, np.nan)
    obs = C.sum(axis=0) / total
    rs = C.sum(axis=1)
    w = rs / rs.sum()
    exp = (w[:, None] * P_grav).sum(axis=0)
    return obs, exp


def geometry_excess_bootstrap(sequences, lam, n_boot=N_BOOT, seed=SEED):
    """Bootstrap (over trials) the geometry-controlled baited-in excess.

    Excess = observed baited-in mass - gravity-expected baited-in mass, using a
    fixed lambda. Returns (excess_obs, lo95, hi95, p_one_sided_gt0, n_seqs).
    """
    seqs = [s for s in sequences if len(s) >= 2]
    if not seqs:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    P_grav = gravity_matrix(lam)

    def _excess(seq_list):
        C = counts_from_sequences(seq_list)
        if C.sum() == 0:
            return np.nan
        return baited_in_mass(C) - expected_baited_in(C, P_grav)

    obs = _excess(seqs)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    idx = np.arange(len(seqs))
    for k in range(n_boot):
        pick = rng.choice(idx, size=len(seqs), replace=True)
        boot[k] = _excess([seqs[i] for i in pick])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    p_gt0 = float((np.sum(boot <= 0) + 1) / (n_boot + 1))  # one-sided: excess>0
    return (float(obs), float(lo), float(hi), p_gt0, len(seqs))


def shuffle_sequence(seq, rng):
    """Shuffle the ORDER of a visit sequence, then collapse immediate repeats.

    Preserves the multiset of sites visited in the trial but destroys the
    temporal ordering, providing a null for 'is the transition structure
    non-random?'.
    """
    if len(seq) < 2:
        return seq
    perm = rng.permutation(seq)
    out = []
    for s in perm:
        if out and out[-1] == s:
            continue
        out.append(s)
    return out


def perm_test_baited_in(sequences, n_perm=N_PERM, seed=SEED):
    """Permutation test on baited-in transition mass vs visit-order shuffle null.

    Returns (observed, null_mean, lo95, hi95, two_sided_p, n_seqs_used).
    """
    seqs = [s for s in sequences if len(s) >= 2]
    if len(seqs) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    obs = baited_in_mass(counts_from_sequences(seqs))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        shuffled = [shuffle_sequence(s, rng) for s in seqs]
        null[k] = baited_in_mass(counts_from_sequences(shuffled))
    null_mean = float(np.nanmean(null))
    lo, hi = np.nanpercentile(null, [2.5, 97.5])
    # Two-sided empirical p (centre on null mean), +1 smoothing.
    diff_obs = abs(obs - null_mean)
    diff_null = np.abs(null - null_mean)
    p = (np.sum(diff_null >= diff_obs) + 1) / (n_perm + 1)
    return (float(obs), null_mean, float(lo), float(hi), float(p), len(seqs))


def cohort_diff_test(early, late, n_perm=N_PERM, seed=SEED):
    """Late-minus-early baited-in mass with a cohort-label shuffle null."""
    e = [s for s in early if len(s) >= 2]
    l = [s for s in late if len(s) >= 2]
    if not e or not l:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    obs = baited_in_mass(counts_from_sequences(l)) - \
        baited_in_mass(counts_from_sequences(e))
    pool = e + l
    n_e = len(e)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    idx = np.arange(len(pool))
    for k in range(n_perm):
        rng.shuffle(idx)
        pe = [pool[i] for i in idx[:n_e]]
        pl = [pool[i] for i in idx[n_e:]]
        null[k] = baited_in_mass(counts_from_sequences(pl)) - \
            baited_in_mass(counts_from_sequences(pe))
    nm = float(np.nanmean(null))
    p = (np.sum(np.abs(null - nm) >= abs(obs - nm)) + 1) / (n_perm + 1)
    return (float(obs), nm, float(np.nanpercentile(null, 2.5)),
            float(np.nanpercentile(null, 97.5)), float(p))


# ===========================================================================
# Load data
# ===========================================================================
print("Loading data (trials + tracks cache)...")
trials = build_trials()
tracks_cache = build_tracks_cache(trials)

test_trials = [t for t in trials
               if t["config"] in TEST_CONFIGS
               and t["assay"] is not None
               and isinstance(t["assay"], int)
               and t["date"] >= PHASE2_START
               and t["group_num"] not in CTRL_GROUPS]
print(f"Total trials: {len(trials)}, Phase 2 test trials: {len(test_trials)}")


# ===========================================================================
# Build visit sequences
# ===========================================================================
print("Building visit sequences...")
records = []  # (trial, assay, sequence)
n_skipped = 0
for trial in test_trials:
    try:
        seq = trial_visit_sequence(trial, tracks_cache)
    except Exception as exc:  # robustness: don't crash on mislabels etc.
        print(f"  WARN: trial group {trial.get('group_num')} assay "
              f"{trial.get('assay')} failed: {exc!r}; skipping")
        n_skipped += 1
        continue
    if len(set(seq)) < 2:  # need >=2 distinct sites to have a transition
        n_skipped += 1
        continue
    records.append((trial, trial["assay"], seq))

all_seqs = [r[2] for r in records]
print(f"Usable trials: {len(records)}  "
      f"(skipped {n_skipped} with <2 distinct visits / errors)")

# Sequence-length diagnostics
seq_lens = np.array([len(s) for s in all_seqs])
n_trans = int(sum(max(len(s) - 1, 0) for s in all_seqs))
print(f"Visit-sequence length: median {np.median(seq_lens):.0f}, "
      f"range [{seq_lens.min()}, {seq_lens.max()}]; total transitions {n_trans}")


# ===========================================================================
# Pooled transition matrix (all usable trials)
# ===========================================================================
C_all = counts_from_sequences(all_seqs)
P_all = row_normalise(C_all)
row_sums = P_all.sum(axis=1)
active_rows = int((row_sums > 0).sum())
# Sanity: active rows must sum to ~1.
assert np.allclose(row_sums[row_sums > 0], 1.0), "transition rows must sum to 1"

pi = stationary_distribution(P_all)
H_trans = transition_entropy(P_all, C_all)

print(f"\nPooled chain: {active_rows}/{N_SITES} sites have outgoing transitions; "
      f"row-weighted transition entropy = {H_trans:.3f} bits "
      f"(max {np.log2(N_SITES):.3f}).")
if pi is not None:
    top = np.argsort(pi)[::-1][:5]
    print("Stationary distribution (top 5 sites): "
          + ", ".join(f"{SITE_LABELS[i]}={pi[i]:.3f}"
                      + ("*" if i in BAITED_IDX else "") for i in top))
    pi_baited = float(pi[BAITED_IDX].sum())
    print(f"Stationary mass on baited sites = {pi_baited:.3f} "
          f"(uniform expectation {len(BAITED_IDX) / N_SITES:.3f}).")
else:
    pi_baited = np.nan
    print("Stationary distribution: degenerate (insufficient connected states).")


# ===========================================================================
# Permutation tests: transitions INTO baited sites vs shuffled-order null
# ===========================================================================
print(f"\nPermutation tests (visit-order shuffle null, {N_PERM} perms)...")
res_all = perm_test_baited_in(all_seqs)
obs_a, nm_a, lo_a, hi_a, p_a, n_a = res_all
print(f"  ALL  trials (n={n_a}): baited-in mass obs={obs_a:.3f}, "
      f"null={nm_a:.3f} [{lo_a:.3f}, {hi_a:.3f}], p={p_a:.4f}")

early_seqs = [r[2] for r in records if r[1] <= EARLY_MAX_ASSAY]
late_seqs = [r[2] for r in records if r[1] >= LATE_MIN_ASSAY]

res_early = perm_test_baited_in(early_seqs)
res_late = perm_test_baited_in(late_seqs)
obs_e, nm_e, lo_e, hi_e, p_e, n_e = res_early
obs_l, nm_l, lo_l, hi_l, p_l, n_l = res_late
print(f"  EARLY (assay<= {EARLY_MAX_ASSAY}, n={n_e}): obs={obs_e:.3f}, "
      f"null={nm_e:.3f} [{lo_e:.3f}, {hi_e:.3f}], p={p_e:.4f}")
print(f"  LATE  (assay>= {LATE_MIN_ASSAY}, n={n_l}): obs={obs_l:.3f}, "
      f"null={nm_l:.3f} [{lo_l:.3f}, {hi_l:.3f}], p={p_l:.4f}")

diff_obs, diff_nm, diff_lo, diff_hi, diff_p = cohort_diff_test(early_seqs, late_seqs)
print(f"  LATE - EARLY baited-in difference: obs={diff_obs:+.3f}, "
      f"null={diff_nm:+.3f} [{diff_lo:+.3f}, {diff_hi:+.3f}], p={diff_p:.4f}")


# ===========================================================================
# Geometry-controlled test: distance-decay (gravity) null
# ===========================================================================
print("\nGeometry / distance-decay (gravity) null...")
LAMBDA = fit_gravity_lambda(C_all)
P_grav = gravity_matrix(LAMBDA)
grav_baited_in = expected_baited_in(C_all, P_grav)
obs_in, exp_in = inflow_vectors(C_all, P_grav)
print(f"  Fitted distance-decay scale lambda = {LAMBDA:.1f} m.")
print(f"  Baited-in mass: observed {obs_a:.3f} vs geometry-expected "
      f"{grav_baited_in:.3f}.")

gx_obs, gx_lo, gx_hi, gx_p, gx_n = geometry_excess_bootstrap(all_seqs, LAMBDA)
ge_obs, ge_lo, ge_hi, ge_p, _ = geometry_excess_bootstrap(early_seqs, LAMBDA)
gl_obs, gl_lo, gl_hi, gl_p, _ = geometry_excess_bootstrap(late_seqs, LAMBDA)
print(f"  Geometry-controlled excess (obs - gravity), bootstrap over trials:")
print(f"    ALL  : {gx_obs:+.3f} [{gx_lo:+.3f}, {gx_hi:+.3f}], p(excess>0)={gx_p:.4f}")
print(f"    EARLY: {ge_obs:+.3f} [{ge_lo:+.3f}, {ge_hi:+.3f}], p={ge_p:.4f}")
print(f"    LATE : {gl_obs:+.3f} [{gl_lo:+.3f}, {gl_hi:+.3f}], p={gl_p:.4f}")

# Per-site excess inflow (observed - geometry-expected), baited vs unbaited.
excess_in = obs_in - exp_in
baited_excess_mean = float(np.nanmean(excess_in[BAITED_IDX]))
unbaited_excess_mean = float(np.nanmean(excess_in[UNBAITED_IDX]))
print(f"  Mean per-site inflow excess: baited {baited_excess_mean:+.4f} vs "
      f"unbaited {unbaited_excess_mean:+.4f}.")
order = np.argsort(excess_in)[::-1]
print("  Top inflow-excess sites: " + ", ".join(
    f"{SITE_LABELS[i]}{'*' if i in set(BAITED_IDX) else ''}={excess_in[i]:+.3f}"
    for i in order[:5]))


# ===========================================================================
# Figure: transition-probability heatmap (pooled, early, late)
# ===========================================================================
print("\nGenerating transition heatmap...")
P_early = row_normalise(counts_from_sequences(early_seqs))
P_late = row_normalise(counts_from_sequences(late_seqs))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, P, title in zip(
        axes, [P_all, P_early, P_late],
        [f"Pooled (n={len(all_seqs)})",
         f"Early assay<= {EARLY_MAX_ASSAY} (n={n_e})",
         f"Late assay>= {LATE_MIN_ASSAY} (n={n_l})"]):
    im = ax.imshow(P, origin="upper", cmap="magma", vmin=0, vmax=max(P.max(), 1e-6))
    ax.set_xticks(range(N_SITES))
    ax.set_yticks(range(N_SITES))
    ax.set_xticklabels(SITE_LABELS, rotation=90, fontsize=7)
    ax.set_yticklabels(SITE_LABELS, fontsize=7)
    ax.set_xlabel("To site")
    ax.set_ylabel("From site")
    ax.set_title(title, fontsize=10)
    # Highlight baited columns/rows.
    for bi in BAITED_IDX:
        ax.axvline(bi, color="#3BD8A0", lw=0.6, alpha=0.5)
        ax.axhline(bi, color="#3BD8A0", lw=0.6, alpha=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(to | from)")

fig.suptitle("First-order site-to-site transition probabilities "
             "(baited sites A1/A2/A3 marked)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGDIR / "visit_markov_chain.pdf")
fig.savefig(FIGDIR / "visit_markov_chain.png")
plt.close(fig)
print("  -> visit_markov_chain")


# ===========================================================================
# Figure: arena MAP with transition traces (directed edges weighted by P)
# ===========================================================================
print("Generating arena transition map...")
GRID = POS / 10.0          # back to 0-5 grid coords for plotting
P_THRESH = 0.06            # only draw edges above this probability (declutter)


def draw_transition_map(ax, C, title):
    P = row_normalise(C)
    total = C.sum()
    inflow = C.sum(axis=0) / total if total > 0 else np.zeros(N_SITES)
    # Arena frame.
    ax.add_patch(plt.Rectangle((0, 0), 5, 5, fill=False, ec="#cccccc", lw=1.0))
    ax.set_xlim(-0.4, 5.4)
    ax.set_ylim(-0.4, 5.4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    # Directed edges (curved so i->j and j->i don't overlap).
    for i in range(N_SITES):
        for j in range(N_SITES):
            if i == j or P[i, j] < P_THRESH:
                continue
            to_baited = j in set(BAITED_IDX)
            ax.add_patch(FancyArrowPatch(
                GRID[i], GRID[j],
                connectionstyle="arc3,rad=0.12",
                arrowstyle="-|>", mutation_scale=8,
                lw=0.4 + 4.0 * P[i, j],
                alpha=min(0.9, 0.25 + P[i, j]),
                color="#B8860B" if to_baited else "#9aa0a6",
                zorder=2 if to_baited else 1))
    # Nodes: size ∝ inflow, baited gold.
    for k in range(N_SITES):
        baited = k in set(BAITED_IDX)
        ax.add_patch(Circle(
            GRID[k], 0.16 + 1.1 * inflow[k],
            fc="#E8B83D" if baited else "#e6e6e6",
            ec="#B8860B" if baited else "#888", lw=1.0, zorder=3))
        ax.text(GRID[k, 0], GRID[k, 1], SITE_LABELS[k], ha="center", va="center",
                fontsize=6.5, zorder=4, fontweight="bold" if baited else "normal")


fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
for ax, C, ttl in zip(
        axes,
        [C_all, counts_from_sequences(early_seqs), counts_from_sequences(late_seqs)],
        [f"Pooled (n={len(all_seqs)})",
         f"Early assay<= {EARLY_MAX_ASSAY} (n={n_e})",
         f"Late assay>= {LATE_MIN_ASSAY} (n={n_l})"]):
    draw_transition_map(ax, C, ttl)
fig.suptitle("Inter-site transition probabilities on the arena "
             f"(edges P>= {P_THRESH}; baited A1/A2/A3 gold; node size ∝ inflow)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGDIR / "visit_markov_map.pdf")
fig.savefig(FIGDIR / "visit_markov_map.png")
plt.close(fig)
print("  -> visit_markov_map")


# ===========================================================================
# FINDINGS
# ===========================================================================
def verdict(obs, null_mean, p):
    if np.isnan(p):
        return "indeterminate (no data)"
    direction = "above" if obs > null_mean else "below"
    sig = "significantly" if p < 0.05 else "not significantly"
    return f"{sig} {direction} null (p={p:.3f})"


print("\n" + "=" * 70)
print("FINDINGS:")
print(f"- Frame: visit sequences are built on canonically ORIENTED tracks "
      f"(apply_orient=True), so baited sites are always {sorted(BAITED_CANONICAL)} "
      f"and the transition matrix is in the transformed frame.")
print(f"- Pooled over {len(all_seqs)} Phase-2 test trials ({n_trans} inter-site "
      f"transitions), movement between sites is structured: the row-weighted "
      f"transition entropy is {H_trans:.2f} bits vs a maximum of "
      f"{np.log2(N_SITES):.2f} bits, so transitions are far from uniform.")
print(f"- Transition mass INTO baited sites (A1/A2/A3) = {obs_a:.3f}; under a "
      f"within-trial visit-order shuffle the null is {nm_a:.3f} "
      f"[{lo_a:.3f}, {hi_a:.3f}] -> {verdict(obs_a, nm_a, p_a)}. "
      f"(Baited sites are 3/{N_SITES} = {3/N_SITES:.3f} of all sites.)")
print(f"- GEOMETRY CONTROL: baited sites are spatially central, so the order-"
      f"shuffle test above is confounded by position. Against a distance-decay "
      f"(gravity) null (lambda={LAMBDA:.0f} m), geometry alone predicts "
      f"{grav_baited_in:.3f} baited-in mass; the observed excess is "
      f"{gx_obs:+.3f} [{gx_lo:+.3f}, {gx_hi:+.3f}] "
      f"({'baited sites OVER-visited beyond geometry' if gx_p < 0.05 and gx_obs > 0 else 'not distinguishable from geometry'}, "
      f"bootstrap p={gx_p:.3f}). Mean per-site inflow excess: baited "
      f"{baited_excess_mean:+.4f} vs unbaited {unbaited_excess_mean:+.4f}.")
if not np.isnan(pi_baited):
    print(f"- The chain's stationary distribution places {pi_baited:.3f} of its "
          f"mass on baited sites (uniform = {3/N_SITES:.3f}); note this too "
          f"partly reflects baited-site centrality (see geometry control).")
print(f"- Learning: late-cohort geometry-controlled excess {gl_obs:+.3f} "
      f"(p={gl_p:.3f}) vs early {ge_obs:+.3f} (p={ge_p:.3f}); order-shuffle "
      f"late-minus-early baited-in difference {diff_obs:+.3f} "
      f"{verdict(diff_obs, diff_nm, diff_p).split(' (')[0]} (p={diff_p:.3f}).")
print(f"- See figures/visit_markov_map.png for the transition traces on the arena "
      f"(pooled / early / late).")
print("=" * 70)
