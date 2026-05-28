# Nature-Style Figure & Beamer Presentation Template

## Overview

This directory contains a reusable framework for generating publication-quality
figures (Nature journal style) and compiling them into a Beamer presentation.
The style choices follow Nature's author guidelines for submitted figures.

## Font & Style Specification

### Figures (matplotlib)

Nature requires **sans-serif fonts (Arial or Helvetica)** for all figure text.
The template falls back to DejaVu Sans when Arial/Helvetica are unavailable.

Key rcParams settings:
```python
plt.rcParams.update({
    # Font: Nature mandates sans-serif (Arial preferred)
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",

    # Font sizes: Nature specifies 5-7pt for text, 8pt bold for panel labels
    "font.size": 9,           # base size (adjusted for presentation scale)
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,

    # Spacing
    "axes.titlepad": 8,       # space between title and plot
    "axes.labelpad": 5,       # space between label and axis
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "savefig.pad_inches": 0.12,

    # Clean axes: no top/right spines (Nature convention)
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,

    # Output
    "figure.dpi": 200,
    "savefig.bbox": "tight",

    # Legend
    "legend.frameon": False,
    "legend.fontsize": 7,
})
```

### Colour Palette
```python
PALETTE = {
    "blue":   "#3B7DD8",
    "orange": "#E8823A",
    "green":  "#4AAD5B",
    "purple": "#8B6DAF",
    "red":    "#D64550",
    "grey":   "#888888",
    "gold":   "#E8B83D",
}
```

### Presentation (LaTeX Beamer)

The Beamer template uses a minimal, modern theme:
- **No navigation symbols** — clean footer with frame number only
- **Frame titles** — bold, dark grey, with a thin horizontal rule separator
- **Bullet style** — filled circles (level 1), dashes (level 2)
- **No ornamental colours** — black/grey text with accent from figure colours
- Panel labels on figures use lowercase bold letters: **a**, **b**, **c**

## Key Functions

### `_nature_box(ax, data_list, group_labels, colour, ylabel, title, ...)`
Draws a Nature-style box plot:
- Semi-transparent box fill (alpha=0.35) with coloured edge
- Jittered individual data points overlaid (s=10, alpha=0.55)
- Sample sizes (n=) annotated below the x-axis (axes-fraction coordinates)
- Median line in black
- No outlier markers (outliers visible as jittered points)

Parameters:
- `show_points=True` — toggle jittered scatter overlay
- `show_n=True` — toggle sample size annotations
- `ylim=None` — optional fixed y-axis limits

### `_kw_annotation(ax, data_list, y_frac=0.95)`
Adds Kruskal-Wallis H-test result (H statistic + p-value) as italic text
in the top-right corner of the axes.

### `_mwu_annotation(ax, d1, d2, y_frac=0.88)`
Adds Mann-Whitney U test p-value as italic text.

### Text positioning convention
- **KW annotation**: `y_frac=0.95` (top-right, axes fraction)
- **Spearman ρ**: `y_frac=0.85` (below KW)
- **MWU annotation**: `y_frac=0.88` (between KW and Spearman)
- **n-labels**: bottom of axes via `ax.get_xaxis_transform()`
- **Reference lines** (e.g., "fully egalitarian"): placed in data coordinates
  with sufficient margin above the reference value

This layering prevents text collisions.

## File Structure

```
analysis/
├── generate_figures.py   # All figure generation (run with: uv run python analysis/generate_figures.py)
├── presentation.tex      # Beamer source (compile with: tectonic presentation.tex)
├── presentation.pdf      # Compiled output
├── TEMPLATE_README.md    # This file
└── figures/              # Generated PDF + PNG figures
    ├── *.pdf             # Vector figures (for LaTeX)
    └── *.png             # Raster figures (for Beamer \includegraphics)
```

## Compilation

### Figures
```bash
uv run python analysis/generate_figures.py
```

### Presentation
```bash
cd analysis && tectonic presentation.tex
```
Tectonic auto-downloads LaTeX packages on first run. No TeX Live installation needed.

Install tectonic (static musl build, no system deps):
```bash
curl -sL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.16.9/tectonic-0.16.9-x86_64-unknown-linux-musl.tar.gz" | tar xz -C ~/.local/bin/
```

## Beamer Theme Template

Minimal starting point for new presentations:
```latex
\documentclass[aspectratio=169, 10pt]{beamer}
\usetheme{default}
\usecolortheme{default}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{frametitle}{%
  \vspace{0.5em}%
  {\usebeamerfont{frametitle}\usebeamercolor[fg]{frametitle}\insertframetitle}%
  \vspace{0.35em}%
  \textcolor{black!15}{\hrule height 0.4pt}%
  \vspace{0.15em}%
}
\setbeamertemplate{footline}{%
  \hfill\textcolor{black!30}{\scriptsize\insertframenumber/\inserttotalframenumber\kern1em}%
  \vspace{0.3em}%
}
\setbeamercolor{frametitle}{fg=black!85}
\setbeamercolor{title}{fg=black!85}
\setbeamerfont{frametitle}{series=\bfseries, size=\normalsize}
\setbeamerfont{title}{series=\bfseries, size=\Large}
\setbeamercolor{structure}{fg=black!70}
\setbeamercolor{itemize item}{fg=black!50}
\setbeamertemplate{itemize item}{\textbullet}
\setbeamertemplate{itemize subitem}{--}

\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}

\begin{document}

\begin{frame}[plain]
\vspace{2.5em}
\begin{center}
{\LARGE\bfseries Title Here}\\[1em]
{\normalsize\color{black!60} Subtitle}\\[2em]
{\small\color{black!40} Date}
\end{center}
\end{frame}

\begin{frame}{Slide Title}
\centering
\includegraphics[width=0.85\textwidth]{figure.png}
\vspace{0.1em}
\small
\begin{itemize}\setlength\itemsep{0.08em}
    \item Key finding 1.
    \item Key finding 2.
\end{itemize}
\end{frame}

\end{document}
```

## Nature Figure Guidelines Reference

- **Font**: Sans-serif (Arial or Helvetica preferred)
- **Font size**: 5-7pt for all text; 8pt bold for panel labels
- **Panel labels**: Lowercase bold letters (a, b, c), not in parentheses
- **Figure width**: 90mm (single column) or 180mm (double column)
- **Max height**: 170mm
- **Spines**: Remove top and right spines
- **Colours**: Use colourblind-friendly palettes
- **Statistics**: Annotate p-values, test names, effect sizes on plots
- **Sample sizes**: Always show n per group
- **Individual data points**: Overlay on summary statistics when n < ~50

Source: https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
