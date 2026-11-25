# Quantitative Analysis of Rapid Chess: Stochastic Match Duration Modeling and Spatial Piece Capture Distributions

An empirical investigation of competitive 10-minute rapid chess games ($N = 2,484$ matches, $44,166$ piece captures) sampled from historical Lichess databases. The project is divided into two modules: parametric modeling of match duration distributions and spatial probability modeling of piece captures across board coordinates.

---

## Executive Summary

1. **Part 1 — Match Duration Distribution Modeling**  
   - Formal evaluation of match duration distributions for rapid 10-minute chess (time control `600+0`), testing the Gaussian hypothesis $\mathcal{N}(\mu, \sigma^2)$ against asymmetric alternatives ($\text{Log-Normal}$ and $\text{Gamma}$).
   - Parameter estimation via Maximum Likelihood Estimation (MLE), model selection through Log-Likelihood, Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), and two-sample Kolmogorov-Smirnov (KS) tests.
   - Derivation and empirical estimation of conditional outcome probabilities $P(\text{White Win} \mid T = t)$, $P(\text{Black Win} \mid T = t)$, and $P(\text{Draw} \mid T = t)$.

2. **Part 2 — Spatial Capture Probability and Exchange Dynamics**  
   - Empirical estimation of discrete spatial capture probability across the 64 squares:
     $$p_{\text{emp}}(i) = \frac{D_i}{D_{\text{total}}}$$
     where $D_i$ denotes total captures on square $i \in \{a1, \dots, h8\}$ and $D_{\text{total}} = \sum_{j=1}^{64} D_j$.
   - Identification and comparative analysis of central capture hotspots: `d4` ($7.54\%$), `d5` ($7.32\%$), `e5` ($5.30\%$), and `e4` ($4.67\%$). Together, these four central squares account for $24.83\%$ of all captures on the board ($4.83\times$ uniform baseline expectation).
   - Tactical exchange analysis on square `d4`: breakdown of captured piece types, primary opening variations, and the correlation between local exchange success and overall match victory.

---

## Part 1: Match Duration Distribution Modeling

### Theoretical Framework
In a 10-minute rapid game without increment (`600+0`), both players receive an initial clock allocation of $T_0 = 600\text{ seconds}$ ($10\text{ minutes}$). Total match duration $T$ satisfies the bounded support:
$$0 < T \le 2T_0 = 20.0\text{ minutes}$$

We test the base assumption that game length follows a Gaussian distribution:
$$T \sim \mathcal{N}(\mu, \sigma^2), \quad f_N(t; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(t - \mu)^2}{2\sigma^2} \right)$$

We compare this baseline against two continuous non-negative models commonly applied in survival analysis:
- **Log-Normal:** $T \sim \text{LogN}(\mu_{\ln}, \sigma_{\ln}^2)$
- **Gamma:** $T \sim \Gamma(\alpha, \beta)$

### Empirical Parameter Estimation and Goodness-of-Fit
Across $N = 2,484$ verified games:
- **Sample Mean ($\hat{\mu}$):** $13.00\text{ minutes}$ ($779.8\text{ seconds}$)
- **Sample Standard Deviation ($\hat{\sigma}$):** $5.34\text{ minutes}$ ($320.4\text{ seconds}$)
- **Skewness:** $-0.42$ | **Kurtosis:** $-0.81$

| Distribution Model | Fitted Parameters (MLE) | Log-Likelihood ($\ln L$) | AIC ($2k - 2\ln L$) | BIC ($k\ln n - 2\ln L$) | KS Statistic ($D$) | KS $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal $\mathcal{N}(\mu, \sigma^2)$ [Base]** | $\mu = 13.001$, $\sigma = 5.344$ | **$-7686.1$** | **$15376.3$** | **$15387.9$** | **$0.0776$** | **$1.91 \times 10^{-13}$** |
| **Gamma $\Gamma(\alpha, \beta)$** | $\alpha = 5.923$, $\theta = 2.195$ | $-7972.7$ | $15949.4$ | $15961.0$ | $0.1212$ | $2.88 \times 10^{-32}$ |
| **Log-Normal $\text{LogN}(\mu_{\ln}, \sigma_{\ln}^2)$** | $\mu_{\ln} = 2.457$, $\sigma_{\ln} = 0.449$ | $-8303.0$ | $16610.0$ | $16621.6$ | $0.1544$ | $3.65 \times 10^{-52}$ |

#### Analytical Interpretation
Standard time-to-event phenomena with open-ended horizons exhibit positive skewness and are typically modeled with Gamma or Log-Normal distributions. In fixed-clock rapid chess, however, the presence of a deterministic ceiling ($T \le 20\text{ minutes}$) truncates the right tail. Clock management pushes the mass toward the later stages, yielding negative skewness ($-0.42$). Consequently, the Normal distribution provides a closer parametric fit than both Gamma ($\Delta\text{AIC} = -573.1$) and Log-Normal ($\Delta\text{AIC} = -1233.7$).

### Conditional Win Rate Dynamics: $P(\text{Outcome} \mid T = t)$
Empirical outcome tracking demonstrates three distinct operational phases:
1. **Early Phase ($t < 6\text{ minutes}$):**  
   Terminations are predominantly tactical knockouts or early blunders. First-move initiative confers a measurable advantage to White:
   $$P(\text{White Win} \mid T < 6\text{ min}) \ge 55.4\%$$
2. **Middlegame Phase ($6 \le t \le 14\text{ minutes}$):**  
   Outcomes reflect balanced tactical play, converging toward typical baseline frequencies ($\approx 48\%$ White, $46\%$ Black, $6\%$ Draw).
3. **Endgame and Time-Scramble Phase ($t > 15\text{ minutes}$):**  
   Due to material simplification (transition to King-and-Pawn endgames) and mutual clock pressure, draw probability increases threefold:
   $$P(\text{Draw} \mid T > 16\text{ min}) \approx 15.5\% \quad \text{vs.} \quad 4.8\% \text{ for } 6 \le t \le 14\text{ min}$$

---

## Part 2: Spatial Piece Capture Probability and Exchange Dynamics

### Spatial Distribution Formulation
Across the 2,500 games examined, $44,166$ piece capture events were recorded. Under a discrete uniform null hypothesis $H_0$, the expected capture probability per square is:
$$E[p(i)] = \frac{1}{64} \approx 1.5625\% \quad (\approx 690 \text{ captures/square})$$

#### Top Capture Squares by Frequency:
| Rank | Square | Total Captures ($D_i$) | Empirical Probability $p_{\text{emp}}(i)$ | Relative Ratio to Uniform Baseline |
| :---: | :---: | :---: | :---: | :---: |
| **1** | **d4** | **3,332** | **7.54%** | **4.83×** |
| **2** | **d5** | **3,231** | **7.32%** | **4.68×** |
| **3** | **e5** | **2,339** | **5.30%** | **3.39×** |
| **4** | **e4** | **2,062** | **4.67%** | **2.99×** |
| **5** | **c4** | **1,421** | **3.22%** | **2.06×** |

Central control dictates capture frequency: squares `d4`, `d5`, `e5`, and `e4` concentrate $24.83\%$ of all casualties.

---

### Focus on Square d4

#### Piece Demographics
Distribution of captured pieces on square `d4` ($N = 3,332$):
- **Pawns:** $2,245$ ($67.38\%$) — central pawn tension and pawn breaks (`...cxd4`, `...exd4`)
- **Knights:** $640$ ($19.21\%$) — central outposts and recaptures (`Nxd4`)
- **Bishops:** $197$ ($5.91\%$) — central pin resolution and piece trades (`Bxd4`)
- **Queens:** $136$ ($4.08\%$) — early queen recaptures and central queen trades
- **Rooks:** $114$ ($3.42\%$) — central d-file operations

#### Primary Opening Catalysts
1. **Sicilian Defense Variations (ECO B54, B23, B40, B44):**  
   $$\text{1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4}$$
2. **French Defense Advance Variation (ECO C02):**  
   $$\text{1. e4 e6 2. d4 d5 3. e5 c5 4. c3 Nc6 5. Nf3 Qb6}$$
3. **Queen's Pawn Systems / Queen's Gambit (ECO D01, D02):**  
   $$\text{1. d4 d5 2. c4}$$

#### Multi-Move Exchange Brawls
A brawl is defined as two or more successive captures occurring on the same square within two plies of each other. Across $1,251$ recorded brawls on `d4`:
- $87.7\%$ are equal-value exchanges.
- In asymmetrical exchanges where one side captures greater material value:
  - If Black secures a positive material exchange on `d4`, Black's overall game win rate is **$54.7\%$** ($P(\text{Black Win Game} \mid \text{Black Wins Brawl on d4}) = 0.5465$).
  - If White secures a positive material exchange on `d4`, White's overall game win rate is **$47.1\%$** ($P(\text{White Win Game} \mid \text{White Wins Brawl on d4}) = 0.4706$), with a higher proportion of drawn endgames ($21.8\%$).

---

## Interactive Visualizations

The repository contains three self-contained HTML applications:

1. **`index.html`**  
   Unified dashboard combining tabbed navigation between the duration modeling module, the spatial capture heatmap, and the statistical summary.
2. **`part1_match_duration_distribution.html`**  
   Interactive chart displaying empirical density histograms alongside Normal, Gamma, and Log-Normal fitted curves. Cursor hovering computes point density, sample counts, and empirical outcome probabilities ($P(\text{White})$, $P(\text{Black})$, $P(\text{Draw})$) in real time.
3. **`part2_chessboard_death_heatmap.html`**  
   Interactive 8x8 chessboard visualization where square color saturation reflects empirical capture probability $p_{\text{emp}}(i)$. Selecting any square updates a tactical dossier displaying piece distributions, baseline multipliers, associated opening lines, and exchange decisiveness.

---

## Repository Structure

```
├── .gitignore                          # Standard Python, environment, and data ignore rules
├── data/
│   ├── rapid_10min_matches.jsonl       # Historical match records (time control 600+0)
│   ├── part1_results.json              # MLE parameters, AIC/BIC, KS statistics, win rates
│   └── part2_results.json              # 64-square capture statistics, piece breakdowns, brawls
├── figures/
│   ├── part1_match_duration_distribution.png  # High-resolution figure for Part 1
│   └── part2_chessboard_death_heatmap.png     # High-resolution figure for Part 2
├── download_data.py                    # Data ingestion script via Lichess API
├── part1_analysis.py                   # Parametric duration modeling and hypothesis testing
├── part2_analysis.py                   # Spatial capture probability and brawl tracking
├── generate_visualizations.py          # HTML visualization compiler
├── generate_figures.py                 # Matplotlib publication chart generator
├── part1_match_duration_distribution.html  # Standalone Part 1 interactive module
├── part2_chessboard_death_heatmap.html     # Standalone Part 2 interactive module
├── index.html                          # Master dashboard
└── README.md                           # Documentation and technical report
```

---

## Execution and Replication

```bash
# 1. Install prerequisites
pip install numpy scipy pandas matplotlib chess

# 2. Run statistical analysis scripts
python part1_analysis.py
python part2_analysis.py

# 3. Generate interactive HTML dashboards and publication figures
python generate_visualizations.py
python generate_figures.py

# 4. Inspect the interactive application
# Open index.html in any standard browser
```
