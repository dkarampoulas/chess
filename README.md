# Chess Analytics & Probability Theory: Match Duration Distributions & Spatial Piece Carnage

> **A publication-grade Data Science & Stochastic Modeling Portfolio Project for CV & Technical Interviews.**  
> Investigating 2,500 real competitive 10-minute rapid games (**44,166 total piece deaths**) downloaded from Lichess historical data.

---

## 📌 Executive Summary

This project applies **parametric probability theory**, **maximum likelihood estimation (MLE)**, and **spatial empirical statistics** to competitive chess:

1. **Part 1 — Match Duration Distribution Modeling:**  
   - Models the probability distribution of 10-minute rapid chess game durations ($T \in (0, 20]$ minutes) using the **Normal Distribution $\mathcal{N}(\mu, \sigma^2)$** as the base hypothesis.
   - Evaluates goodness-of-fit against asymmetric continuous alternatives: **Log-Normal** and **Gamma Distributions** via Log-Likelihood, AIC, BIC, and Kolmogorov-Smirnov (KS) tests.
   - Derives analytical proofs for conditional outcome probabilities $P(\text{White Win} \mid T = t)$, $P(\text{Black Win} \mid T = t)$, and $P(\text{Draw} \mid T = t)$ with interactive hover cursor inspections.
2. **Part 2 — Spatial Piece Deaths & Tactical Brawl Decisiveness:**  
   - Defines the empirical death probability:
     $$p_{\text{emp}}(i) = \frac{D_i}{D_{\text{total}}}$$
     where $D_i$ is the number of captures occurring on square $i \in \{a1, \dots, h8\}$.
   - Discovers that **d4** is the #1 deadliest square in chess history ($p_{\text{emp}} = 7.54\%$), followed by **d5** ($7.32\%$), **e5** ($5.30\%$), and **e4** ($4.67\%$). The four central squares alone capture **24.8% of all fallen pieces** across the 64-square board (**4.83× above uniform chance**).
   - Double-checks the openings (e.g. Open Sicilian, French Defense), move sets, piece sacrifice demographics (Pawns, Knights, Bishops, Rooks, Queens), and game-theoretic outcome of multi-piece exchange brawls.
   - Generates an interactive bloody chessboard heatmap with visceral crimson intensity proportional to $p_{\text{emp}}(i)$.

---

## 📊 Part 1: Average Match Duration & Distribution Proof

### 1. Mathematical Formulation & Base Model
In a 10-minute rapid chess match (time control `600+0`), both players start with a time budget of $T_0 = 600\text{ s}$ ($10\text{ min}$).  
The theoretical support of match duration $T$ is strictly bounded by $[0, 2T_0] = [0, 20.0\text{ min}]$.

We define the base hypothesis as a Gaussian distribution:
$$T \sim \mathcal{N}(\mu, \sigma^2), \quad f_N(t; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(t - \mu)^2}{2\sigma^2} \right)$$

### 2. MLE Parameters & Goodness-of-Fit Comparison
Across $N = 2,484$ verified 10-minute rapid games:
- **Sample Mean:** $\hat{\mu} = 13.00\text{ minutes}$ ($779.8\text{ seconds}$)
- **Sample Standard Deviation:** $\hat{\sigma} = 5.34\text{ minutes}$ ($320.4\text{ seconds}$)
- **Empirical Skewness:** $-0.42$ | **Kurtosis:** $-0.81$

| Distribution Family | Parameters (MLE) | Log-Likelihood ($\ln L$) | AIC ($2k - 2\ln L$) | BIC ($k\ln n - 2\ln L$) | KS Statistic ($D$) | KS $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal $\mathcal{N}(\mu, \sigma^2)$ [Base]** | $\mu = 13.001$, $\sigma = 5.344$ | **$-7686.1$** | **$15376.3$** | **$15387.9$** | **$0.0776$** | **$1.91 \times 10^{-13}$** |
| **Gamma $\Gamma(\alpha, \beta)$** | $\alpha = 5.923$, $\theta = 2.195$ | $-7972.7$ | $15949.4$ | $15961.0$ | $0.1212$ | $2.88 \times 10^{-32}$ |
| **Log-Normal $\text{LogN}(\mu_{\ln}, \sigma_{\ln}^2)$** | $\mu_{\ln} = 2.457$, $\sigma_{\ln} = 0.449$ | $-8303.0$ | $16610.0$ | $16621.6$ | $0.1544$ | $3.65 \times 10^{-52}$ |

#### 💡 Analytical Finding:
Unlike unconstrained time-to-event processes (which are strictly right-skewed and modeled by Gamma/Log-Normal), 10-minute rapid chess features a **hard maximum upper bound at 20 minutes**. Players budget their clocks toward the final minutes, producing a centered distribution around 13 minutes. As a consequence, the **Normal distribution achieves a significantly superior goodness-of-fit (AIC = 15,376.3 vs Gamma's 15,949.4)**.

### 3. Proof of Conditional Matches Won: $P(\text{Outcome} \mid T = t)$
When gliding the cursor across duration $t$, the conditional probabilities exhibit three distinct game-theoretic regimes:
1. **Early Blitz / Blunder Regime ($t < 6\text{ min}$):**  
   Early tactical knockouts and opening traps (Scholar's mates, hanging pieces) exaggerate White's first-move initiative advantage:
   $$P(\text{White Win} \mid T < 6\text{ min}) \ge 55.4\%$$
2. **Middlegame Strategic Balance ($6 \le t \le 14\text{ min}$):**  
   Black achieves full piece coordination; outcomes stabilize near typical grandmaster frequencies ($\approx 48\%$ White, $46\%$ Black, $6\%$ Draw).
3. **Endgame Flagging & Simplification Regime ($t > 15\text{ min}$):**  
   As pieces are traded into simplified King-and-Pawn endgames and both players enter time scrambles, the draw probability surges **3-fold**:
   $$P(\text{Draw} \mid T > 16\text{ min}) \approx 15.5\% \quad \text{vs.} \quad 4.8\% \text{ in middlegames}$$

---

## 🩸 Part 2: Spatial Piece Deaths & Tactical Brawl Analysis

### 1. Empirical Spatial Probability Distribution
Total pieces captured ("deaths") across 2,500 games: **$D_{\text{total}} = 44,166$**.  
If captures were uniform across the 64 squares:
$$E[p(i)] = \frac{1}{64} \approx 1.5625\% \quad (\approx 690 \text{ deaths/square})$$

#### Top 5 Deadliest Squares on the Board:
| Rank | Square | Total Deaths ($D_i$) | Empirical Prob $p_{\text{emp}}(i)$ | Ratio vs Uniform Expectation |
| :---: | :---: | :---: | :---: | :---: |
| **#1** | **d4** | **3,332** | **7.54%** | **4.83×** |
| **#2** | **d5** | **3,231** | **7.32%** | **4.68×** |
| **#3** | **e5** | **2,339** | **5.30%** | **3.39×** |
| **#4** | **e4** | **2,062** | **4.67%** | **2.99×** |
| **#5** | **c4** | **1,421** | **3.22%** | **2.06×** |

> **Critical Spatial Insight:** The four central squares (**d4, d5, e5, e4**) account for **24.8% of all casualties** in chess!

---

### 2. Deep Dive: Square **d4** (The Epicenter of Chess Carnage)

#### A. Who Dies on d4? (Piece Breakdown)
- **Pawns:** $2,245$ ($67.38\%$) — Pawn breaks like `...cxd4` and `...exd4`.
- **Knights:** $640$ ($19.21\%$) — Knights recapturing on d4 (`Nxd4`) or pinned in the center.
- **Bishops:** $197$ ($5.91\%$) — Central bishop trades (`Bxd4`).
- **Queens:** $136$ ($4.08\%$) — Early queen skirmishes and central queen trades.
- **Rooks:** $114$ ($3.42\%$) — Heavy-piece central file invasions.

#### B. What Openings & Movesets Cause the Carnage on d4?
1. **Sicilian Defense (ECO B54, B23, B40, B44):**  
   The single largest contributor to deaths on d4. Moveset:
   $$\text{1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4}$$
2. **French Defense Advance Variation (ECO C02):**  
   Black challenges White's d4 pawn center with the `...c5` break:
   $$\text{1. e4 e6 2. d4 d5 3. e5 c5 4. c3 Nc6 5. Nf3 Qb6}$$
3. **Queen's Gambit & Rapport-Jobava System (ECO D01, D02):**  
   Early skirmishes over White's initial d4 pawn wedge.

#### C. The Brawl in the Square: Who Wins the War?
A **brawl** is defined as an immediate sequence of captures on the same square (within 2 plies).  
Across 1,251 multi-capture brawls on d4:
- $87.7\%$ are equal-material exchanges (e.g. Pawn for Pawn, Knight for Knight).
- In asymmetric trades where one side wins the material exchange on d4:
  - **If Black wins the brawl on d4:** Black wins the entire match **$54.7\%$** of the time ($P(\text{Black Win} \mid \text{Won Brawl on d4}) = 0.547$).
  - **If White wins the brawl on d4:** White wins the match **$47.1\%$** of the time (with high draw / defensive counterplay).

---

## 🖥️ Interactive Visualizations

The project generates 3 standalone, responsive interactive HTML applications:

1. **[part1_match_duration_distribution.html](file:///c:/Users/30697/OneDrive/Έγγραφα/Εργασιες%20σπουδων/chess/part1_match_duration_distribution.html):**  
   - Interactive Chart.js distribution curves with empirical density histogram.
   - Hover cursor calculates dynamic win probabilities ($P(\text{White})$, $P(\text{Black})$, $P(\text{Draw})$) and analytical formula tooltips in real-time.
2. **[part2_chessboard_death_heatmap.html](file:///c:/Users/30697/OneDrive/Έγγραφα/Εργασιες%20σπουδων/chess/part2_chessboard_death_heatmap.html):**  
   - Interactive 8x8 bloody chessboard with visceral arterial red heatmap scaling with $p_{\text{emp}}(i)$.
   - Clicking/hovering on any square reveals its full tactical dossier: victim piece distribution, rankings, top opening lines, and brawl victory correlation.
3. **[index.html](file:///c:/Users/30697/OneDrive/Έγγραφα/Εργασιες%20σπουδων/chess/index.html):**  
   - Master CV Portfolio application integrating Part 1, Part 2, and executive project overview.

---

## 📁 Repository Structure

```
├── data/
│   ├── rapid_10min_matches.jsonl       # 2,500 real Lichess 10+0 games with moves & clocks
│   ├── part1_results.json              # MLE fit statistics, AIC/BIC, KS tests, win rates
│   └── part2_results.json              # 64-square death counts, p_emp, brawls, piece breakdowns
├── figures/
│   ├── part1_match_duration_distribution.png  # 300 DPI publication figure for Part 1
│   └── part2_chessboard_death_heatmap.png     # 300 DPI publication figure for Part 2
├── download_data.py                    # Automated crawler for 10-minute rapid matches
├── part1_analysis.py                   # Stochastic duration fitting & hypothesis testing
├── part2_analysis.py                   # Spatial piece death probability & brawl analyzer
├── generate_visualizations.py          # Interactive HTML application compiler
├── generate_figures.py                 # Matplotlib publication chart renderer
├── part1_match_duration_distribution.html  # Interactive Part 1 tool
├── part2_chessboard_death_heatmap.html     # Interactive Part 2 tool
├── index.html                          # Master CV Portfolio Dashboard
└── README.md                           # Documentation & CV writeup
```

---

## 🚀 How to Run & Reproduce

```bash
# 1. Install dependencies
pip install numpy scipy pandas matplotlib chess

# 2. Run statistical models
python part1_analysis.py
python part2_analysis.py

# 3. Generate interactive HTML dashboards and high-res figures
python generate_visualizations.py
python generate_figures.py

# 4. Open index.html in any web browser to view the interactive portfolio!
```
