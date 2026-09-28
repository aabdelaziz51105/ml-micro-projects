# Company Revenue — Support Vector Regression

Support vector regression on the 2022 Fortune 1000, used as a vehicle for working out what
`epsilon` and `C` actually do: sweeping the ε-insensitive tube across five orders of magnitude,
finding out why feature scaling barely mattered when it should have, and breaking the model on
purpose to see which part gives first.

**Data:** the 2022 Fortune 1000 by revenue — 1,000 companies, 996 after dropping four rows with
missing profits or headcount. Target is `revenues`, log-transformed. Nine predictors: profits,
assets, market value, employees, revenue and profit percent-change, rank change, and two
indicator columns built during cleaning.

**Method:** 80/20 split, `SimpleImputer` → `StandardScaler` → `SVR(kernel='rbf')` in a
`Pipeline`, with `y = log(revenues)` left unscaled. Baseline is `LinearRegression`. Everything
is scored in log units; `exp(RMSE)` converts an error into a multiplicative factor, which is the
only interpretable form when the target spans $2.1bn to $573bn.

---

## The baseline was broken, and one cell proved it

`LinearRegression` on the raw features scored **R² = 0.331** — implausibly low for predicting
company size from three other measures of company size. Two candidate explanations: SVR has a
huge opportunity here, or the target was logged and the predictors weren't, making a straight
line the wrong shape.

One extra fit settled it. Log-transforming the three strictly-positive size columns took the
*same* linear model from **0.331 to 0.730**.

| | R² | RMSE (log) | typical error |
|---|---:|---:|---:|
| `LinearRegression`, raw features | 0.331 | 0.875 | ×2.40 |
| `LinearRegression`, log features | **0.730** | **0.556** | ×1.74 |

That reframed the whole project. SVR gets the *raw* features; the linear model had to be handed
`log(assets)` by hand. So the question stopped being "does SVR win" and became **can the kernel
discover what I had to hand-engineer?** — written down before any SVR was fitted.

## The support vector count is not a diagnosis — it's a number you set

Sweeping `epsilon` at default `C` and `gamma`:

| ε | 0.001 | 0.01 | 0.1 | 0.32 | 1.0 | 10.0 |
|---|---:|---:|---:|---:|---:|---:|
| test R² | 0.666 | 0.667 | **0.673** | 0.659 | 0.548 | **−1.757** |
| support vectors | **796/796** | 789 | 682 | 435 | 99 | **0** |

Before running this I had three hypotheses about why 87% of the training data were support
vectors: SVR is wrong for this problem, SVR fits it well, or it's overfitting. All three treat
the count as a *symptom* — something the data has, which you read off to diagnose the model.

**It isn't. It's a direct consequence of a number I chose.** It moved from 796 to 0 without
touching the data, the kernel, or `C`.

`epsilon` sets how much error the model agrees to ignore. Inside the tube a point contributes
nothing and never becomes a support vector; outside it, the point gets a say. At `ε = 10` the
tube is wider than the entire 5.6-log-unit spread of the target, so no point qualifies, the sum
in the prediction formula is empty, and the model collapses to a bare intercept — hence
**R² = −1.757**, worse than predicting the mean.

The two failures aren't symmetric. `ε = 10` costs 2.43 of R²; `ε = 0.001` costs 0.007. If a
narrow tube were plainly overfitting those would be comparable. They aren't, because `C` caps
how hard any single qualifying point can pull. **`epsilon` decides *which* points get a say;
`C` decides *how much*.**

## Scaling the features barely mattered, and the reason is about the data

| | R² | support vectors |
|---|---:|---:|
| unscaled features | 0.635 | 691/796 |
| scaled features | 0.673 | 682/796 |

I predicted scaling would matter a lot with nine features. It moved R² by 0.038 and the support
vector count by 9.

`gamma='scale'` is `1/(p · Var(X))`, where `Var(X)` is a **single scalar pooled over the whole
array**. The per-column variances here span twelve orders of magnitude — `assets` at 7×10¹⁰
down to a binary indicator at 0.04. Since the RBF kernel sums squared differences across
columns, the distance between two companies on unscaled data is decided by `assets`,
`market_value` and `employees` alone; the other six are numerically invisible.

So the two rows are *a kernel that sees three columns* versus *a kernel that sees all nine* —
and all nine together buy 0.038 of R². **That's a statement about this dataset, not about SVR.**
Revenue, assets, market value and headcount are four measurements of the same underlying thing,
and the columns that could have said something *else* about a company barely register.

With one feature, `Var(X)` *is* that column's variance, the normalisation is exact, and scaling
is genuinely inert — which is why the textbook one-feature demo can't show any of this.

## Breaking it: SVR ignored a corrupted target completely

One training company's log-revenue was overwritten with increasingly absurd values. The real
range is 7.65 to 13.26.

| corruption | SVR R² | LinearRegression R² |
|---|---:|---:|
| none | 0.702 | 0.331 |
| y = 100 | 0.702 | 0.299 |
| **y = 1000** | **0.702** | **−1.859** |

Identical to three decimal places, every time. Mean absolute shift in test predictions at
y = 100: SVR **0.0025** log units, LinearRegression **0.1217** — a factor of 49.

I expected the ε-insensitive loss to absorb small corruption and eventually break, since the
corrupted point sits far outside any tube and *must* become a support vector. It never breaks.
It becomes a support vector and is then ignored anyway, and the mechanism is directly
inspectable: the dual coefficients are box-constrained to [−C, C], and that row's coefficient
comes out at exactly **10.000000** with `C = 10`. A company whose revenue was recorded as e¹⁰⁰⁰
gets precisely the same weight as any other saturated support vector. Squared-error loss has no
such bound, which is why one point destroys the linear fit.

## The most useful result: a plot that lies

Pushing `gamma` up:

| γ | 0.1 | 1 | 10 | 100 | 1000 |
|---|---:|---:|---:|---:|---:|
| train R² | 0.796 | 0.906 | 0.971 | 0.992 | 0.992 |
| test R² | **0.702** | 0.671 | 0.556 | 0.253 | **0.028** |
| span of the plotted curve | 1.98 | 1.58 | 1.94 | 0.97 | **0.06** |

I expected the standard picture: high `gamma` produces a wildly oscillating curve that spikes at
each training point. On the training data it's textbook overfitting — train R² climbs to 0.99
while test collapses to 0.03. **On the plot, the curve goes flat**, its span shrinking from 1.98
log units to 0.06 and settling at the intercept, which reads as maximal *under*fitting.

Both readings are correct about different things. The plot holds eight features at their median
and varies one. At γ = 1000 the kernel's effective radius is so small that K(x, x′) ≈ 0 unless
two points nearly coincide in *all nine* dimensions — and no point on that synthetic line is
that close to any real company, because real companies differ on the other eight axes. Every
kernel term vanishes and the prediction falls back to the intercept. The model has memorised the
training set and has nothing to say anywhere else, and "anywhere else" is exactly where the
plotted line runs.

**A flat partial-dependence slice is not evidence of underfitting.** Check the train/test gap,
which is a statement about where the data actually is.

## Did the kernel find what I hand-engineered?

| | test R² |
|---|---:|
| `LinearRegression`, raw features | 0.331 |
| `SVR(rbf)`, defaults | 0.673 |
| **`LinearRegression`, log features (hand-engineered)** | **0.730** |
| `SVR(rbf)`, tuned, `StandardScaler` (C=10, γ=0.1) | 0.703 |
| `SVR(rbf)`, tuned, `MinMaxScaler` (C=10, γ=100) | 0.732 |

**Roughly a draw.** A tuned RBF SVR on raw features lands within noise of a linear model handed
one log transform. The kernel found most of the curvature on its own — at the cost of a grid
search, a scaler choice that silently matters by 0.25 of cross-validated R², and a `gamma`
default that is wrong by two orders of magnitude under one of the two obvious scalers.

That last point is worth its own line. Moving from `StandardScaler` to `MinMaxScaler` moved the
optimal `gamma` by a factor of **1,000**, while `gamma='scale'` moved by **7**. It uses
*variance*, and min-max scaling on right-skewed columns crushes the bulk of the data near zero
while leaving a few stragglers at 1 — variance sees the stragglers, the kernel feels the bulk.
The median squared pairwise distance collapses 112× where the pooled variance drops only 7×.
Trusting the default under `MinMaxScaler` costs 0.25 of R² with no warning issued.

**One log transform, chosen off a skew statistic in five minutes, was worth about as much as all
of that.** Not an argument against SVR — an argument for looking at the data first.

## Caveats

- The test set is 200 rows. `MinMaxScaler` scores higher on test (0.732 vs 0.703) while
  cross-validation ranks it *lower* (0.677 vs 0.686); on 200 rows that gap is noise. The
  defensible claim is that the two scalers are equivalent once each is tuned for.
- `epsilon` was tuned at default `C` and `gamma`, then held at 0.1 through the grid. Re-tuning
  it afterwards moves the nominal optimum to 0.01, but the spread across 0.01–0.2 is 0.006.
- The target is truncated from below: the smallest companies here are at $2.1bn because that's
  where the Fortune 1000 stops. The model is only entitled to speak about companies above that.

## Data

`Fortune 1000 Companies by Revenue.csv` — the 2022 Fortune 1000, 1,000 rows, sourced from
Kaggle. Included here; redistribution terms checked before adding it.

Columns: `rank`, `name`, `revenues`, `revenue_percent_change`, `profits`,
`profits_percent_change`, `assets`, `market_value`, `change_in_rank`, `employees`. Money is
formatted with `$` and thousands separators, losses use accounting parentheses rather than a
minus sign, and `-` is the null marker — four different meanings of it, worked out in the
notebook's cleaning section.
