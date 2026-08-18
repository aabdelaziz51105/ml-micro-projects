# Player Age vs Market Value — Polynomial Regression

A from-scratch look at how a footballer's market value moves with age, fitting polynomial
regression of degree 1 through 12 on 423,823 (player, valuation-date) pairs from
Transfermarkt.

**Data:** `players.csv` and `player_valuations.csv` — [Football Data from
Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) (Kaggle,
`davidcariboo/player-scores`, built on
[dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets),
CC0-1.0). Predictor: age in years at the valuation date. Target: market value in EUR.

**Method:** single 80/20 train/test split, `PolynomialFeatures` + `LinearRegression` swept
across degree 1–12, scored by R² and RMSE on held-out data.

## Findings

**The well-behaved model fails *inside* the data it was trained on.** The degree-2 model —
the one that actually generalizes best — predicts negative market value for every player
aged 35 and older, even though the training data contains plenty of real 35+ players with
positive values. A parabola is symmetric and unbounded below, and nothing in ordinary least
squares knows the target has a floor at zero. The tails are also sparse — few players are
that old — so the fit is effectively unconstrained there while still technically "inside the
data." Being inside the data range and being *supported* by the data turn out to be
different things.

**Extrapolation fails in two ways, and the quiet one is worse.** Pushed 30% past the edge of
the data, the degree-12 model explodes to €8.4 billion at age 60 — absurd, and impossible to
miss. The degree-2 model returns −€34M at the same age: wrong, but smooth and plausible-
looking. The loud failure is the safer one, because it's the one you'd actually catch.

**Model disagreement as an uncertainty signal.** The degree-2 and degree-12 models agree to
within about 1% at age 30, where the data is dense, and diverge by nine orders of magnitude
by age 60, where it isn't. With no ground truth past the edge of the data, the spread between
two models that both fit the training data well *is* the uncertainty estimate.

**423,823 rows is more than a low-degree polynomial can overfit — and that's a result, not a
failure to find one.** Train and test error tracked each other almost exactly across the
whole degree sweep, no matter how high the degree went. Checked against a synthetic dataset
built to the same shape with n=300: the usual overfitting gap opens immediately at that
size. A flat validation curve isn't automatically evidence that a model is safe — sometimes
it just means the model doesn't have enough parameters to overfit that much data.

**R² and RMSE can disagree about which split scored better.** R² is a ratio against a
baseline computed from whichever set is being scored, so it isn't directly comparable across
datasets or splits; RMSE is absolute, and is.

## Chosen model

Degree 2 — both on the visual read (a clean, opening-down parabola) and the sweep (degree 4
scored marginally better on test RMSE, but by well under 1%, and the ranking wasn't stable
across random seeds).

## What I'd trust this for

Roughly the right shape for ages 18–34, where the data is dense — value rising into a
player's mid-to-late 20s and declining after. Outside that range I wouldn't trust a
prediction from it at all; the negative prices past 35 are proof the fit is unconstrained
where the data thins out. This is a demonstration of what polynomial regression can and can't
do, not a valuation tool.
