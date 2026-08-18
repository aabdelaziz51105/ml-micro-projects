# Food Delivery Time — Multiple Linear Regression

Predicting delivery time from 43 predictors (17 raw features, 8 of them categorical) on a
50,000-row food delivery dataset, then working through what backward elimination and
multicollinearity diagnostics actually do — and don't do — on a real-shaped dataset.

**Data:** [Food Delivery Time Prediction
Dataset](https://www.kaggle.com/datasets/dharmendrapandit12/food-delivery-time-prediction-dataset)
(Kaggle, synthetic, 50,000 orders). Target: delivery time in minutes.

**Method:** one-hot encoding with `drop='first'`, `train_test_split`, `LinearRegression`,
then backward elimination by p-value and VIF for multicollinearity.

## Findings

**The target turned out to be censored at 180 minutes — found by reading a plot, not a
metric.** The residuals-vs-fitted plot has a sharp straight edge, slope exactly −1, cutting
through the cloud. Working out what produces that: if a residual falls by exactly one unit
for every unit the prediction rises, the actual value isn't varying at all — it's pinned.
Checked against the raw data: 1,661 of 50,000 rows (3.3%) sit at exactly 180 minutes, against
436 rows across the entire nine minutes below it. That's not a long tail, it's a wall —
deliveries that would have taken longer were recorded as capped at 180. Nothing else in the
workflow surfaces this: R² was 0.930 (looks great), p-values were mostly significant,
backward elimination ran cleanly and dropped 21 columns. Only the residual plot showed that
3.3% of the target was fabricated.

**The model is a plane fit to a hyperbola.** Delivery time is roughly preparation time plus
distance ÷ speed — a ratio. A linear model can only fit distance and speed as separate
additive terms: a flat plane approximating a curved surface. That single structural mismatch
explains three things that look unrelated at first glance: R² capping around 0.93 instead of
going higher, heavy-tailed non-normal residuals (kurtosis 7.2 vs. 3 for a normal
distribution), and speed's implausibly large t-statistic and VIF — it isn't just correlated
with the target, it's literally part of the formula that generates it. The fix is one line of
feature engineering (add distance/speed as a column), not a different model family.

**VIF is a property of the whole variable set, not of a single predictor.** After dropping 7
of 8 cuisine-type dummy columns, `Preparation_Time_Min`'s VIF collapsed from 17.2 to 1.03 —
without that variable itself changing at all. Its apparent collinearity lived entirely in the
columns removed around it. Meanwhile a dominant one-hot category (`Vehicle_Type_Bike`, 46% of
rows) sat at a VIF of 70.7 throughout, unmoved by anything else that was dropped — dummies
from the same categorical are *necessarily* correlated with each other by construction.
Applying "VIF over 10 means drop it" indiscriminately would have deleted one of the model's
strongest predictors.

**Backward elimination can quietly break a categorical variable.** Column-by-column
p-value elimination left `Day_of_Week` as a fragment — Saturday and Sunday survived, Monday
through Thursday didn't — because each dummy's p-value only tests "is this day different from
the baseline," never "does day-of-week matter as a whole." The right check is a joint F-test
across all six day dummies at once. Elimination also removed 21 of 43 predictors while
changing test R² by 5×10⁻⁸ — a real result, not a null one: the columns dropped were already
contributing nothing.

## What I'd trust this model for

This project was about diagnostics more than model selection, so there's no single "chosen
model" the way a degree gets chosen in a polynomial fit. The honest summary: the linear model
is a reasonable approximation of a relationship that's structurally non-linear (a ratio), it's
wrong for about 3.3% of rows in a way no standard metric catches, and its multicollinearity
warnings need to be read at the level of *variable blocks* (all the dummies from one
category together), not individual columns in isolation.
