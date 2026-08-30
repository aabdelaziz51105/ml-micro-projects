# Spam Classification — Regularization and Feature Scaling

Logistic regression on the UCI Spambase dataset, used as a vehicle for working out what
regularization actually does: sweeping the penalty strength across eight orders of magnitude,
reading the resulting curve honestly, and testing whether the unpenalized model even has a
solution.

**Data:** [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) — 4,601 emails, 57
features (48 word frequencies, 6 character frequencies, 3 capital-run-length measures), binary
target. 39.4% spam. Included here as `spambase_csv.csv` under **CC BY 4.0**: Hopkins, M.,
Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase*. UCI Machine Learning Repository.
https://doi.org/10.24432/C53G6X

**Method:** 80/20 split, `StandardScaler` + `LogisticRegression` in a `Pipeline`, 5-fold
cross-validation over `C` from 1e-4 to 1e4, then an unpenalized fit at two iteration budgets.

**Baseline: 0.6060** — always guess "not spam". Every accuracy below reads against that, not
against zero.

## Findings

**The most interesting result was that my first proof was wrong, and it was wrong because it
was dramatic.** Comparing coefficients from a scaled fit against an unscaled one produced three
sign flips. That looked decisive: under a pure change of units the scaled coefficient is the
unscaled one times the feature's standard deviation, and a standard deviation is always
positive, so signs *cannot* change. Except the unscaled model in that comparison had stopped at
the default `max_iter=100` without converging — so two things were varying at once, scaling and
convergence. Against a properly converged unscaled model, **no coefficient flips sign at all.**
The flips were an artifact of the confound. Fixing the comparison deleted the headline result
and forced the claim to be re-proved by measurement.

**It survived, with quieter evidence.** Pure rescaling requires `scaled = unscaled × σ`
exactly. It doesn't hold: the gap reaches **0.525 on `char_freq_$`**, about a third of that
coefficient. And the direction is informative — `capital_run_length_longest` (values in the
hundreds) came out *smaller* than rescaling predicts, `char_freq_$` (values near zero) came out
*larger*. The L2 penalty charges by raw coefficient magnitude and knows nothing about units, so
it had been sparing the first and suppressing the second. Not universal, though: `word_freq_hp`
moves the other way, because correlated features redistribute weight among themselves.

**A small coefficient is not an unimportant feature.** `capital_run_length_longest` has an
unscaled coefficient of 0.0081 — rank features by magnitude on unscaled data and it looks like
one of the least useful columns in the set. Scaled, it's 0.907, among the largest. A
coefficient is per-unit: a feature measured in the hundreds needs a tiny one to contribute
anything at all.

**The failure mode was silence.** Unscaled, the model needed **3,074** iterations to converge;
scaled it needed **32**. At the default budget of 100 the unscaled fit is truncated, returns an
object, scores 0.9251, and raises nothing — the only signal is a warning you have to know how
to read. The truncated model is diagnosably underfit by an absent train advantage (0.9207 train
against 0.9251 test); a model that never finished fitting has no reason to favour the data it
was fitted on. Stopping an optimizer early is itself a form of shrinkage.

**Accuracy was never the story.** Scaling moved test accuracy by +0.005 — about five emails on
a 920-row test set, comfortably inside what a different train/test split would move it. What
scaling changed was whether the model was fitted at all.

**The regularization sweep has no U-curve.**

| C | 1e-4 | 1e-2 | 1 | 10 | 100 | 1e4 |
|---|---:|---:|---:|---:|---:|---:|
| ‖w‖ | 0.19 | 1.81 | 6.55 | 13.11 | 27.23 | 39.01 |
| CV accuracy | 0.705 | 0.912 | 0.930 | 0.931 | **0.933** | 0.932 |

Coefficient size and accuracy agree at the crushed end and stop agreeing entirely past C ≈ 1:
from there ‖w‖ grows sixfold and buys **+0.0033** accuracy. Cross-validated accuracy never
falls, and the train-minus-CV gap stays flat at 0.005 throughout. Nothing overfits.

**Why nothing overfits turned out to be the best finding.** Switching the penalty off entirely
gives `n_iter_` = 77 and ‖w‖ = 38.9537 — **identical at `max_iter=1000` and `max_iter=100000`.**
The unpenalized model converges and settles. That only happens when the classes overlap: if a
hyperplane could separate them perfectly, growing the weights would sharpen every prediction
forever and there would be no finite optimum at all. So:

> **Whether unregularized logistic regression converges is a property of the data, not the
> algorithm.** Separable → no finite solution, weights diverge. Overlapping → finite maximum
> likelihood, converges normally.

I had previously measured the other case on a small separable dataset, where ‖w‖ ran 3.10 →
16.01 between 1,000 and 1,000,000 iterations with the cost still falling. Two contrasting
measurements, three weeks apart.

**Which means regularization has two jobs and only one of them applied here** — making the
problem well-posed at all (needed on separable data; not needed here), and the ordinary
bias–variance trade (all it was doing). ‖w‖ at C=1e4 is 39.01 against 38.95 unpenalized, so the
right-hand edge of the sweep *is* the no-penalty model — and that model is perfectly
well-behaved, which is why nothing collapses.

**Picking `C` honestly is harder than picking the maximum.** The peak is 0.9329 at C=100, with
neighbours at 0.9313 and 0.9323 — differences of 0.0016 and 0.0005 against a **fold-to-fold
standard deviation of 0.0070**. The candidates are four to fourteen times closer together than
the noise in measuring any one of them, so `argmax` is selecting the luckiest fold split rather
than the best model. I'd ship **C=1**: no meaningful gain past it, so prefer the more
regularized model. The formal version of that is the one-standard-error rule, and it's
ambiguous here — using the standard error the threshold is 0.9298 and C=1 misses it by 0.0002
(the rule picks C=10); using the raw standard deviation the threshold is 0.9259 and C=1
qualifies. The two conventions disagree on this data.

**One thing worth knowing about the warnings.** During the sweep, `ConvergenceWarning`s fired
at high `C` while the fits I was measuring reported 75–78 iterations against a limit of 100 —
they had converged. The warnings came from `cross_val_score`'s refits on the folds, a different
set of models entirely. Attributing a warning to the wrong object is an easy way to write down
a false conclusion.

## In plain language

The goal was to create a classification model that predicts whether or not an email is spam.
The way the model works, in simple terms, is that it learns how much each word, symbol or
sequence counts toward "this is spam" and adds them up.

The dataset has 57 signals and only a few thousand emails, therefore it is prone to overfit
instead of generalize. In other words, it can learn things that were true of these particular
emails without being true of emails in general. Here is where regularization comes in.
Regularization penalizes a model for relying too much on one single signal. The `C` value here
is like a knob, from "the model is suppressed, barely allowed to have opinions" up to "no
restraint at all".

When experimenting with different `C` values, a low `C` scored about 70%, barely above the
baseline of about 61% (which is derived from the dataset split — 61% of emails aren't spam).
Turning the knob up brought it to about 93%. What didn't happen, however, was a drop in
accuracy at an extreme `C` value; anything after `C` = 1 was effectively the same. Removing the
penalty entirely confirmed it: the model settled on an answer instead of growing more and more
extreme forever, which is what it would have done if a clean dividing line existed. That told
us something about the data rather than the algorithm — there's no clean dividing line, some
spam reads like ordinary email and some ordinary email reads like spam.
