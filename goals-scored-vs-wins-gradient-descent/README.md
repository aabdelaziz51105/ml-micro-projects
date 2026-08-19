# Goals Scored vs Wins — Gradient Descent from Scratch

Implementing cost function and batch gradient descent by hand — no library helpers until the
final comparison — to predict a football team's wins from goals scored across a season, then
checking the result against `sklearn.linear_model.LinearRegression`.

**Data:** `understat.com-selected-columns.csv` — season-level team stats (goals scored → wins),
compiled by hand from [understat.com](https://understat.com/), whose xG and match data are the
source for every figure in it. Predictor: goals scored across a season. Target: wins.

**What was built from scratch:** the cost function, the gradient computation, and the batch
gradient descent loop. `sklearn` is used only at the end, to check the answer.

## Result

The from-scratch fit converged to **w = 5.316029, b = 13.509472**, against scikit-learn's
closed-form solution of **w = 5.316259, b = 13.510055** — agreement to 5 significant figures.
It also reproduced an earlier, separately-run attempt at the same problem exactly, which is
the more interesting check: not just that the answer is right, but that the *method*
reproduces — two independent runs converged to the same place.

## What was harder than expected

Understanding gradient descent and implementing it turned out to be two different skills.
The algorithm is simple to state — and noticeably fiddlier to get right by hand than
expected, even with the underlying idea already clear going in. Data collection and cleaning,
normally the tedious part, ended up being the more enjoyable half of this one.
