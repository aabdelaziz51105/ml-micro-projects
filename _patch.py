import os
p = os.path.expanduser('~/mnt/ml-micro-projects/README.md')
s = open(p, encoding='utf-8').read()
new = '''- [`company-revenue-support-vector-regression/`](company-revenue-support-vector-regression/) —
  support vector regression on the 2022 Fortune 1000. Sweeping the ε-insensitive tube from 796
  support vectors down to zero, and a plot that reads as underfitting while the model is
  memorising its training set.

More will be added as they're finished.'''
assert "More will be added as they're finished." in s
s = s.replace("More will be added as they're finished.", new)
open(p, 'w', encoding='utf-8').write(s)
print('top-level README updated')
