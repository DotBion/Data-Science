# 01 — Predicting College Graduation Rates

**Notebook:** [`college_graduation_rate_regression.ipynb`](college_graduation_rate_regression.ipynb)

## The question

Given what we know about a college — how many students apply, how many are accepted,
tuition, room and board, faculty credentials, spending per student — can we predict
the share of its students who actually graduate?

This was the first modelling exercise of the course, so the point was less the answer
and more the shape of the workflow: load, clean, split, fit, evaluate, interpret.
Every later project reuses this skeleton.

## Data

`College.csv` from *An Introduction to Statistical Learning* — 777 US colleges and
universities, indexed by name, with 17 predictors and `GradRate` as the target.

## What I did

**Cleaning.** The original column names contain punctuation (`F.Undergrad`,
`S.F.Ratio`, `perc.alumni`), which makes them awkward to reference. I normalised them
with a small regex function that strips punctuation and collapses whitespace to
underscores.

**Encoding.** `Private` is the one categorical predictor, so `pd.get_dummies(...,
drop_first=True)` turns it into a single `Private_Yes` indicator.

**Sanity-checking the target.** A graduation rate is a percentage, so it has to sit
between 0 and 100. One row — Cazenovia College, at 118 — does not, and I dropped it.
This is a small thing but it is the habit worth keeping: check that your target
respects its own physical bounds before you fit anything to it.

**Modelling.** An 80/20 train/test split (`random_state=123`) and an ordinary
`LinearRegression`, evaluated on the held-out 20%.

## Results

| Metric | Test | Train |
|---|---|---|
| RMSE | 13.27 | 12.32 |
| MAE | 10.08 | 9.25 |
| R² | 0.391 | — |

Read in context, an RMSE of 13.3 means a typical prediction is off by roughly 13
percentage points of graduation rate — the difference between a struggling school and
a healthy one. This model is not good enough to make decisions with, and saying so
plainly is more useful than reporting R² and moving on.

The coefficients are more informative than the fit. Holding everything else constant,
being a private institution is associated with about 3.6 more percentage points of
graduation rate, and the strongest continuous contributors are the share of students
from the top 25% of their high school class, the student/faculty ratio, and the
percentage of alumni who donate — the last of which is plausibly a *consequence* of a
good student experience rather than a cause of it.

## What I learned

- **Training error is optimistic by construction.** Test RMSE (13.27) exceeded train
  RMSE (12.32). The gap was small here because linear regression on 17 features has
  little capacity to memorise 600 rows, but it establishes why the test set exists.
- **Interpret the metric in the units of the problem.** "RMSE = 13.27" means nothing
  on its own; "typically wrong by 13 percentage points" is a statement a dean can
  react to.
- **Predicted-vs-actual scatter beats a single number.** Plotting predictions against
  truth with a 45° reference line shows *where* the model fails, not just how much.

## Known issues

- The notebook's intro text is inherited from a template and mentions predicting car
  MPG; the work itself is entirely about `GradRate`.
- The scatter plot at the end has its axis labels commented out, and plots predictions
  on the x-axis against truth on the y-axis without labelling which is which.
- The optional extension exercises at the bottom (scatterplot matrix, boxplots by
  `Private`, an `Elite` binned feature, an `AcceptPerc` ratio feature) are listed but
  not attempted.
