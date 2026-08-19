# 02 — Predicting Patient No-Shows

**Notebook:** [`patient_no_show_prediction.ipynb`](patient_no_show_prediction.ipynb)

## The question

SHMC is a medical centre in Brazil that books doctors on demand: once a doctor is
called in, they are paid whether or not the patient turns up. Every no-show is a
doctor paid for nothing. If we can predict which appointments will be missed, we can
staff the day correctly.

## Data

110,527 appointments from Brazilian public health clinics (April–June 2016), with
patient demographics, welfare status (`Scholarship`), four chronic conditions, whether
an SMS reminder was sent, the neighbourhood, and the no-show outcome.

## What I did

### Turning dates into a feature

`ScheduledDay` carries a timestamp while `AppointmentDay` is midnight-only, so
comparing them directly is misleading. I normalised `ScheduledDay` to a date and
derived **`TimeInAdvance`** — days between booking and appointment. The longest lead
time in the data is 179 days.

This is the most valuable feature in the dataset and it does not exist until you
create it. That was the lesson.

### Auditing the data instead of trusting it

Histograms and value counts across every column surfaced three classes of impossible
records:

| Problem | Rows | Why it can't be real |
|---|---|---|
| `Age` = -1 | 1 | Negative age |
| `TimeInAdvance` < 0 | 5 | Appointment dated before it was booked |
| `Handicap` > 1 | 199 | Documented as a 0/1 flag, but contains values up to 4 |

204 rows in total. There are no missing values anywhere in the dataset, which is
exactly why this step matters: `isna().sum()` returning all zeros tells you nothing
about whether the values that *are* present make sense.

### Taming a high-cardinality categorical

`Neighborhood` has 80-odd levels, most of them rare. One-hot encoding all of them
produces a wide, sparse matrix full of columns that appear a handful of times and
mostly encode noise. I collapsed every neighbourhood appearing fewer than 2,000 times
into a single `OTHER` bucket, leaving 21 named neighbourhoods plus `OTHER` (which
absorbs 43,878 appointments), then dummy-encoded the result.

### Building a history feature without leaking the future

The dataset has repeat patients, and someone who has missed appointments before seems
likely to miss the next one. The trap is that a naive per-patient no-show count
includes the appointment you are trying to predict.

```python
noshows = noshows.sort_values(['PatientId', 'ScheduledDay'])
noshows['PreviousNoShows'] = noshows.groupby('PatientId')['No-show'].cumsum() - noshows['No-show']
```

Sorting chronologically, taking a cumulative sum within each patient, then subtracting
the current row leaves a count of *strictly prior* no-shows. Plotting no-show rate
against `PreviousNoShows` shows the rate climbing with history, confirming the feature
carries signal.

### Modelling

A `DecisionTreeClassifier` with `max_depth=3` on an 80/20 split (`random_state=99`),
chosen shallow so the tree could be plotted and read as a set of rules rather than
treated as a black box.

## Results

**79.9% test accuracy.**

That number deserves suspicion rather than celebration, because roughly 80% of
appointments are attended anyway — a model that predicts "everyone shows up" would
score about the same. This is precisely the observation that motivates project 03's
switch to AUC, precision/recall, and profit-based evaluation.

## What I learned

- **Feature engineering beat model selection.** `TimeInAdvance` and `PreviousNoShows`,
  neither of which is in the raw file, are the two features a domain expert would ask
  for first.
- **Past-behaviour features are a leakage trap.** The `cumsum() - current_row` pattern
  is the general fix for "what did I know at the time?"
- **Clean data and valid data are different things.** Zero nulls, 204 impossible rows.
- **Accuracy is meaningless without a base rate.** 79.9% sounds respectable until you
  notice the majority class is 80%.
- **A depth-3 tree is a communication tool.** You can hand the plotted tree to a clinic
  manager and they can follow the logic.

## Known issues

These are real defects in the notebook, left in place rather than silently patched.

- **The target is inverted.** The exercise asks for 1 = did not show, but the code is
  `(noshows['No-show'] == 'No').astype(int)`, and in this dataset `No-show == 'No'`
  means the patient *did* attend. The column therefore encodes "showed up". The
  accuracy figure is unaffected, but every directional interpretation of the tree
  reads backwards, and `PreviousNoShows` is really "previous attendances".
- **Rows flagged were not all dropped.** The 199 `Handicap > 1` rows were counted and
  reported but never actually filtered out; only the 5 negative `TimeInAdvance` rows
  and the negative-age row were removed.
- **Four questions are unanswered** — the `max_depth` sweep from 2 to 50 scored by F1,
  the labelled confusion matrix with precision and recall, the feature-importance
  discussion, and the threshold-tuning exercise. The empty cells are still in the
  notebook. The threshold work in particular is what would turn 79.9% accuracy into
  something decision-useful.
- One narrative answer was written before its tree was plotted, so it reasons about
  rules in general terms rather than the specific splits the model found.
