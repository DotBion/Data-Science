# What I Learned

A consolidated account of the techniques these four projects taught me, the mistakes I
made along the way, and the single idea that ties them together.

---

## The through-line

I started the course thinking the job was to make a number go up — accuracy, then AUC.
I finished it convinced that the job is to make a **decision** better, and that the
model is one input to that decision among several.

The clearest illustration is project 03. Three models scored between 0.805 and 0.833
AUC, a spread narrow enough to be noise. Choosing between them changed almost nothing.
Changing the *targeting threshold* from the budgeted 25% of customers to the
profit-maximising 42% changed the outcome enormously — and the threshold is not a
property of the model at all. It falls out of the cost of a wrong offer versus the
value of a saved customer.

That reordering of priorities is the main thing I took away.

---

## Data preparation

**Missing data and invalid data are separate problems.** The no-show dataset has zero
nulls and 205 impossible rows: a negative age, appointments scheduled after they
occurred, and a documented 0/1 flag containing values up to 4. `isna().sum()` finds
none of these. Histograms, `value_counts()`, `describe()`, and min/max checks against
what the values *physically mean* find all of them.

**Check the target's bounds first.** A graduation rate above 100% is not an outlier to
model around, it is a broken record.

**Encoding categoricals:**
- `pd.get_dummies(..., drop_first=True)` to avoid perfect multicollinearity.
- Watch for *semantic* redundancy that `drop_first` does not catch. In the churn data,
  `OnlineSecurity_No internet service` is fully determined by `InternetService_No`;
  six such columns had to be identified by reading the data dictionary, not by running
  a correlation check.
- High-cardinality columns need collapsing. Bucketing every neighbourhood with fewer
  than 2,000 appointments into `OTHER` took ~80 dummy columns down to 22.

---

## Feature engineering

This is where the largest gains came from, and it is the part no library does for you.

**Temporal differences.** Neither `ScheduledDay` nor `AppointmentDay` predicts much
alone. `TimeInAdvance`, the gap between them, is one of the strongest signals in the
dataset.

**Historical aggregates without leakage.** To use a patient's past no-shows as a
feature, the count must exclude the appointment being predicted:

```python
df = df.sort_values(['PatientId', 'ScheduledDay'])
df['PreviousNoShows'] = df.groupby('PatientId')['No-show'].cumsum() - df['No-show']
```

Sort chronologically, cumulative-sum within the group, subtract the current row. The
general principle — *only use information that existed at prediction time* — is the
one I expect to re-derive most often.

**Text as features.** TF-IDF with bigrams turns prose into 97,619 numeric columns.
Bigrams are what let the model distinguish "not good" from "good".

---

## Models I used, and when

| Model | Used in | Good for |
|---|---|---|
| Linear regression | 01 | Numeric target, interpretable coefficients, honest baseline |
| Decision tree | 02, 03 | Rules you can read aloud; captures interactions without manual terms |
| Logistic regression + L1/L2 | 03, 04 | Probabilities, signed coefficients, strong on wide sparse data |
| Random forest | 03 | Zero-tuning baseline |
| XGBoost | 04 | High-capacity benchmark |

Two lessons about model choice:

**Depth is a communication decision as much as a statistical one.** The depth-3 tree in
project 02 exists so it can be plotted and handed to a clinic manager.

**A regularised linear model is a serious competitor, not a strawman.** It matched
XGBoost on 98k sparse text features and beat a random forest on churn.

---

## Tuning and validation

- **Train/test split** on every project, with a fixed `random_state` for
  reproducibility.
- **`GridSearchCV`** over `max_depth` × `min_samples_leaf`, 5-fold, scored on
  `roc_auc` — tuning inside cross-validation so the test set stays untouched.
- **Regularisation strength is a hyperparameter with real consequences.** Lasso at
  `C=0.1` beat weakly-penalised fits and zeroed four coefficients outright, performing
  feature selection as a side effect.
- **Test error exceeds training error**, reliably, and the size of the gap is
  diagnostic.

**A mistake I made:** in project 03 I selected the logistic regression's `C` by
scoring each candidate on the test set. That contaminates the held-out data and makes
the resulting AUC optimistic. The tree in the same notebook was tuned correctly. Having
both, side by side, in one notebook is a useful permanent reminder.

---

## Evaluation, and why accuracy is not enough

**Accuracy without a base rate is meaningless.** 79.9% on no-shows looks fine until you
notice ~80% of patients attend anyway.

**The metrics that replaced it:**
- **AUC / ROC** — threshold-independent ranking quality, and the right tool for
  plotting several models on one axis.
- **Precision and recall** — at a 0.30 churn threshold: precision 0.53, recall 0.75.
  Catching three-quarters of churners while wasting half the offers is a *choice*, and
  whether it is a good one depends entirely on the price of an offer.
- **F1** — one number when precision and recall genuinely matter equally, which is
  rarer than it is used.
- **RMSE, MAE, R²** for regression, always interpreted in the units of the problem.

**Predicted-probability histograms** turned out to be the most practical diagnostic for
choosing a threshold: they show where the model's confidence actually concentrates,
which is rarely near 0.5.

---

## Cost-benefit analysis and profit curves

The material I value most, and the part that is genuinely about business rather than
statistics.

**Build a cost/benefit matrix with the same shape as the confusion matrix.** Each cell
gets a dollar value:

|  | Actually churns | Actually stays |
|---|---|---|
| **Offer sent** | +$577.58 (saved, net of the $200 offer) | −$200 (wasted offer) |
| **No offer** | $0 | $0 |

Element-wise multiply, sum, and the confusion matrix becomes a profit figure.

**Then improve the ranking — and check that you actually did.** Expected value per
customer should be `P(churn) × that customer's annual_revenue − offer_cost`, so that a
$118/month customer outranks an $18/month one at equal churn risk. I wrote the formula
with the *population average* revenue instead, which is a constant: the ranking that
came out was the plain probability ranking, unchanged. The profit figure still rose by
about a third, but that came from valuing the saved customers correctly after the fact,
not from targeting different people. Reporting it as a ranking improvement was the
error; a ranking change you cannot point to in the sort order did not happen.

**Then improve the ranking.** Using each customer's own monthly charges instead of the
population average, expected value per customer becomes
`P(churn) × annual_revenue − offer_cost`. Targeting in that order raised profit by
roughly a third with no change to the model and no change to the budget.

**And benchmark against doing nothing clever.** Comparing against random targeting of
the same number of customers is what converts "the model is good" into "the model is
worth this much".

---

## Text and NLP

- Cleaning pipeline: strip HTML → remove punctuation, URLs, digits → tokenise →
  remove stopwords → lemmatise.
- **Keep the raw text in a separate column.** Every worthwhile question afterwards
  needed it.
- **Fit the vectorizer on training data only.** Vocabulary and IDF weights are learned
  parameters and leak exactly like model parameters do.
- **Lexicon methods like VADER** are a free independent check on a supervised model.

---

## Interpretability

- **Feature importances (trees)** and **coefficients (linear models)** answer different
  questions and can disagree. On churn, the tree put 83% of its importance on two
  features while the regression spread it across contract length and add-on services.
  Correlated features dilute tree importances in a way coefficients resist.
- **`eli5`** shows the terms driving a text model globally, and the tokens driving one
  specific prediction.
- **Plotted trees** are the most persuasive artefact I produced for a non-technical
  reader.
- **Read your confident mistakes.** Sorting test predictions by probability and
  examining the extremes surfaced a sarcastic 10/10 review written entirely in negative
  vocabulary — an error that no additional data would fix, because bag-of-words cannot
  encode irony. Knowing which errors are irreducible under your representation tells you
  when to stop tuning.

---

## What I would do differently

1. **Never touch the test set while tuning.** Use `GridSearchCV` or a separate
   validation split for every hyperparameter, including regularisation strength.
2. **Fix the target encoding before interpreting anything.** In project 02 I inverted
   the no-show label, which leaves the accuracy intact and every interpretation
   backwards.
3. **Actually drop the rows I flag as invalid.** Counting 199 bad records and then
   leaving them in the dataframe is worse than not checking.
4. **Compute every profit number on one consistent population.** Mixing test-set counts
   with full-dataset profit arrays produced figures I can no longer defend.
5. **Use pipelines.** Chaining preprocessing and the estimator into a `Pipeline` would
   have prevented most of the leakage above by construction, rather than by vigilance.
6. **Report an operating point, not just AUC.** AUC ranks; a business needs a threshold
   and the precision/recall that comes with it.
