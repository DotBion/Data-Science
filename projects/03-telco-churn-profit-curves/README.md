# 03 — Telco Churn: From a Classifier to a Targeting Decision

**Notebook:** [`telco_churn_prediction.ipynb`](telco_churn_prediction.ipynb)
**Also here:** [`telco_churn_working_draft.ipynb`](telco_churn_working_draft.ipynb) — an
earlier, partially completed pass at the same problem, kept for the record.

This is the project I would show first. It is the only one where the modelling is the
easy half.

## The question

MegaTelCo has a retention offer that works: every customer who receives it renews for
another year. It costs **$200 per customer**, and marketing has budgeted enough to
send it to the **top 25%** of customers whose contracts are expiring.

Two questions follow. Which customers should get it? And — the more interesting one —
is 25% actually the right number?

## Data

7,032 customers with 19 attributes: tenure, monthly charges, contract type, payment
method, internet service type, and a set of add-on services (online security, backup,
device protection, tech support, streaming).

## What I did

### Preparation

One-hot encoded the categorical columns with `drop_first=True` to avoid the dummy
variable trap. The service columns needed a second pass: features like
`OnlineSecurity` have three levels (`Yes`, `No`, `No internet service`), and the
`No internet service` dummies are perfectly redundant with `InternetService_No`, so I
dropped them. `Churn` was mapped to 1/0 and an 80/20 split taken with
`random_state=42`.

### Three models

**Decision tree, grid searched.** `GridSearchCV` over `max_depth` ∈ [2, 12] and
`min_samples_leaf` ∈ [1, 100] step 10, with 5-fold cross-validation scored on
`roc_auc`. Best configuration: `max_depth=6`, `min_samples_leaf=91`, cross-validated
**AUC 0.833**.

**L1-regularised logistic regression.** Swept `C` over [0.01, 0.1, 1, 10, 100]. Best
at `C=0.1`, **AUC 0.830**. Four coefficients were driven exactly to zero — Lasso doing
feature selection as a side effect of regularisation.

**Random forest**, default hyperparameters, as an off-the-shelf baseline: **AUC 0.805**.

Plotted all three ROC curves together. Logistic regression won on the test set, so it
became `best_model`.

### What the two models disagree about

The tree and the regression tell noticeably different stories about *why* customers
leave:

| Tree — feature importance | | Logistic regression — largest negative coefficients | |
|---|---|---|---|
| `tenure` | 0.481 | `Contract_Two year` | −1.28 |
| `InternetService_Fiber optic` | 0.346 | `InternetService_No` | −0.92 |
| `InternetService_No` | 0.038 | `Contract_One year` | −0.75 |
| `Contract_One year` | 0.031 | `OnlineSecurity_Yes` | −0.37 |
| `MonthlyCharges` | 0.029 | `PhoneService_Yes` | −0.33 |
(`TechSupport_Yes` is next at −0.332, just behind `PhoneService_Yes` at −0.335 — the
add-on-services reading holds, but `PhoneService_Yes` is the one that actually places
fifth.)

The tree concentrates almost 83% of its importance in two features; the regression
spreads credit across contract length and add-on services. Both agree on the direction
of the story — long contracts and bundled services retain customers, fiber optic
customers churn more — but tree importances are diluted by correlated features in a way
that coefficients are not. Reading one without the other would have been misleading.

### The part that actually matters: cost and benefit

A confusion matrix counts errors. A business does not care about counts, it cares about
dollars, and the four cells are not worth the same amount:

- Offer a customer who would have churned → we keep them. Annual revenue is
  12 × mean `MonthlyCharges` = 12 × $64.80 = **$777.58**, minus the $200 offer = **+$577.58**.
- Offer a customer who was staying anyway → **−$200**, pure waste.
- Do not offer → **$0** either way. We neither spend nor save.

Multiplying that matrix element-wise against the confusion matrix at a 0.30 threshold
(precision 0.53, recall 0.75, F1 0.62) gives about **$112,700** of profit on the test set.

### The profit curve

Sorting customers by predicted churn probability and walking the threshold from
strictest to loosest, computing profit at every cut point, produces a curve with a
clear interior maximum:

**Maximum profit ≈ $114,900, reached by targeting ≈ 42% of customers.**

An independent implementation using cumulative response curves agrees: $112,700 at
42.6% for logistic regression, $106,300 at 41.8% for the tree.

That is the whole argument. The budgeted 25% is not on the peak, and the curve is the
evidence for widening it. Because a false positive costs $200 while a true positive
earns $578, being wrong is nearly three times cheaper than being right is valuable —
which is exactly why the optimum sits well past the conservative cut.

### Individualised expected value

The analysis above uses the *average* monthly charge for everyone, but a $118/month
customer is worth far more to save than an $18/month one. Recomputing profit with each
customer's own annual revenue raised the reported figure from **$589,000 to $813,000**
across the same 2,917 customers.

The framing is right but the code does not deliver it. The expected value is computed
as `churn_probabilities * (avg_monthly_charge * 12) - offer_cost` — the *average*
charge, not the individual one. Multiplying every probability by the same constant is a
monotone transform, so sorting by that expected value produces exactly the probability
ranking it was supposed to improve on. Both policies target an identical set of
customers in an identical order. The entire $224,000 difference comes from the benefit
*accounting*, not from a better *decision*. Doing this properly means putting
`df['MonthlyCharges']` into the expected value itself, which is what the exercise asked
for.

## Results summary

| Model | AUC |
|---|---|
| Logistic regression (L1, C=0.1) | **0.830** |
| Decision tree (depth 6, min leaf 91) | 0.833 cross-validated |
| Random forest (default) | 0.805 |

On the 1,407-customer test set:

| Targeting policy | Customers targeted | Profit |
|---|---|---|
| Fixed 0.30 probability threshold | 37.6% | ~$112,700 |
| Profit-curve optimum | 42.4% | **~$114,900** |

The budgeted 25% falls short of the peak on that curve, which is the substance of the
recommendation. Two further figures — the individualised expected-value result and the
comparison against random targeting scaled to a 100,000-customer book — come from cells
with the defects noted below: they rank by probability rather than by individual value,
and they index profit arrays the notebook never defines. Treat the direction as sound
and the magnitudes as unusable.

## What I learned

- **A classifier is not a decision.** The model outputs a probability; the decision is
  where you cut it, and that cut belongs to the cost structure, not to the model.
- **0.5 is an arbitrary threshold.** It is only optimal when false positives and false
  negatives cost the same, which is almost never.
- **Asymmetric costs move the optimum.** $578 upside against $200 downside is why the
  answer is 42% and not 25%.
- **Regularisation strength is a real hyperparameter.** Lasso at `C=0.1` beat the
  barely-penalised fits, and zeroed four coefficients on the way.
- **Grid search with cross-validation** is how you tune without burning the test set —
  a rule I followed for the tree and broke for the regression (see below).
- **The expected-value framing generalises.** Any per-customer intervention with a
  known cost and a modellable benefit fits the same template.
- **Reporting a dollar figure invites scrutiny of your assumptions**, which is a good
  thing. Averaging revenue across customers was the weakest link, and fixing it was
  worth more than any modelling change.

## Known issues

- **`C` was selected on the test set.** The logistic regression sweep scores each `C`
  against `X_test`, so the reported 0.830 is optimistically biased and not directly
  comparable to the tree's cross-validated 0.833. The tree was tuned correctly with
  `GridSearchCV`; the regression should have been too.
- **The pitch cells depend on variables the notebook never defines.** Cells 55 and 60
  read `profits` and `thresholds`, and neither is assigned anywhere in the notebook —
  the only `profits` in the file is a local inside `plot_profit_curve`. They are
  leftover state from a cell that was edited or deleted, so the notebook raises
  `NameError` on a clean top-to-bottom run and every number those cells print
  ($148,299.56, $589,188.38, 2,917 customers, the $8.1M scaled comparison) is
  unreproducible from the file as committed.
- **Mixed denominators in the pitch calculation.** On top of that, the comparison
  indexes a 7,032-row profit array with a customer count derived from the 1,407-row
  test set. The *shape* of the argument (the optimum is near 42%, not 25%) is supported
  by the profit curve, but the dollar deltas in that cell should be recomputed on a
  single consistent population.
- **Profit is evaluated partly on training data.** Some cells score `best_model` on the
  full dataframe, including rows the model was fit on, which inflates the result.
- An early correlation heatmap raises a `KeyError` because it references the original
  categorical column names after they have already been replaced by dummies.
- Several exploratory cells re-fit trees with `criterion="entropy"` and hand-rolled
  loops that duplicate the grid search; they are scratch work, not part of the final
  argument.
- The `DecisionTreeClassifier` instances are constructed without `random_state`, so the
  tree figures are not exactly reproducible between runs.
- The working draft notebook stops after question 3 and is superseded by the main
  notebook throughout.
