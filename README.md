# Data Science for Business — Applied Modelling Portfolio

Four end-to-end predictive modelling projects built while working through
**TECH-GB 2336: Data Science for Business** (NYU Stern). Each project starts from a
raw, messy dataset and finishes with a model plus a decision a business could
actually act on.

The thread running through all four is that **model accuracy is not the deliverable**.
A churn model with a 0.83 AUC is only interesting once you can say who to call, how
many people to call, and how many dollars that produces. The last project in this
repo spends more effort on the profit curve than on the classifier.

- **What each project is:** see the table below, and the README inside each folder.
- **What I took away from the whole thing:** [`docs/skills-learned.md`](docs/skills-learned.md).

---

## Projects

| # | Project | Problem | Data | Model | Headline result |
|---|---------|---------|------|-------|-----------------|
| 01 | [College graduation rates](projects/01-college-graduation-rate-regression) | Predict a college's graduation rate from its admissions and finance profile | 777 US colleges, 17 predictors | Linear regression | Test RMSE 13.3 points, R² 0.39 — and a lesson in why that is a weak model |
| 02 | [Patient no-shows](projects/02-patient-no-show-prediction) | Predict which patients skip their appointment, so a clinic can staff on-demand doctors | 110,527 Brazilian clinic appointments | Decision tree | 79.9% test accuracy, plus a history feature built from each patient's own past |
| 03 | [Telco churn and retention targeting](projects/03-telco-churn-profit-curves) | Decide which expiring customers get a $200 retention offer | 7,032 telecom customers | Tuned tree, L1 logistic regression, random forest | AUC 0.83, and a profit curve arguing for targeting ~42% of customers rather than the budgeted 25% |
| 04 | [Movie review sentiment](projects/04-movie-review-sentiment) | Classify review text as positive or negative | 19,999 labelled movie reviews | TF-IDF + logistic regression, XGBoost | AUC 0.927, plus an error analysis that traced the model's worst miss to a sarcastic review |

Every notebook has an **Open in Colab** badge at the top, so you can run them without
installing anything.

---

## Repository layout

```
projects/
  01-college-graduation-rate-regression/   Linear regression, train vs. test error
  02-patient-no-show-prediction/           EDA, feature engineering, decision trees
  03-telco-churn-profit-curves/            Model comparison, cost/benefit, profit curves
  04-movie-review-sentiment/               NLP, TF-IDF, model explainability
docs/
  skills-learned.md                        Techniques picked up, and mistakes made
requirements.txt
```

Each project folder holds one notebook and a README covering the business question,
the approach, the results, and what I would do differently.

## Running the notebooks locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The datasets are not committed to this repository — they are hosted externally and
each notebook links to its own source. Project 01 expects `College.csv` in the
working directory; the rest download their data at runtime.

Project 04 additionally needs a few NLTK corpora, which its first cell downloads:
`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`, and `vader_lexicon`.

## P.S

These were coursework notebooks before they were a portfolio, and I have not gone
back and quietly fixed them. Where a notebook contains a bug, an unanswered
question, or a number I no longer trust, the project README says so under
**Known issues**. Being able to audit your own work is part of the skill.

## License

MIT — see [LICENSE](LICENSE).
