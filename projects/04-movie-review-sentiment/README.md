# 04 — Movie Review Sentiment Analysis

**Notebook:** [`movie_review_sentiment.ipynb`](movie_review_sentiment.ipynb)

## The question

Given 19,999 movie reviews already labelled positive or negative by human annotators,
can a supervised model recover the sentiment from the raw text alone — and when it
fails, can we work out whether the model or the label was wrong?

## What I did

### Text preprocessing

Reviews are scraped web text, so they arrive with HTML tags embedded. After stripping
those with a regex, each review goes through a cleaning pipeline:

1. Remove punctuation, URLs, digits, and collapse repeated whitespace
2. Lowercase and tokenise with NLTK
3. Drop English stopwords
4. Lemmatise with WordNet, so *watching*, *watched*, and *watch* collapse to one token

The cleaned text goes into a **new** `clean_review` column rather than overwriting
`text`. That decision paid off immediately: the error analysis below is only readable
because the original review survived preprocessing.

### Vectorisation

`TfidfVectorizer` with `min_df=2`, `max_df=0.7`, and `ngram_range=(1, 2)`, producing
**97,619 features** from 15,999 training documents.

The parameters do specific jobs. TF-IDF down-weights terms that appear everywhere, so
"movie" stops drowning out "brilliant". `max_df=0.7` discards terms present in more
than 70% of reviews. `min_df=2` drops hapaxes, mostly typos and proper nouns that
cannot generalise. Bigrams matter most of all here — unigrams cannot tell *"not good"*
from *"good"*, and negation is the core difficulty in sentiment.

Critically, the vectorizer is **fitted on the training split only** and merely applied
to the test split. Fitting on everything would leak test vocabulary and document
frequencies into training.

### Models

| Model | Test AUC |
|---|---|
| Logistic regression (L2, `C=0.1`, liblinear) | **0.927** |
| XGBoost (defaults) | 0.925 |

A linear model on TF-IDF matched gradient boosting, and trains in a fraction of the
time. With ~98,000 sparse features and 16,000 documents, the problem is close to
linearly separable and regularisation is doing the heavy lifting; the extra capacity
of a boosted ensemble has nothing to add.

### Error analysis — the most interesting part

I pulled the test review the model was *most confident* was negative among reviews
humans had labelled positive. The model gave it a 0.198 probability of being positive.
The review:

> "I have never seen such a movie before... I never thought such horrible acting
> existed it was all just too funny... I have never seen such a stupid movie in my
> life **which is why I think it's worth watching**. I give this movie 10 out of 10 for
> being the most pathetic movie ever created..."

This is a genuine 10/10 recommendation composed almost entirely of negative words. It
is the "so bad it's good" review, and it is not a labelling error — the label is right
and the model is wrong, for a reason no amount of extra training data fixes. Bag-of-words
representations cannot represent irony, because the sentiment is carried by the
relationship between the words and the rating, not by the words.

That is my reading now. The notebook reached the opposite conclusion at the time —
see **Known issues**.

Cross-checking with **VADER**, a lexicon and rule-based sentiment analyser that knows
nothing about this training set, gave an independent read on the same text — and it
disagreed with my classifier. VADER scored the review `compound: 0.9745`, strongly
positive, siding with the human label. The two methods do not fail alike: the
supervised model reads the review as negative, the lexicon reads it as positive.
VADER is not decoding the irony so much as counting the review's many positive tokens
("laughing", "funny", "worth watching"), but the disagreement is the useful result —
it corroborates that the label is defensible and that my model, not the annotator, is
what got this review wrong.

### Explainability

`eli5` was used to inspect both models: `show_weights` for the globally
most positive- and negative-weighted terms, and `show_prediction` to attribute the
sarcastic review's score to the individual tokens that drove it. Being able to point at
*which words* produced a decision is what makes a text classifier auditable.

## What I learned

- **Preserve the raw text.** Every interesting question during error analysis needed
  the original review, not the lemmatised token soup.
- **Bigrams are not optional for sentiment.** Negation lives between words.
- **Fit the vectorizer on train only.** Vocabulary and IDF weights are learned
  parameters and leak like any other.
- **A well-regularised linear model is a serious baseline.** It tied XGBoost here.
- **Inspect the confident mistakes.** Sorting by probability and reading the extremes
  taught me more about the model than the AUC did.
- **Some errors are irreducible under your representation.** Sarcasm is not a data
  problem or a tuning problem; it is a limitation of bag-of-words, and recognising that
  is the difference between iterating usefully and iterating blindly.
- **Cross-check with an independent method.** VADER cost nothing and corroborated the
  diagnosis.

## Known issues

- `SentimentIntensityAnalyzer` is used without an explicit
  `from nltk.sentiment.vader import SentimentIntensityAnalyzer` in the notebook; the
  cell only works if the name is already bound in the session.
- **The notebook's written verdict is the opposite of this README's.** The
  error-analysis cell concludes "It is a bad review" and blames "a potential labeling
  issue"; the VADER cell then contradicts that same cell in the other direction. I now
  think the label is correct and the model is wrong, but the notebook prose was never
  reconciled and is left as written.
- `show_prediction` is only ever applied to the XGBoost model, not to the logistic
  regression, even though the surrounding discussion covers both.
- The error-analysis cell reuses the variable `y_pred_prob`, which at that point holds
  the logistic regression's predictions even though the surrounding narrative discusses
  both models.
- `eli5`'s XGBoost explainer warns that it is only verified against xgboost < 2.0.0,
  so the per-token attributions for the boosted model should be treated as indicative.
- Models are compared on AUC alone; a precision/recall breakdown at a chosen operating
  threshold would make the comparison more actionable.
