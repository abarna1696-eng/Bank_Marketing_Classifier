[Bank_Marketing_Classifier.zip](https://github.com/user-attachments/files/31342804/Bank_Marketing_Classifier.zip)
[bank-additional-full.csv](https://github.com/user-attachments/files/31342796/bank-additional-full.csv)
<img width="1408" height="486" alt="16_temporal_drift" src="https://github.com/user-attachments/assets/d85506d7-ef15-4dc5-8684-5aeb1710b652" />
<img width="970" height="640" alt="15_permutation_importance" src="https://github.com/user-attachments/assets/d1bd57fb-c6ea-45dc-ac21-1e78012e73ea" />
<img width="970" height="641" alt="14_tree_importances" src="https://github.com/user-attachments/assets/98cad178-720a-452b-a14c-19fc2f174797" />
<img width="2182" height="971" alt="13_decision_tree" src="https://github.com/user-attachments/assets/4d13832c-1914-4519-8885-81bc96ef5717" />
<img width="1080" height="970" alt="12_logistic_coefficients" src="https://github.com/user-attachments/assets/6c056413-61ba-44ae-a4ae-405a1f2327e7" />
<img width="1520" height="540" alt="11_cumulative_gains" src="https://github.com/user-attachments/assets/08458684-50d5-4a41-a8c3-732fe0fb1479" />
<img width="1138" height="540" alt="10_confusion_matrices" src="https://github.com/user-attachments/assets/278f0f75-bc1c-4a07-aca7-c2bb8a590437" />
<img width="1520" height="594" alt="09_roc_pr_curves" src="https://github.com/user-attachments/assets/b5e8515d-0793-4ccc-90e8-03388a878e81" />
<img width="1522" height="495" alt="08_model_comparison" src="https://github.com/user-attachments/assets/38b1531c-0754-4517-87cf-0d7bf2c5cf08" />
<img width="971" height="586" alt="07_job_confidence_intervals" src="https://github.com/user-attachments/assets/bbb62b0e-b343-4fde-b807-058f173f5eb0" />
<img width="870" height="750" alt="06_correlation_heatmap" src="https://github.com/user-attachments/assets/d7bbb9a5-edde-4c3a-bd03-5b787d314a18" />
<img width="1520" height="519" alt="05_timing_and_history" src="https://github.com/user-attachments/assets/5d462355-1f4e-47a6-bfa5-96e3d585460c" />
<img width="1522" height="1726" alt="04_categorical_conversion" src="https://github.com/user-attachments/assets/d9b1bd54-7c93-420b-a874-658dec21975c" />
<img width="1522" height="864" alt="03_numeric_by_outcome" src="https://github.com/user-attachments/assets/8e9e31c6-a526-401a-a0ee-6663137f9c96" />
<img width="1519" height="1081" alt="02_numeric_distributions" src="https://github.com/user-attachments/assets/b7c66467-5b94-4c41-9ac5-b111704f8b96" />
<img width="1190" height="433" alt="01_class_balance" src="https://github.com/user-attachments/assets/7a3fbcb8-8b59-4f03-be4c-aff69e5d978f" />
# Bank Marketing Campaign — Comparing Classifiers

Comparing k-Nearest Neighbors, Logistic Regression, Decision Trees and Support Vector Machines on
41,188 telemarketing calls made by a Portuguese bank, to decide **who the bank should call first**.

**Jupyter Notebook:** [notebooks/comparing_classifiers.ipynb](notebooks/comparing_classifiers.ipynb)

---

## Summary of Findings

**The bank can reach roughly two-thirds of its term-deposit subscribers while making 80% fewer calls.**

Only 1 call in 9 currently ends in a subscription. Sorting the same clients by model score and
working the list from the top down changes the economics of the campaign:

| If we call... | We reach | Conversion rate | vs. calling at random |
|---|---|---|---|
| the top **5%** | 26% of all subscribers | **58%** | **5.2x** |
| the top **10%** | 44% of all subscribers | **50%** | **4.4x** |
| the top **20%** | **64% of all subscribers** | **36%** | **3.2x** |
| the top **50%** | 82% of all subscribers | 19% | 1.6x |
| everyone (today) | 100% | 11% | — |

![Cumulative gains](images/11_cumulative_gains.png)

### Five actionable findings

1. **The bank calls hardest in its worst month.** May takes a third of all calls (13,769) and
   converts at **6.4%** — the lowest of any month. March, September, October and December convert at
   **40–50%** on barely 2,000 calls between them. *Worth a controlled test in a strong month; the
   low volumes there mean those rates would not survive being scaled up unchanged.*
2. **A prior relationship is the strongest asset available, and it is underused.** Clients contacted
   in an earlier campaign subscribe at **63.8%** vs. **9.3%** for everyone else; those who accepted
   an earlier offer accept again **65%** of the time. Only 4% of calls went to this group.
3. **Repeat calling backfires.** Each additional contact within a campaign *lowers* the odds of
   subscribing. One client was called 56 times. A cap of about three contacts is supported by the data.
4. **Mortgage and personal-loan status are worthless as targeting criteria** (Cramér's V = 0.012 and
   0.005 — statistically indistinguishable from noise), while **mobile contacts convert at 14.7% vs.
   5.2% for landlines**, and **students (31%) and retirees (25%)** convert two to three times better
   than average.
5. **Timing beats targeting — which is also the biggest risk.** The strongest predictors are interest
   rates and employment levels, not client attributes. The model is substantially reading the economy
   rather than the customer, and it must be retrained quarterly (see *Limitations* below).

---

## Business Understanding

A Portuguese retail bank sold term deposits by telephone across 17 campaigns between May 2008 and
November 2010. Outbound calling is expensive and yields roughly 11%, so agent time — not leads — is
the binding constraint.

> **The business question:** given what the bank knows about a client *before* the phone rings, how
> likely is that client to subscribe, and can that likelihood be used to decide who to call first?

The deliverable the business needs is a **ranked call list**, not a yes/no verdict. That framing
drives every modelling decision in the notebook, above all the choice of evaluation metric.

---

## Data Understanding

**Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(`bank-additional-full.csv`: 41,188 records, 20 inputs, 1 target)

- **Client:** `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
- **Current campaign:** `contact`, `month`, `day_of_week`, `duration`, `campaign`
- **Previous campaign:** `pdays`, `previous`, `poutcome`
- **Macroeconomic:** `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
- **Target:** `y` — did the client subscribe to a term deposit?

The target is heavily imbalanced: **11.3% positive**. A model that predicts "no" for everyone is
88.7% accurate and completely useless, which is why accuracy is never used to select a model here.

![Class balance](images/01_class_balance.png)
![Conversion by category](images/04_categorical_conversion.png)

---

## Data Preparation

| Step | Decision | Why |
|---|---|---|
| Duplicates | Removed 12 exact duplicate rows | — |
| **`duration`** | **Dropped entirely** | Not known before a call is placed, so a model using it cannot answer "who should we call?" See note below. |
| `pdays` | Split into `previously_contacted` (flag) + `days_since_prior` | `999` is a *never contacted* sentinel in 96% of rows, not a day count |
| `unknown` values | Kept as an explicit category | Informative (e.g. `default=unknown` converts at half the base rate), and 21% of `default` is too much to impute |
| Encoding | One-hot (categorical) + standardisation (numeric), inside a `Pipeline` | Keeps cross-validation honest — transformers refit per fold |
| Split | Stratified 75/25 | Preserves the 11.3% positive rate in both halves |

### A note on `duration` — why these results differ from the usual ones

Most published analyses of this dataset report ROC-AUC around **0.93–0.94**. Those models include
`duration` (call length in seconds), and the UCI documentation is explicit that it
*"should be discarded if the intention is to have a realistic predictive model"* — you only know a
call lasted 512 seconds *after* making it, and clients say yes and *therefore* stay on the phone.

Including it here would have raised ROC-AUC from 0.79 to 0.93 (measured in the notebook, §4.2). It
was dropped anyway. **Every number in this repository comes from a model that could actually be used
to plan a call list.**

---

## Methodology

- **Models:** k-Nearest Neighbors, Logistic Regression, Decision Tree, Support Vector Machine
- **Tuning:** `GridSearchCV` over all four, **720 model fits**, 5-fold stratified cross-validation
- **Selection metric:** **ROC-AUC** — the deliverable is a ranking, the right threshold depends on
  weekly agent capacity rather than on the model, and AUC is insensitive to class imbalance
- **Also reported:** PR-AUC, recall, precision, F1, top-decile lift
- **Threshold:** tuned on out-of-fold training predictions, never on the test set

---

## Results

### Tuned models on the held-out test set

| Model | CV ROC-AUC | **Test ROC-AUC** | Test PR-AUC | Accuracy | Recall | F1 | Tuning time |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.7889 | **0.8021** | 0.4461 | 0.832 | 0.643 | 0.464 | 1.3 min |
| Decision Tree | 0.7848 | 0.8003 | **0.4588** | 0.901 | 0.235 | 0.349 | 1.4 min |
| K-Nearest Neighbors | 0.7727 | 0.7871 | 0.4450 | 0.899 | 0.214 | 0.323 | **0.1 min** |
| Support Vector Machine | 0.7722 | 0.7852 | 0.3878 | 0.857 | 0.631 | 0.499 | 33.8 min |

*Baseline (predict "no" for everyone): 88.7% accuracy, 0.500 ROC-AUC, 0.0 recall.*

**Best model: Logistic Regression**, ROC-AUC **0.802**

![Model comparison](images/08_model_comparison.png)
![ROC and PR curves](images/09_roc_pr_curves.png)

### Best hyperparameters

| Model | Parameters |
|---|---|
| Logistic Regression | `C=0.1, penalty=l1, class_weight=balanced, solver=liblinear` |
| Decision Tree | `max_depth=7, min_samples_leaf=100, criterion=gini` |
| K-Nearest Neighbors | `n_neighbors=51, metric=manhattan, weights=uniform` |
| Support Vector Machine | `C=1, gamma=0.05, class_weight=balanced` |

### What the comparison actually showed

- **The simplest model won.** Logistic regression beat everything, and tuning moved it almost not at
  all (0.8023 → 0.8021) — the ceiling is set by the data, not the algorithm.
- **Pruning rescued the decision tree**: 0.619 → 0.800 test ROC-AUC. Unconstrained it hit 99.99%
  training AUC — a textbook overfit.
- **The SVM spent 33.8 minutes of fitting to finish last.** k-NN scored better in 6 seconds. On
  30,000 mostly-one-hot rows with largely linear structure, the expensive kernel buys nothing.
- **Ranked by PR-AUC the decision tree edges ahead** (0.459 vs. 0.446) — worth knowing if precision
  on the rare class ever matters more than ranking quality.

### What drives the prediction

![Logistic regression coefficients](images/12_logistic_coefficients.png)

Largest effects (odds ratios on standardised / one-hot features): `emp.var.rate` **x0.20**,
`month_mar` **x2.90**, `contact_telephone` **x0.57**, `month_may` **x0.51**,
`previously_contacted` **x1.40**, `job_retired` **x1.40**, `campaign` **x0.89**.

The L1 penalty eliminated **19 of 62** encoded features outright — including the `housing` and `loan`
indicators the chi-square tests had already flagged as noise.

> **Caveat:** `euribor3m`, `emp.var.rate` and `nr.employed` correlate at r = 0.90–0.97, so credit is
> split between them arbitrarily and individual coefficients flip sign. The *combined* business-cycle
> effect is reliable; those three coefficients individually are not.

---

## Limitations

**The most serious one:** every headline number above comes from a **random** train/test split, the
standard approach. But the rows are chronological and the strongest features are macroeconomic, so a
random split lets the model see one September-2009 client in training and predict another at test
time — sharing an identical interest rate.

Retrained on the earliest 75% of calls and asked to predict the most recent 25%, the model falls from
**0.802 to 0.567 ROC-AUC** — barely better than guessing. The positive rate itself shifts from 6.4%
to 25.8% between the two periods.

![Temporal drift](images/16_temporal_drift.png)

Also worth stating plainly: the data is **observational** (the bank chose who to call, so nothing
here is causal), `month` carries no year across a 2.5-year span, and no resampling technique or real
cost matrix was tried.

---

## Next Steps

1. **Deploy the ranking as an A/B test**, not a switchover — half the call centre on the ranked list,
   half on the current process, compared on subscriptions per agent-hour.
2. **Work the previous-campaign list first.** No model required; it is the 63.8% segment.
3. **Switch to `TimeSeriesSplit`** for all future validation, and **retrain quarterly** with drift
   monitoring on the macro indicators.
4. **Build a client-only model** without macro features and compare on a chronological split — a
   lower headline score may well be more durable.
5. **Replace F1 with expected value** in the threshold decision, using real deposit margin and
   agent-minute cost.
6. **Try gradient boosting** (XGBoost/LightGBM) — typically +0.02–0.05 ROC-AUC on tabular data of
   this shape, at a fraction of the SVM's cost.
7. **Collect better data:** the year attached to each contact, account tenure and balance band, and a
   call-outcome reason code.

---

## Repository Structure

```
Bank_Marketing_Classifier/
├── README.md                            # This file
├── notebooks/
│   └── comparing_classifiers.ipynb      # Full analysis
├── data/
│   └── bank-additional-full.csv         # UCI Bank Marketing dataset
└── images/                              # Figures generated by the notebook
```

## How to Run

```bash
git clone https://github.com/abarna1696-eng/Bank_Marketing_Classifier.git
cd Bank_Marketing_Classifier
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook notebooks/comparing_classifiers.ipynb
```

Run all cells. Expect **35–45 minutes** — the SVM grid search dominates the runtime.

## Author

Abarna Mathiyalagan

## Acknowledgments

- Moro, S., Cortez, P. & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank
  Telemarketing.* Decision Support Systems, 62:22–31.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence
