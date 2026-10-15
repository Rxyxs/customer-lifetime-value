**English** | [Español](README.es.md)

# Customer lifetime value: predicting the next six months of every customer

![CI](https://github.com/Rxyxs/customer-lifetime-value/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Two years of real transactions (December 2009 to December 2011) from a UK online wholesaler of
gift items: 1.07 million invoice lines, 5,878 identified customers. The top 10% of customers
bring 64% of the revenue, so the useful question is **who will be worth the most next**, and
how much the business can expect from its current base.

I fit the classic buy-till-you-die models, BG/NBD for purchase counts and Gamma-Gamma for spend,
implemented from the likelihoods with scipy, and test them the honest way: calibrate on
everything before 10 June 2011, predict the following 26 weeks, and compare with what each
customer actually did.

## Results

Holdout: 10 June to 9 December 2011, 4,944 customers.

| Model | Total purchases, error | MAE per customer | Spearman | Revenue from top 10% ranked | Total revenue, error |
|---|---|---|---|---|---|
| Naive: past rate x 26 weeks | +68% | 1.67 | 0.56 | 60% | +39% |
| LightGBM, fixed-window RFM features | +24% | 1.26 | 0.53 | 62% | +15% |
| **BG/NBD x Gamma-Gamma** | **-2%** | **1.04** | **0.62** | **64%** | -13% |

![Conditional expectations](docs/img/conditional.png)

**Three numbers per customer are enough.** BG/NBD only sees how many times a customer bought, when
they last bought and how long they have been a customer. It predicts the total number of
purchases within 2%, and tracks actual behaviour at every frequency level. The naive rate assumes
nobody ever leaves and overshoots by two thirds.

**The machine-learning baseline needed a fix before it was a fair comparison.** It has to be
trained on an earlier period with known outcomes (features up to June 2010, target the following
26 weeks, the same season). My first version used lifetime totals as features and overshot the
holdout by 151%: in June 2010 the data was six months old, in June 2011 eighteen, so lifetime
counts were far outside anything the model had seen. Rebuilding the features over fixed 13- and
26-week lookback windows brought the error to 24%. It still ranks customers slightly worse than
BG/NBD, with far more machinery.

**Revenue is underestimated by 13%.** The holdout contains the Christmas peak and the models are
stationary; spend per order in the second half of the year was higher than in the calibration
period.

![Revenue capture](docs/img/capture.png)

For targeting, ranking matters more than totals: the 10% of customers ranked highest by
BG/NBD x Gamma-Gamma brought 64% of the next six months' revenue, against 62% for LightGBM and
60% for the naive rate.

### Two things the models get wrong, and why

![P(alive)](docs/img/p_alive.png)

**P(alive) is right in order but wrong at the top.** Customers the model gives a 50-75% chance of
still being active bought again 38% of the time; 75-95%, 56%. But the top band (95-100%) bought
again only 53% of the time, because it contains 1,607 one-time buyers: in BG/NBD a customer can
only drop out after a repeat purchase, so a one-time buyer is "alive" by construction. Only 26% of
them came back, against 85% of the repeat buyers in the same band. For re-engagement, one-time
buyers need their own rule.

**Spend and frequency are nearly independent** (correlation 0.13 among repeat buyers), which is the
assumption Gamma-Gamma needs; it holds well enough here.

## Cohorts

![Cohorts](docs/img/cohorts.png)

Customers acquired from September to December 2010 bought again in months 2-6 at half the rate of
other cohorts (10.7% against 21.2%). With one year of data this cannot separate "holiday
customers are lower quality" from the calendar: their months 2-6 fall in the January-May low
season. The December 2009 row is inflated because it includes every customer who already existed
when the data starts, and the last observed month (December 2011) is only nine days long.

## Method

```
online_retail_II.csv ──► identified customers, no cancellations, real products only,
                          invoices collapsed to purchase days (data.py)
        │
        ├── summary at cutoff: x (repeat purchase days), t_x, T, average spend; holdout counts and revenue
        ├── BG/NBD and Gamma-Gamma by maximum likelihood (btyd.py, scipy)
        ├── baselines: naive rate; LightGBM on fixed-window RFM, trained a year earlier
        └── holdout comparison, P(alive) check, cohort retention (evaluate.py)
```

The implementation is tested against simulations of the model's own story: parameters are
recovered from simulated customers, and the conditional expectation for the holdout matches
simulated future purchases within 5% overall and 10% for frequent recent buyers.

## Data

Online Retail II, Daqing Chen, UCI Machine Learning Repository (2019). Kaggle mirror
[mashlyn/online-retail-ii-uci](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci),
license CC0. Downloaded October 2026; not redistributed here.

## Reproduce

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m clv.evaluate      # downloads with ~/.kaggle/kaggle.json on first run
python -m clv.figures
pytest -q tests
```

## Limitations

- The customers are mostly businesses buying wholesale, which behave differently from consumers.
- No seasonality in the models; a covariate or a seasonal adjustment would address the revenue gap.
- One calibration/holdout split. A second split a few months earlier would show how stable the
  ranking is.

## License

Code: MIT. Data: CC0 (Kaggle mirror of the UCI dataset), not redistributed.
