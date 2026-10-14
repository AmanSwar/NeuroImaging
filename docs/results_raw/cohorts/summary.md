# Cohort experiment summary

Seed 42 (repeat *k* uses seed+*k*) · 95 subjects · cohorts: ds003568, ds007181 · 3 repeat(s). Mean ± sd over repeats. Labels: 0 = control, 1 = patient.

> **How to read this.** Test sets contain only tens of subjects, so single numbers are noisy; chance balanced accuracy is 0.50. "patient" means a different disorder in each cohort, so cross-cohort transfer (LODO) is a hard, honest test.

**Cohort-identity probe:** a linear model on the 1-D spectrum alone identifies the cohort with 100.0% accuracy (majority baseline 50.5%). If this is near 100%, the model *could* use cohort identity as a shortcut; it only helps if label balance differs between cohorts.


## Pooled model — within-cohort results

| split | n | head | accuracy | balanced acc | AUC | sensitivity | specificity |
|---|---|---|---|---|---|---|---|
| val:ds003568 | 7 | **CAFN fused** | 0.619 ± 0.067 | 0.556 ± 0.079 | 0.750 ± 0.245 | 1.000 ± 0.000 | 0.111 ± 0.157 |
| val:ds003568 | 7 | MRI-only head | 0.524 ± 0.135 | 0.556 ± 0.079 | 0.333 ± 0.204 | 0.333 ± 0.471 | 0.778 ± 0.314 |
| val:ds003568 | 7 | spectrum-only head | 0.524 ± 0.067 | 0.500 ± 0.000 | 0.750 ± 0.204 | 0.667 ± 0.471 | 0.333 ± 0.471 |
| val:ds003568 | 7 | spectrum logistic-reg baseline | 0.571 ± 0.000 | 0.500 ± 0.000 | 0.361 ± 0.079 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| test:ds003568 | 7 | **CAFN fused** | 0.524 ± 0.067 | 0.458 ± 0.059 | 0.611 ± 0.142 | 0.917 ± 0.118 | 0.000 ± 0.000 |
| test:ds003568 | 7 | MRI-only head | 0.476 ± 0.067 | 0.500 ± 0.000 | 0.444 ± 0.039 | 0.333 ± 0.471 | 0.667 ± 0.471 |
| test:ds003568 | 7 | spectrum-only head | 0.524 ± 0.067 | 0.500 ± 0.000 | 0.444 ± 0.039 | 0.667 ± 0.471 | 0.333 ± 0.471 |
| test:ds003568 | 7 | spectrum logistic-reg baseline | 0.571 ± 0.000 | 0.500 ± 0.000 | 0.611 ± 0.283 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| val:ds007181 | 7 | **CAFN fused** | 0.524 ± 0.067 | 0.556 ± 0.052 | 0.750 ± 0.068 | 0.778 ± 0.314 | 0.333 ± 0.312 |
| val:ds007181 | 7 | MRI-only head | 0.571 ± 0.000 | 0.514 ± 0.020 | 0.667 ± 0.204 | 0.111 ± 0.157 | 0.917 ± 0.118 |
| val:ds007181 | 7 | spectrum-only head | 0.524 ± 0.067 | 0.542 ± 0.059 | 0.694 ± 0.104 | 0.667 ± 0.471 | 0.417 ± 0.425 |
| val:ds007181 | 7 | spectrum logistic-reg baseline | 0.571 ± 0.000 | 0.500 ± 0.000 | 0.472 ± 0.104 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| test:ds007181 | 7 | **CAFN fused** | 0.524 ± 0.067 | 0.542 ± 0.059 | 0.528 ± 0.142 | 0.667 ± 0.471 | 0.417 ± 0.425 |
| test:ds007181 | 7 | MRI-only head | 0.619 ± 0.067 | 0.556 ± 0.079 | 0.694 ± 0.079 | 0.111 ± 0.157 | 1.000 ± 0.000 |
| test:ds007181 | 7 | spectrum-only head | 0.524 ± 0.067 | 0.542 ± 0.059 | 0.528 ± 0.104 | 0.667 ± 0.471 | 0.417 ± 0.425 |
| test:ds007181 | 7 | spectrum logistic-reg baseline | 0.571 ± 0.000 | 0.500 ± 0.000 | 0.528 ± 0.079 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| val:ALL | 14 | **CAFN fused** | 0.571 ± 0.058 | 0.571 ± 0.058 | 0.741 ± 0.118 | 0.905 ± 0.135 | 0.238 ± 0.243 |
| val:ALL | 14 | MRI-only head | 0.548 ± 0.067 | 0.548 ± 0.067 | 0.517 ± 0.107 | 0.238 ± 0.337 | 0.857 ± 0.202 |
| val:ALL | 14 | spectrum-only head | 0.524 ± 0.034 | 0.524 ± 0.034 | 0.721 ± 0.107 | 0.667 ± 0.471 | 0.381 ± 0.442 |
| val:ALL | 14 | spectrum logistic-reg baseline | 0.571 ± 0.000 | 0.571 ± 0.000 | 0.531 ± 0.017 | 0.571 ± 0.000 | 0.571 ± 0.000 |
| test:ALL | 14 | **CAFN fused** | 0.524 ± 0.034 | 0.524 ± 0.034 | 0.612 ± 0.073 | 0.810 ± 0.269 | 0.238 ± 0.243 |
| test:ALL | 14 | MRI-only head | 0.548 ± 0.067 | 0.548 ± 0.067 | 0.626 ± 0.010 | 0.238 ± 0.337 | 0.857 ± 0.202 |
| test:ALL | 14 | spectrum-only head | 0.524 ± 0.034 | 0.524 ± 0.034 | 0.531 ± 0.073 | 0.667 ± 0.471 | 0.381 ± 0.442 |
| test:ALL | 14 | spectrum logistic-reg baseline | 0.571 ± 0.000 | 0.571 ± 0.000 | 0.605 ± 0.077 | 0.571 ± 0.000 | 0.571 ± 0.000 |

## Leave-one-cohort-out — train on the other cohort(s), test on **ds003568**

| split | n | head | accuracy | balanced acc | AUC | sensitivity | specificity |
|---|---|---|---|---|---|---|---|
| test:ds003568 | 47 | **CAFN fused** | 0.482 ± 0.072 | 0.458 ± 0.032 | 0.571 ± 0.068 | 0.563 ± 0.400 | 0.352 ± 0.459 |
| test:ds003568 | 47 | MRI-only head | 0.518 ± 0.096 | 0.507 ± 0.038 | 0.398 ± 0.099 | 0.552 ± 0.399 | 0.463 ± 0.412 |
| test:ds003568 | 47 | spectrum-only head | 0.574 ± 0.046 | 0.487 ± 0.014 | 0.502 ± 0.080 | 0.862 ± 0.149 | 0.111 ± 0.120 |
| test:ds003568 | 47 | spectrum logistic-reg baseline | 0.539 ± 0.110 | 0.500 ± 0.000 | 0.614 ± 0.030 | 0.667 ± 0.471 | 0.333 ± 0.471 |

## Leave-one-cohort-out — train on the other cohort(s), test on **ds007181**

| split | n | head | accuracy | balanced acc | AUC | sensitivity | specificity |
|---|---|---|---|---|---|---|---|
| test:ds007181 | 48 | **CAFN fused** | 0.451 ± 0.020 | 0.486 ± 0.040 | 0.439 ± 0.024 | 0.762 ± 0.272 | 0.210 ± 0.201 |
| test:ds007181 | 48 | MRI-only head | 0.500 ± 0.045 | 0.490 ± 0.028 | 0.446 ± 0.027 | 0.413 ± 0.427 | 0.568 ± 0.399 |
| test:ds007181 | 48 | spectrum-only head | 0.493 ± 0.049 | 0.511 ± 0.008 | 0.479 ± 0.094 | 0.651 ± 0.461 | 0.370 ± 0.445 |
| test:ds007181 | 48 | spectrum logistic-reg baseline | 0.521 ± 0.059 | 0.500 ± 0.000 | 0.571 ± 0.058 | 0.333 ± 0.471 | 0.667 ± 0.471 |

## Runs

| repeat | experiment | train cohorts | epochs run | batch size | minutes |
|---|---|---|---|---|---|
| 0 | pooled | ds003568, ds007181 | 33 | 8 | 1.44 |
| 0 | lodo_ds003568 | ds007181 | 41 | 8 | 1.2 |
| 0 | lodo_ds007181 | ds003568 | 52 | 8 | 1.57 |
| 1 | pooled | ds003568, ds007181 | 27 | 8 | 1.62 |
| 1 | lodo_ds003568 | ds007181 | 28 | 8 | 0.87 |
| 1 | lodo_ds007181 | ds003568 | 33 | 8 | 1.0 |
| 2 | pooled | ds003568, ds007181 | 29 | 8 | 1.77 |
| 2 | lodo_ds003568 | ds007181 | 53 | 8 | 1.64 |
| 2 | lodo_ds007181 | ds003568 | 27 | 8 | 0.82 |
