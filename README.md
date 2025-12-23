# De-correlated Nearest Shrunken Centroids for Tensor Data (TDNSC)

**Authors:** Shaokang Ren (Microsoft Corporation), Munwon Yang (Florida State University), Qing Mai (Florida State University)

This repository provides R code implementing the method described in _"De-correlated Nearest Shrunken Centroids for Tensor Data"_.

---

## Main Functions
The following R source files implement TDNSC:
- `Basic Setting.R`
- `TDNSC Train.R`
- `TDNSC Predict.R`
- `Cross validation of TDNSC.R`

### `TDNSC.train()`
Solves the classification problem and performs variable selection by fitting the Tensor De-correlated Nearest Shrunken Centroid (TDNSC) model.

**Arguments**
- `X`: Array/list of `n` predictor tensors, each with dimensions $\(\mathbb{R}^{p_1 \times \cdots \times p_M}\)$.
- `Y`: Class labels corresponding to each tensor in `X`.
- `lambda`: User-specified sequence of threshold parameters (optional; default `NULL`).
- `nlambda`: Number of lambda values to generate if `lambda` is not provided (default `20`).

**Value**
- `prob`: Class label proportions.
- `hatmu`: Estimated grand mean tensor for data `X`.
- `Sigma`: Estimated covariance matrices in the covariance structure.
- `invSigma`: Inverses of the estimated covariance matrices.
- `lambda`: Actual lambda sequence used (either user-specified or auto-generated; length determined by `nlambda`).
- `predU`: Array/list of estimated class mean tensors for each threshold parameter (`lambda`), after soft-thresholding.
- `Yhat`: Estimated class labels for each sample across all `lambda` values (rows = samples, columns = `lambda` values).
- `error`: Classification error comparing `Y` and `Yhat`.
- `nonzero`: Number of nonzero elements in each shrunken centroid (rows = `lambda` values, columns = class labels).

### `predict.TDNSC()`
Predicts categorical responses on new matrix/tensor data using a fitted TDNSC model.

**Arguments**
- `model`: Output of a call to `TDNSC.train()`.
- `X`: Array/list of predictor tensors with dimensions $\(\mathbb{R}^{p_1 \times \cdots \times p_M}\)$.

### `cv.TDNSC()`
Performs K-fold cross-validation for TDNSC and returns the best threshold value `lambda` from user-specified or automatically generated choices.

**Arguments**
- `X`: Array/list of predictor tensors with dimensions $\(\mathbb{R}^{p_1 \times \cdots \times p_M}\)$.
- `Y`: Class labels.
- `lambda`: Optional user-specified lambda sequence (default `NULL`). If `NULL`, a sequence is auto-generated and cross-validated.
- `nlambda`: Number of lambda values if `lambda` is not provided (default `20`).
- `folds`: A list with `nfold` components, each a vector of indices for samples in that fold (default `NULL`).
- `nfold`: Number of cross-validation folds (default `5`).

**Value**
- `folds`: Actual fold assignment used (user-specified or auto-generated).
- `lambda`: Actual lambda sequence used (user-specified or auto-generated).
- `Yhat`: Augmented estimated class labels by TDNSC on the test sets for each `lambda` value.
- `error`: Classification error between `Yhat` and `Y` for each `lambda` value.
- `lambda.min`: Lambda with minimum cross-validation error.

---

## Required Packages
```r
library(Tensr)
library(expm)
library(rTensor)
library(splitTools)
library(ramify)
```

Source the functions in the following order:
```r
source("TDNSC Train.R")
source("TDNSC Predict.R")
source("Cross validation of TDNSC.R")
source("Basic Setting.R")
```

---

## Simulation Example (Model TB1)
We generate 75 i.i.d. tensor observations per class (K = 3). Each tensor has dimensions `(30, 36, 30)`.

**Available scripts**
- `Bayes Error TB1.R`: Reproduces Bayes Error results for Model TB1.
- `CATCH TB1.R`: Reproduces CATCH results (implemented in R package **TULIP**).
- `LASSO TB1.R`: Reproduces LASSO results (\(\ell_1\) penalized logistic regression via **glmnet**).
- `MSDA TB1.R`: Reproduces MSDA results (implemented in **TULIP**).
- `NSC TB1.R`: Reproduces NSC results (implemented in **pamr**).
- `TDNSC TB1.R`: Reproduces TDNSC results.
- `CPGLM TB1.m`: Reproduces CP-GLM results (MATLAB® toolbox **TensorReg**).
- `Basic setting.R`: Utility functions for Simulation, Real Data Analysis, and TDNSC functions.

### Reproduce One TDNSC Replicate (Model TB1)
Load packages and source files:
```r
rm(list = ls())
source("~/TDNSC Train.R")
source("~/TDNSC Predict.R")
source("~/Basic setting.R")
source("~/Cross validation of TDNSC.R")
library(expm)
library(tensr)
library(rTensor)
```

**Model Setting**
Use `atrans()` from **tensr** to compute class tensor means given class centroids $\(\nu_k\)$ for $\(k = 1,\dots,K\)$:
```r
dimen <- c(30, 36, 30)
K <- 2
nk <- 75

V1 <- array(0, dim = dimen)
V2 <- array(0, dim = dimen)
for (i in ini) {
  for (j in inj) {
    for (s in ins) {
      V1[i, j, s] <- -0.8
      V2[i, j, s] <-  0.8
    }
  }
}
V <- array(list(), K)
V[[1]] <- V1
V[[2]] <- V2

sigma <- array(list(), length(dimen))
sigma[[1]] <- CS(0.5, dimen[1])
sigma[[2]] <- AR(0.5, dimen[2])
sigma[[3]] <- CS(0.5, dimen[3])
```

**Data Generation**
```r
# Generate training and test sets
data <- Simulation.Model(75, 2, c(30, 36, 30), V, sigma, seed = 1996)
X    <- data$x
Vax  <- data$vax
Y    <- data$Y
```

**K-fold Cross-Validation and Model Fitting**
```r
# Model fitting
mod <- TDNSC.train(X, Y, nlambda = 30)

# Cross-validation
modcv <- cv.TDNSC(X, Y, lambda = mod$lambda, nfold = 5)

# Fit the model with optimal threshold
mod1 <- TDNSC.train(X, Y, lambda = modcv$lambda.min)
```

**Classification Error Evaluation**
```r
Yhat <- predict.TDNSC(mod1, Vax)
er   <- sum(Yhat != Y) / n * 100
```

---

## Real Data Analyses

### GDS1083
- Tissue-specific gene expression (brain, heart, lung) from 9 individuals.
- Each observation: `4 x 1124` matrix, `n = 27`, classes `K = 3`.

Scripts:
- `CATCH for GDS1083.R`
- `LASSO for GDS1083.R`
- `MSDA for GDS1083.R`
- `NSC for GDS1083.R`
- `TDNSC for GDS1083.R`

### PD
- Each observation: `6 x 28 x 2` tensor, `n = 153`, `K = 2`.
- Labels: `y = 0` (male), `y = 1` (female).

Scripts:
- `CATCH PD.R`
- `LASSO PD.R`
- `MSDA PD.R`
- `NSC PD.R`
- `TDNSC PD.R`
- `CP GLM PD.m`

---

## Citation
If you use this code, please cite the TDNSC paper:
> Ren, S., Yang, M., Mai, Q. _De-correlated Nearest Shrunken Centroids for Tensor Data_.

