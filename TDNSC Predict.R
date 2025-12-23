#### This function is estimating class label by using estimated TDNSC sample statistics
# model: The result of a call to TDNSC.train()
# X: Array of n list predictor data for TDNSC classification, which each tensor dimension is p1*p2*...*pM
library(rTensor)
library(tensr)
library(expm)
library(ramify)

source("~/Basic Setting.R")
source("~/TDNSC Train.R")

predict.TDNSC <- function(model,X){
  # Bring the sample statistic from TDNSC.train()
  prob <- model$prob
  mu <- model$hatmu
  invSigma <- model$invSigma
  predU <- model$predU
  n = length(X)
  lambda <- model$lambda
  nlambda <- length(lambda)
  K = length(prob)
  
  # Decorrelation for new tensor dataset X.
  U = array(list(),n)
  for(i in 1:n){
    U[[i]] = atrans(X[[i]]-mu,invSigma)
  }
  
  # Estimate the new tensor dataset X with sample statistic from TDNSC.train()
  Yhat = matrix(0,n,nlambda)
  for(i in 1:nlambda){
    for(j in 1:n){
      a = array(NA,K)
      for(k in 1:K){
        a[k] = sum((U[[j]] - predU[[i]][[k]])^2) - 2 * log(prob[[k]])
      }
      Yhat[j,i] = which.min(a)
    }
  }
  Yhat
}
