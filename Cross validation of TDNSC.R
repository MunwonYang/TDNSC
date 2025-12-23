#### This function is to cross-validate the TDNSC produced by TDNSC.train()
# X: Array of n list of predictor data, which each dimension is p1*p2*...*pM
# Y: Class label corresponding with each X tensor data
# lambda: User-specified lambda sequence for cross-validation. If not specified, the algorithm will generate a sequence of lambda based on all data and cross-validate the sequence. Default is NULL.
# nlambda: Number of lambda values when the sequence of lambda for TDNSC is not assigned. Default is 20.
# folds: A list with nfold components, each component a vector of indices of the samples in that fold. Default is NULL
# nfold: Number of cross-validation folds. Default is 5.

library(rTensor)
library(tensr)
library(expm)
library(ramify)

source("~/Basic Setting.R")
source("~/TDNSC Train.R")
source("~/TDNSC Predict.R")

cv.TDNSC <- function(X,Y,lambda = NULL,nlambda = 20,folds = NULL, nfold = 5){
  n = length(X) # number of dataset
  dimen = dim(X[[1]]) # dimension of dataset
  nvars = prod(dimen) # number of variables
  check = 0
  for(i in 1:n){
    if(sum(dimen == dim(X[[i]])) == length(dimen)){
      check = check + 1
    }
  }
  if(check == n){
    if(!is.null(lambda)){
      nlambda = length(lambda)
    }
    time_dif = array(list(),nfold)
    if(is.null(folds)){
      # Assigning train data, test data, and folding index, when folds is NULL.
      D <- cv_fold(X,Y,nfold)
      train_x <- D$train
      test_x <- D$test
      y_train <- D$trainy
      lambda_store <- array(NA,c(nfold,nlambda))
      Yhat <- array(NA,c(n,nlambda))
      err <- rep(NA, nlambda)
      # K-fold cross-validation if lambda sequence is specified.
      if(!is.null(lambda)){
        # Apply K-fold cross-validation
        for(i in 1:nfold){
          cat("Fold",i,":")
          # TDNSC on train dataset
          fit <- TDNSC.train(train_x[[i]],y_train[[i]],lambda = lambda)
          # Estimate class label on test data through TDNSC on train dataset.
          prediction <- predict.TDNSC(fit,test_x[[i]])
          Yhat[D$fold[[i]],] <- prediction
          cat("\n")
        }
      }
      # K-fold cross-validation, if lambda sequence is NULL.
      else{
        # Apply K-fold cross-validation
        for(i in 1:nfold){
          cat("Fold",i,":")
          # TDNSC on train dataset
          fit <- TDNSC.train(train_x[[i]],y_train[[i]],nlambda = nlambda)
          lambda_store[i,] = fit$lambda
          # Estimate class label on test data through TDNSC on train dataset.
          prediction <- predict.TDNSC(fit,test_x[[i]])
          Yhat[D$fold[[i]],] <- prediction
          cat("\n")
        }
        # Obtain the lambda sequence from K different lambda sequence on K-fold cross-validation.
        lambda <- colMeans(lambda_store)
      }
      for(i in 1:nlambda){
        err[i] <- sum(Yhat[,i] != Y) / n * 100
      }
      # Obtain the optimal lambda value through K-fold cross-validation.
      lambda.min <- lambda[which.min(err)]
      obj <- list(lambda = lambda, lambda.min = lambda.min, error = err, folds = D$folds, Yhat = Yhat)
      class(obj) <- "Cross validation of TDNSC"
      obj
    }
    # K-fold cross-validation when folds are specified. 
    else{
      D1 <- fold.assigned(X,Y,folds = folds)
      train_x <- D1$train
      test_x <- D1$test
      y_train <- D1$trainy
      nfold <- length(folds)
      Yhat <- array(NA,c(n,nlambda))
      lambda_store <- array(NA,c(nfold,nlambda))
      err <- rep(NA, nlambda)
      # K-fold cross-validation if lambda sequence is specified.
      if(!is.null(lambda)){
        # Apply K-fold cross-validation
        for(i in 1:nfold){
          cat("Fold",i,":")
          # TDNSC on train dataset
          fit <- TDNSC.train(train_x[[i]],y_train[[i]],lambda = lambda)
          # Estimate class label on test dataset through TDNSC on train dataset
          prediction <- predict.TDNSC(fit,test_x[[i]])
          Yhat[D1$fold[[i]],] <- prediction
          cat("\n")
        }
      }
      # K-fold cross-validation if lambda sequence is NULL.
      else{
        # Apply K-fold cross-validation
        for(i in 1:nfold){
          cat("Fold",i,":")
          # TDNSC on train dataset
          fit <- TDNSC.train(train_x[[i]],y_train[[i]],nlambda = nlambda)
          lambda_store[i,] = fit$lambda
          # Estimate class label on test dataset through TDNSC on train dataset
          prediction <- predict.TDNSC(fit,test_x[[i]])
          Yhat[D1$fold[[i]],] <- prediction
          cat("\n")
        }
        # Obtain the lambda sequence from K different lambda sequence on K-fold cross-validation.
        lambda <- colMeans(lambda_store)
      }
      for(i in 1:nlambda){
        err[i] <- sum(Yhat[,i] != Y) / n * 100
      }
      # Obtain the optimal lambda value through K-fold cross-validation.
      lambda.min <- lambda[which.min(err)]
      obj <- list(lambda = lambda, lambda.min = lambda.min, error = err, folds = D1$folds, Yhat = Yhat)
      class(obj) <- "Cross validation of TDNSC when folds are assigned"
      obj
    }
  }
  else{
    stop("Data dimension doesn't match")
  }
}
