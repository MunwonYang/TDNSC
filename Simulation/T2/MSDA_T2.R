# This code obtains the "MSDA" classification result on Model T2
rm(list = ls())
source("~/Basic Setting.R")
# Recall the required packages
library(rTensor)
library(tensr)
library(TULIP)
library(expm)
library(glmnet)
library(pamr)
library(ramify)
library(foreach)
library(doParallel)

# Model Setting for T2
ini = c(1,2,11,12)
inj = c(1,11)
ins = c(1,11)
dimen = c(30,36,30)
nvars = prod(dimen)

K = 3 # Class label
nk = 75
n = nk*K
prob1 = nk/n
prob2 = nk/n
prob3 = nk/n
Y = c(rep(1,nk),rep(2,nk),rep(3,nk))

# Setting the each class centroid
V1 = array(0, dim = dimen)
V2 = array(0, dim = dimen)
V3 = array(0, dim = dimen)
for (i in ini) {
  for (j in inj) {
    for (s in ins) {
      V1[i,j,s] = 1
      V2[i,j,s] = 0.3
      V3[i,j,s] = -1.3
    }
  }
}

V <- array(list(),K)
V[[1]] <- V1
V[[2]] <- V2
V[[3]] <- V3

# set-up of covariance matrix
sigma=array(list(), length(dimen))
sigma[[1]] = CS(0.5,dimen[1])
sigma[[2]] = AR(0.5,dimen[2])
sigma[[3]] = CS(0.5,dimen[3])

# set-up for generating tensor normal random variable
dsigma = array(list(),length(dimen))
for (i in 1:length(dimen)){
  dsigma[[i]] = sqrtm(sigma[[i]])
}

# Calculate each class tensor mean
mu = array(list(),K)
for(i in 1:K){
  mu[[i]] = atrans(V[[i]],dsigma)
}

# Number of replicates. Please set it to 100 to reproduce the simulation results for T2.
R = 100
ncores=8
cl = makeCluster(ncores,type= "SOCK")
registerDoParallel(cl)

MSDA_T2 <- foreach(j = 1:R,.combine = rbind,
                   .packages = c("pamr","tensr","ramify","stats",'rTensor','expm',"TULIP","glmnet")) %dopar% {
                     # Fix the seed for each iteration in Parallel computing
                     set.seed(j*R+1)
                     # Generating train tensor normal random variable(3-way tensor), and store it each row
                     X = matrix(0,n,nvars)
                     for(i in 1:n){
                       Z = array(rnorm(nvars),dimen)
                       X[i,] = array((mu[[Y[i]]] + atrans(Z,dsigma)),nvars)
                     }
                     # Cross-validation for the model
                     cv <- cv.msda(x = X,y = Y, nfolds = 5, maxit = 1e+03)
                     
                     # Fit the model with optimal lambda value 
                     fit <- msda(x = X,y = Y,lambda = cv$lambda.min, maxit = 1e+03)
                     
                     # Generating test tensor normal random variable
                     X1 = matrix(0,n,nvars)
                     for(i in 1:n){
                       Z = array(rnorm(nvars),dimen)
                       X1[i,] =array((mu[[Y[i]]] + atrans(Z,dsigma)),nvars)
                     }
                     # Estimate class label on test set by using trained model
                     Yhat <- predict(fit,newx = X1)
                     ret <- list(er = sum(Yhat != Y)/n*100,lambda = cv$lambda.min)
                     ret                          
                   }
MSDA_T2 <- as.data.frame(MSDA_T2)
error <- as.numeric(MSDA_T2$er)
print("T2 MSDA")
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)
stopCluster(cl)