# This code obtains the "TDNSC" classification result on Model T2
rm(list = ls())
# Recall the TDNSC functions and required packages
source("~/Basic Setting.R")
source("~/TDNSC Train.R")
source("~/TDNSC Predict.R")
source("~/Cross validation of TDNSC.R")
library(expm)
library(tensr)
library(rTensor)
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

TDNSC_T2 <- foreach(j=1:R,.combine = rbind,
                    .packages = c("tensr","ramify","stats",'rTensor','expm')) %dopar% {
                      # Fix the seed for each iteration in Parallel computing
                      set.seed(j*R+1)
                      # Generating train tensor normal random variable
                      X = array(list(),n)
                      for(i in 1:n){
                        Z = array(rnorm(nvars),dimen)
                        X[[i]] = mu[[Y[i]]] + atrans(Z,dsigma)
                      }
                      # Model fitting
                      mod <- TDNSC.train(X,Y,nlambda = 30)
                      
                      # Cross-validation for the model
                      modcv <- cv.TDNSC(X,Y,lambda = mod$lambda,nfold = 5)
                      lambda = modcv$lambda.min
                      
                      # Fit the model with optimal threshold value 
                      mod1 <- TDNSC.train(X,Y,lambda = modcv$lambda.min)
                      
                      # Generating test tensor normal random variable
                      X1 = array(list(),n)
                      for(i in 1:n){
                        Z = array(rnorm(nvars),dimen)
                        X1[[i]] = mu[[Y[i]]] + atrans(Z,dsigma)
                      }
                      # Estimate class label on test set by using trained model
                      Yhat = predict.TDNSC(mod1,X1)
                      ret <- list(er = sum(Yhat != Y)/n*100,lambda = lambda)
                      ret                          
                      }
TDNSC_T2 <- as.data.frame(TDNSC_T2)
error <- as.numeric(TDNSC_T2$er)
print("T2 TDNSC")
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)
stopCluster(cl)
