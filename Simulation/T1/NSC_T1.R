# This code obtains the "NSC" classification result on Model T1
rm(list = ls())
source("~/Basic Setting.R")
# Recall the required packages
library(rTensor)
library(tensr)
library(expm)
library(glmnet)
library(pamr)
library(ramify)
library(foreach)
library(doParallel)

# Model Setting for T1
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
      V1[i,j,s] = 0.2
      V2[i,j,s] = 0.8
      V3[i,j,s] = -1
    }
  }
}

V <- array(list(),K)
V[[1]] <- V1
V[[2]] <- V2
V[[3]] <- V3

# set-up of covariance matrix
sigma=array(list(), length(dimen))
sigma[[1]] = AR(0.2,dimen[1])
sigma[[2]] = CS(0.5,dimen[2])
sigma[[3]] = AR(0.8,dimen[3])

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

# Number of replicates. Please set it to 100 to reproduce the simulation results for T1.
R = 100
ncores=8
cl = makeCluster(ncores,type= "SOCK")
registerDoParallel(cl)

NSC_T1 <- foreach(j = 1:R,.combine = rbind,
                  .packages = c("pamr","tensr","ramify","stats",'rTensor','expm',"TULIP",'glmnet')) %dopar% {
                    # Fix the seed for each iteration in Parallel computing
                    set.seed(j*R+1)
                    # Generating train tensor normal random variable(3-way tensor) and store it in each row.
                    X = array(0,c(n,nvars))
                    for(i in 1:n){
                      Z = array(rnorm(nvars),dimen)
                      X[i,] = array(mu[[Y[i]]] + atrans(Z,dsigma),nvars)
                    }
                    mydata <- list(x = t(X),y=  factor(Y))
                    # Model fitting
                    mod <- pamr.train(mydata)
                    # Cross-validation for the model
                    modcv <- pamr.cv(mod,mydata)
                    lambda = modcv$threshold[which.min(modcv$error)]
                    
                    # Generating test tensor normal random variable and store it in each row.
                    X1 = array(0,c(n,nvars))
                    for(i in 1:n){
                      Z = array(rnorm(nvars),dimen)
                      X1[i,] =array(mu[[Y[i]]] + atrans(Z,dsigma),nvars)
                    }
                    # Estimate class label on test set by using trained model
                    Yhat <- pamr.predict(mod,t(X1),threshold = lambda)
                    ret <- list(er = sum(Yhat != Y)/n*100,lambda = lambda)
                    ret                          
                  }
NSC_T1 <- as.data.frame(NSC_T1)
error <- as.numeric(NSC_T1$er)
print("T1 NSC")
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)
stopCluster(cl)