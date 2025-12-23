# This code obtains the "NSC" classification result on Model M2
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

# Model Setting for M2
d11 = c(1,2,11,12)
d12 = c(1,2)
d21 = c(1,2,11,12)
d22 = c(11,12)

dimen = c(64,64) # dimension of tensor
nvars = prod(dimen) # Number of variables
Y = c(rep(1,75),rep(2,75),rep(3,75)) # Class label(Y)

n1 = 75;n2= 75;n3 = 75
n = n1+n2+n3
prob1 = n1/n
prob2 = n2/n
prb3 = n3/n
K=3 # Number of class

# Setting the each class centroid
v1 = array(0,dimen)
v2 = array(0,dimen)
v3 = array(0,dimen)
v1[c(d11,d21),c(d12,d22)] = -0.9
v3[c(d11,d21),c(d12,d22)] = 0.9
V <- array(list(),K)
V[[1]] <- v1
V[[2]] <- v2
V[[3]] <- v3

# set-up of covariance matrix
sigma= array(list(),length(dimen))
sigma[[1]] = CS(0.5,dimen[1])
sigma[[2]] = CS(0.7,dimen[2])

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

# Number of replicates. Please set it to 100 to reproduce the simulation results for M2.
R <- 100
ncores=8
cl = makeCluster(ncores,type= "SOCK")
registerDoParallel(cl)

NSC_M2 <- foreach(j = 1:R,.combine = rbind,
                  .packages = c("pamr","tensr","ramify","stats",'rTensor','expm',"TULIP",'glmnet')) %dopar% {
                    # Fix the seed for each iteration in Parallel computing
                    set.seed(j*R+1)
                    # Generating train tensor normal random variable
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
                    
                    # Generating test tensor normal random variable
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
NSC_M2 <- as.data.frame(NSC_M2)
error <- as.numeric(NSC_M2$er)
print("M2 NSC")
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)
stopCluster(cl)