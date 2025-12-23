# This code obtains the "TDNSC" classification result on Model BM1
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

# Model Setting for BM1
d11 = c(1,2,11,12)
d12 = c(1,2)
d21 = c(1,2,11,12)
d22 = c(11,12)

dimen = c(64,64) # dimension of tensor
nvars = prod(dimen) # Number of variables
Y = c(rep(1,75),rep(2,75)) # Class label(Y)

n1 = 75;n2= 75
n = n1+n2
prob1 = n1/n
prob2 = n2/n
K=2 # Number of class

# Setting the each class centroid
v1 = array(0,dimen)
v2 = array(0,dimen)
v1[d11,d12] = -0.1
v1[d21,d22] = 1
v2[d11,d12] = 0.1
v2[d21,d22] = -1
V = array(list(),length(dimen))
V[[1]] = v1
V[[2]] = v2

# set-up of covariance matrix
sigma=array(list(), length(dimen))
sigma[[1]] = diag(dimen[1])
sigma[[2]] = diag(dimen[2])

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

# Number of replicates. Please set it to 100 to reproduce the simulation results for BM1.
R <- 100
ncores=strtoi(Sys.getenv("SLURM_NTASKS"))
cl = makeCluster(ncores,type= "SOCK")
registerDoParallel(cl)

TDNSC_BM1 <- foreach(j = 1:R,.combine = rbind,
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

TDNSC_BM1 <- as.data.frame(TDNSC_BM1)
error <- as.numeric(TDNSC_BM1$er)
print("BM1 TDNSC")
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)
stopCluster(cl)