# This code obtains the "TDNSC" classification result on Model BM2
rm(list = ls())
# Recall the TDNSC functions and required packages
source("~/Basic Setting.R")
source("~/TDNSC Train.R")
source("~/TDNSC Predict.R")
source("~/Cross validation of TDNSC.R")
library(expm)
library(tensr)
library(foreach)
library(doParallel)

# Model Setting for BM2
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
v1[d11,d12] = 0.25
v1[d21,d22] = 0.4
v2[d11,d12] = -0.25
v2[d21,d22] = -0.4
V = array(list(),length(dimen))
V[[1]] = v1
V[[2]] = v2

# set-up of covariance matrix
sigma=array(list(), length(dimen))
sigma[[1]] = AR(0.8,dimen[1])
sigma[[2]] = CS(0.5,dimen[2])

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
ncores=8
cl = makeCluster(ncores,type= "SOCK")
registerDoParallel(cl)

TDNSC_BM2 <- foreach(j = 1:R,.combine = rbind,
                     .packages = c("tensr","ramify","stats",'rTensor','expm')) %dopar% {
                       # Recall the required packages
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

TDNSC_BM2 <- as.data.frame(TDNSC_BM2)
error <- as.numeric(TDNSC_BM2$er)
print("BM2 TDNSC")
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)
stopCluster(cl)

