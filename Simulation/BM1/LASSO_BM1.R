# This code obtains the "LASSO" classification result on Model BM1
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

LASSO_BM1 <- foreach(j = 1:R,.combine = rbind,
                   .packages = c("pamr","tensr","ramify","stats",'rTensor','expm',"TULIP","glmnet")) %dopar% {
                     # Fix the seed for each iteration in Parallel computing
                     set.seed(j*R+1)
                     # Generating train tensor normal random variable
                     X = matrix(0,n,nvars)
                     for(i in 1:n){
                       Z = array(rnorm(nvars),dimen)
                       X[i,] = array((mu[[Y[i]]] + atrans(Z,dsigma)),nvars)
                     }
                     # Cross-validation for the model
                     cv <- cv.glmnet(x = X,y = Y, family = 'binomial', nfolds = 5)
                     
                     # Fit the model with optimal lambda value 
                     fit <- glmnet(x = X,y = Y,lambda = cv$lambda.min,family = 'binomial',alpha = 1)
                     
                     # Generating test tensor normal random variable
                     X1 = matrix(0,n,nvars)
                     for(i in 1:n){
                       Z = array(rnorm(nvars),dimen)
                       X1[i,] =array((mu[[Y[i]]] + atrans(Z,dsigma)),nvars)
                     }
                     # Estimate class label on test set by using trained model
                     prediction <- predict(fit,newx = X1,type = 'response')
                     Yhat <- ifelse(prediction>0.5,2,1)
                     ret <- list(er = sum(Yhat != Y)/n*100,lambda = cv$lambda.min)
                     ret                          
                   }
LASSO_BM1 <- as.data.frame(LASSO_BM1)
error <- as.numeric(LASSO_BM1$er)
print("BM1 LASSO")
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)
stopCluster(cl)

