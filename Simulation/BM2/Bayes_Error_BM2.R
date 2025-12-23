# This code obtains the "Bayes Error" result on Model BM2
rm(list = ls())
source("~/Basic Setting.R")
# Recall the required packages
library(rTensor)
library(tensr)
library(expm)
library(ramify)
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
sigma <- array(list(),length(dimen))
sigma[[1]] = AR(0.8,dimen[1])
sigma[[2]] = CS(0.5,dimen[2])

# set-up for generating tensor normal random variable
dsigma = array(list(),length(dimen))
for (i in 1:length(dimen)){
  dsigma[[i]] = sqrtm(sigma[[i]])
}

# Calculate each class tensor mean 
mu = array(list(),length(dimen))
for(i in 1:length(dimen)){
  mu[[i]] = atrans(V[[i]],dsigma)
}

# Calculate the discriminative coefficient tensor
stand = array(list(),length(dimen))
for (i in 1:length(dimen)){
  stand[[i]] = solve(sigma[[i]])
}
B1 = atrans(mu[[1]]-mu[[1]],stand)
B2 = atrans(mu[[2]]-mu[[1]],stand)
B = list(B1,B2)

# Number of replicates. Please set it to 100 to reproduce the simulation results for BM2.
R <- 100
ncores=8
cl = makeCluster(ncores,type= "SOCK")
registerDoParallel(cl)

Bayes_error_BM2 <- foreach(j = 1:R,.combine = rbind,
                           .packages = c("tensr","ramify","stats")) %dopar% {
                             # Fix the seed for each iteration in Parallel computing
                             set.seed(j*R+1)
                             # Generating tensor normal random variable
                             X = array(list(),n)
                             for (i in 1:n){
                               Z = array(rnorm(nvars),dimen)
                               X[[i]] = mu[[Y[i]]] + dsigma[[1]]%*%Z%*%dsigma[[2]]
                             }
                             p = array(0,K)
                             for(i in 1:K){
                               p[i] = sum(Y==i) / n
                             }
                             
                             # Estimate the class label by Bayes Rule
                             Yhat = array(0,n)
                             for(i in 1:n){
                               val = array(0,K)
                               for(j in 1:K){
                                 val[j] = log(p[j]/p[1]) + sum(B[[j]]*(X[[i]] - 0.5*(mu[[j]]+mu[[1]])))
                               }
                               Yhat[i] = which.max(val)
                             }
                             ret <- list(er = sum(Yhat != Y)/n*100)
                             ret                         
                           }

Bayes_error_BM2 <- as.data.frame(Bayes_error_BM2)
error <- as.numeric(Bayes_error_BM2$er)
print("BM2 Bayes Error")
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)

stopCluster(cl)