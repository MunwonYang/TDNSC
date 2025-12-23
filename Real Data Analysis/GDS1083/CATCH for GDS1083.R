# This code obtains the "CATCH" classification result on GDS1083(Multi-categorical classification problem)
rm(list = ls())
# Recall the required packages
library(rTensor)
library(tensr)
library(TULIP)
library(msda)
library(expm)
library(glmnet)
library(pamr)
library(ramify)
library(foreach)
library(doParallel)
library(splitTools)

# Fix the seed for data preprocessing
set.seed(499)

# Recall the GDS1083 data
x <- scan("~\GDS1083.txt")
# Transform it into 108*1124 matrix
x= matrix(x,nrow=108)
# Each observation is 4*1124 matrix, so total sample size n=27
# Multi-categorical classification(Brain, heart and lung)
Y = rep(1:3,9)
K = 3 # Number of class

#screening(Downsizing the gene expression data into 4*20 matrix)
kstest = rep(0,ncol(x))
for (i in 1:ncol(x)) {
  kstest[i] = ks.test(x[,i],"pnorm")$p.value
}
obj = sort(kstest,index.return=TRUE,decreasing=TRUE)
index = obj$ix[1:20]
# Choose 20 genes that shows highest similarity with standard normal distribution in Kolmogorov-Smirnov test
x = x[,index]
dimen0 = c(4,length(index))

# Store the 4*20 downsized matrix
dat = array(list(),27)
a = c(t(x))

for (i in 1:27){
  dat[[i]] = array(a[(80*(i-1)+1):(80*i)],dim=c(dimen0))
}
Y=as.numeric(Y)
n=length(Y)
Xs = dat

# Number of replicates. Please set it to 100 to reproduce the real data analysis for GDS1083.
R = 100
ncores=8
cl = makeCluster(ncores)
registerDoParallel(cl)

GDS1083_CATCH <- foreach(j = 1:R,.combine = rbind,
                    .packages = c("tensr","ramify","stats",'rTensor','expm',"TULIP",'glmnet','splitTools')) %dopar% {
                      # Fix the seed for each iteration in Parallel computing
                      set.seed(j*65)
                      # Randomly split the train data(80%) and test data(20%)
                      part = partition(Y, c(0.8,0.2))
                      Xt = Xs[part$'1'] # Train data on predictor matrix(2-way tensor)
                      Xtr = Xs[part$'2'] # Train data on class label
                      Yt = Y[part$'1'] # Test data on predictor matrix(2-way tensor)
                      Ytr = Y[part$'2'] # Test data on class label
                      nsam = length(Ytr)
                      # Model fitting on train data
                      mod1 = catch(x = Xt, y = Yt)
                      # Cross-validation for the model
                      modcv = cv.catch(x = Xt, y = Yt, nfolds = 5, lambda = mod1$lambda)
                      ind = which(mod1$lambda == modcv$lambda.min)
                      # Fit the model with optimal lambda value 
                      mod2 <- catch(x = Xt,y = Yt,lambda = modcv$lambda.min)
                      # Estimate class label on test set by using trained model
                      res = predict.catch(mod2, Xtr)
                      ret <- list(er = sum(res-Ytr!=0)/nsam*100,lambda = modcv$lambda.min)
                      ret   
                    }
GDS1083_CATCH <- as.data.frame(GDS1083_CATCH)
error <- as.numeric(GDS1083_CATCH$er)
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)
stopCluster(cl)

