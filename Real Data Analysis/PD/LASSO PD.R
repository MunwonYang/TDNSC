# This code obtains the "LASSO" classification result on GDS1083(Multi-categorical classification problem)
rm(list = ls())
source("~/Basic setting.R")
# Fix the seed for data preprocessing
set.seed(379)
library(rTensor)
library(mice)
library(tensr)
library(TULIP)
library(expm)
library(glmnet)
library(pamr)
library(ramify)
library(foreach)
library(doParallel)
library(splitTools)

# Recall the PD data
PD=read.csv('PDdata.csv')
# Indexing PD data
PD1=PD[,c(1,2,4,5,6,7,8,9,10,11)]
colnames(PD1)=c('mid','tooth','surf','PDD','CAL','age','gender','BMI','smoker','Hb')
attach(PD1)

midr=rep(0,311)
for (i in unique(mid)){
  if(sum(is.na(PDD[mid==i]))/length(PDD[mid==i])<0.3&sum(is.na(CAL[mid==i]))/length(PDD[mid==i])<0.3){
    midr[i]=1
  }
}
midd=(1:311)[midr==1]
PD2=PD1[is.element(mid,midd),]
PD3=mice(PD2, meth=c('pmm'))
# Final form of PD data 
PD4=complete(PD3)

# Each dimension of predictor data
p1=6 # Six pre-specified sites for each tooth by hygienists
p2=28 # Total number of tooth
p3=2 # Measurement of periodontal pocket depth(PPD), and clinical attachment level(CAL)

p=p1*p2*p3 # Number of variables
q=5 # Five covariates(Age, gender, body mass index(BMI), smoking status, hemoglobin A1c(HbAlc))
n=length(midd) # Number of data
myy=matrix(0,n,p)
myx=matrix(0,n,q)

for (i in 1:n){
  myy[i,1:(p/2)]=PD4[(168*i-167):(168*i),4]
  myy[i,(p/2+1):p]=PD4[(168*i-167):(168*i),5]
  myx[i,]=as.numeric(PD4[168*i,6:10])
}

dimen <- c(6,28,2)
nvars <- prod(dimen)
X <- array(0,c(n,nvars))
for(i in 1:n){
  X[i,] <- array(myy[i,],nvars)
}

Y <- myx[,2] # Setting gender as a class label(Man, Woman), Binary classification
Y <- Y+1

# Number of replicates. Please set it to 100 to reproduce the real data analysis for PD.
R = 100
ncores <- 8
cl = makeCluster(ncores)
registerDoParallel(cl)

LASSO_Gender <- foreach(j = 1:R,.combine = rbind,
                      .packages = c("pamr","tensr","ramify","stats",'rTensor','expm',"TULIP",'glmnet','splitTools')) %dopar% {
                        # Fix the seed for each iteration in Parallel computing
                        set.seed(j*552)
                        # Randomly split the train data(65%) and test data(35%)
                        part = partition(Y, c(0.65,0.35))
                        Xt = X[part$'1',] # Train data on predictor matrix
                        Xtr = X[part$'2',] # Test data on predictor matrix
                        Yt = Y[part$'1'] # Train data on class label
                        Ytr = Y[part$'2'] # Test data on class label
                        nsam = length(Ytr)
                        mydata = list(x = Xt, y = Yt)
                        # Cross-validation for the model
                        CV1=cv.glmnet(Xt,Yt,family="multinomial",  alpha = 1, standardize = TRUE, nfolds = 5)
                        # Fit the model with optimal lambda value 
                        Model1<-glmnet(Xt,Yt,family="multinomial",  alpha = 1, standardize = TRUE, lambda = CV1$lambda.min)
                        # Estimate class label on test set by using trained model
                        res1=predict(Model1,Xtr,type="response")
                        res1=apply(res1, 1, which.max)
                        ret <- list(er = sum(res1 != Ytr)/nsam*100,lambda = CV1$lambda.min)
                        ret   
                      }
LASSO_Gender <- as.data.frame(LASSO_Gender)
error <- as.numeric(LASSO_Gender$er)
# Average classification error rate on 100 simulations
mean(error)
# Standard error rate on 100 simulations.
sd(error)/sqrt(R)
stopCluster(cl)
