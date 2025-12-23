library(rTensor)
library(tensr)
library(expm)
library(splitTools)

# Since the tensor data is stored in array, n mode matricization on tensor data
unfold <- function(X,mode){
  dimen = dim(X)
  order = seq(1,length(dimen))
  order[mode] = NA
  order <- as.numeric(na.omit(c(mode,order)))
  A <- array(aperm(X,order),c(dimen[mode],prod(dimen)/dimen[mode]))
  A
}

# Covariance function(CS)
CS <- function(r,p){
  S <- diag(p)
  for (i in 1:p){
    for (j in 1:p){
      if(i != j){
        S[i,j] = r
      }
    }
  }
  S
}

# Covariance function(AR)
AR <- function(r,p){
  S <- diag(p)
  for(i in 1:p){
    for (j in 1:p){
      S[i,j] = r^abs(i-j)
    }
  }
  S
}

# Univariate soft-thresholding function
st = function(X,lambda){
  if(abs(X) >= lambda){
    a = sign(X) * (abs(X) - lambda)
  }
  else{
    a = 0
  }
  a
}

# Multi-dimensional soft-thresholding function
shrink <- function(X,lambda){
  nvars = prod(dim(X))
  A = array(X,nvars)
  for (i in 1:nvars){
    A[i] = st(A[i],lambda)
  }
  B = array(A,dim(X))
  B
}

#### Setting the model for simulation
# nk: Number of samples for each class
# K: Number of class
# dim: Dimension for each tensor data
# V: List of each class centroid
# Sigma: Covariance structure
# seed: Specifiy seed
Simulation.Model <- function(nk,K,dim,V,Sigma,seed){
  nvars <- prod(dim)
  n = nk * K
  # Setting class label
  Y <- c()
  for(i in 1:K){
    Y <- c(Y,rep(i,nk))
  }
  
  dsigma = array(list(),length(dim))
  for (i in 1:length(dim)){
    dsigma[[i]] = sqrtm(sigma[[i]])
  }
  
  mu = array(list(),K)
  for(i in 1:K){
    mu[[i]] = atrans(V[[i]],dsigma)
  }
  
  set.seed(seed)
  # Train Set
  x <- array(list(),n)
  for(i in 1:n){
    x[[i]] <- atrans(array(rnorm(nvars),dim),dsigma)
  }
  
  # Validation Set
  vax <- array(list(),n)
  for(i in 1:n){
    vax[[i]] <- atrans(array(rnorm(nvars),dim),dsigma)
  }
  
  result <- list()
  result$X <- x
  result$Vax <- vax
  result$Y <- Y
  class(result) <- "Data Generating for simulation"
  return(result)
}


# Splitting train data and test data on k-fold cross-validation.
cv_fold <- function(X,Y,nfold = 5){
  n = length(X)
  dimen = dim(X[[1]])
  K=  length(unique(Y))
  nvars = prod(dimen)
  check = 0
  for(i in 1:n){
    if(sum(dimen == dim(X[[i]])) == length(dimen)){
      check = check +1
    }
  }
  if(check == n){
    foldid <- sample(rep(seq(nfold),length = n))
    folds = array(list(),nfold)
    train <- array(list(),nfold)
    y.train <- array(list(),nfold)
    test <- array(list(),nfold)
    y.test <- array(list(),nfold)
    for(j in 1:nfold){
      which1 <- foldid == j
      folds[[j]] = which(foldid == j)
      Xnew <- array(0,c(n,nvars+1))
      for(i in 1:n){
        Xnew[i,] = c(array(X[[i]],nvars),Y[i])
      }
      train.data <- Xnew[!which1,,drop = FALSE]
      test.data <- Xnew[which1,,drop = FALSE]
      ntrain = nrow(train.data)
      ntest = nrow(test.data)
      for(i in 1:ntrain){
        train[[j]][[i]] = array(train.data[i,1:nvars],dimen)
        y.train[[j]] = array(train.data[,nvars+1],ntrain)
      }
      for(i in 1:ntest){
        test[[j]][[i]] = array(test.data[i,1:nvars],dimen)
        y.test[[j]] = array(test.data[,nvars+1],ntest)
      }
    }
    Data <- list(train = train, trainy = y.train, test = test, testy = y.test, folds = folds)
    class(Data) <- "Data for Cross Validation"
    Data
  }
  else{
    stop("Data Dimension doesn't match")
  }
}

# Assigning fold index for K-fold cross-validation.
fold.assigned <- function(X,Y,folds){
  n = length(Y)
  dimen = dim(X[[1]])
  K = length(unique(Y))
  nvars <- prod(dimen)
  check = 0
  for(i in 1:n){
    if(sum(dimen == dim(X[[i]])) == length(dimen)){
      check = check +1
    }
  }
  if(check == n){
    entire <- seq(1:n)
    nfold <- length(folds)
    train <- array(list(),nfold)
    y.train <- array(list(),nfold)
    test <- array(list(),nfold)
    y.test <- array(list(),nfold)
    for(i in 1:nfold){
      train.index <- setdiff(entire, folds[[i]])
      test.index <- folds[[i]]
      Xnew <- array(0,c(length(Y),nvars+1))
      for(j in 1:n){
        Xnew[j,] <- c(array(X[[j]],nvars),Y[j])
      }
      train.data <- Xnew[train.index,,drop = FALSE]
      test.data <- Xnew[test.index,,drop = FALSE]
      ntrain <- nrow(train.data)
      ntest <- nrow(test.data)
      
      for(j in 1:ntrain){
        train[[i]][[j]] <- array(train.data[j,1:nvars],dimen)
        y.train[[i]] <- array(train.data[,nvars + 1],ntrain)
      }
      for(j in 1:ntest){
        test[[i]][[j]] <- array(test.data[j,1:nvars],dimen)
        y.test[[i]] <- array(test.data[,nvars + 1],ntest)
      }
    }
    Data <- list(train = train, trainy = y.train, test = test, testy = y.test, folds = folds)
    class(Data) <- "Data for Cross Validation when folds are assigned"
    Data
  }
  else{
    stop("Data Dimension doesn't match")
  }
}

# Assigning fold index for K-fold cross-validation.
balanced.folds <- function(y, nfolds = min(min(table(y)), 10)) {
  totals <- table(y)
  fmax <- max(totals)
  nfolds <- min(nfolds, fmax)     
  nfolds= max(nfolds, 2)
  # makes no sense to have more folds than the max class size
  folds <- as.list(seq(nfolds))
  yids <- split(seq(y), y) 
  # nice we to get the ids in a list, split by class
  ###Make a big matrix, with enough rows to get in all the folds per class
  bigmat <- matrix(NA, ceiling(fmax/nfolds) * nfolds, length(totals))
  for(i in seq(totals)) {
    cat(i)
    if(length(yids[[i]])>1){bigmat[seq(totals[i]), i] <- sample(yids[[i]])}
    if(length(yids[[i]])==1){bigmat[seq(totals[i]), i] <- yids[[i]]}
    
  }
  smallmat <- matrix(bigmat, nrow = nfolds)# reshape the matrix
  ### Now do a clever sort to mix up the NAs
  smallmat <- permute.rows(t(smallmat))   ### Now a clever unlisting
  # the "clever" unlist doesn't work when there are no NAs
  #       apply(smallmat, 2, function(x)
  #        x[!is.na(x)])
  res <-vector("list", nfolds)
  for(j in 1:nfolds) {
    jj <- !is.na(smallmat[, j])
    res[[j]] <- smallmat[jj, j]
  }
  return(res)
}

# Randomly assigning Train data and Test data(Real data analysis)
Train.Test.split <- function(X,Y,part = NULL,ratio = c(0.7,0.3)){
  n = length(Y)
  K = length(unique(Y))
  dimen = dim(X[[1]])
  nvars <- prod(dimen)
  check = 0
  for(i in 1:n){
    if(sum(dimen == dim(X[[i]])) == length(dimen)){
      check = check +1
    }
  }
  if(check == n){
    if(is.null(part)){
      part <- partition(Y,p = ratio)
      train.set <- array(list(),length(part[[1]]))
      y.train <- array(NA,length(part[[1]]))
      test.set <- array(list(),length(part[[2]]))
      y.test <- array(NA,length(part[[2]]))
      
      Xnew <- array(0,c(n,nvars+1))
      for(i in 1:n){
        Xnew[i,] <- c(array(Xs[[i]],nvars),Y[i])
      }
      
      train.data <- Xnew[part[[1]],,drop = FALSE]
      test.data <- Xnew[part[[2]],,drop = FALSE]
      ntrain <- nrow(train.data)
      ntest <- nrow(test.data)
      
      for(i in 1:ntrain){
        train.set[[i]] <- array(train.data[i,1:nvars],dimen)
        y.train[i] <- train.data[i,nvars+1]
      }
      
      for(i in 1:ntest){
        test.set[[i]] <- array(test.data[i,1:nvars],dimen)
        y.test[i] <- test.data[i,nvars+1]
      }
      Data <- list(train = train.set, trainy = y.train, test = test.set, testy = y.test, part = part)
      class(Data) <- "Train and Test data split"
      Data
    }
    else{
      train.set <- array(list(),length(part[[1]]))
      y.train <- array(NA,length(part[[1]]))
      test.set <- array(list(),length(part[[2]]))
      y.test <- array(NA,length(part[[2]]))
      Xnew <- array(0,c(n,nvars+1))
      for(i in 1:n){
        Xnew[i,] <- c(array(Xs[[i]],nvars),Y[i])
      }
      train.data <- Xnew[part[[1]],,drop = FALSE]
      test.data <- Xnew[part[[2]],,drop = FALSE]
      ntrain <- nrow(train.data)
      ntest <- nrow(test.data)
      for(i in 1:ntrain){
        train.set[[i]] <- array(train.data[i,1:nvars],dimen)
        y.train[i] <- train.data[i,nvars+1]
      }
      for(i in 1:ntest){
        test.set[[i]] <- array(test.data[i,1:nvars],dimen)
        y.test[i] <- test.data[i,nvars+1]
      }
      Data <- list(train = train.set, trainy = y.train, test = test.set, testy = y.test, part = part)
      class(Data) <- "Train and Test data split when partition is assigned"
      Data
    }
  }
  else{
    stop("Data Dimension doesn't match")
  }
}

