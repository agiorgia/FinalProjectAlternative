library(readxl)
library(stats)
library(mclust)
library(writexl)

# range of cluster counts
candidate_k <- seq(2,6)
# number of folds
F <- 10

# a function to find knee in a curve
findKnee <- function(val,percent=10) {
  val <- which(abs(val/val[length(val)]-1) <= percent/100)
  return(min(val))
}

# load data
measPt <- read_excel("FinalProjData_Q2_RGBOnly.xlsx")

# create folds
foldIndex <- sample(seq(1,F), nrow(measPt), replace=T)
loglikeEM <- matrix(0,nrow=length(candidate_k)*F, ncol=3)
colnames(loglikeEM) <- c("k","holdoutLogLike","fold")

idx <- 0
for (i in seq(1,F)) {
  measPtTrain <- measPt[foldIndex != i,]
  measPtTest <- measPt[foldIndex == i,]
  # measPtTest <- measPtTrain
  
  for (Kest in candidate_k) {
    idx <- idx + 1
    # kmeans will initialize the assignments
    mykmeans <- kmeans(measPtTrain, centers=Kest, iter.max=10, nstart=1)
    # run Mstep first to get cluster parameter estimates
    msEst <- mstep(modelName="VVV", data=measPtTrain, z=unmap(mykmeans$cluster))
    # run EM until convergence for training data
    myEMtrain <- em(modelName=msEst$modelName, data=measPt, parameters=msEst$parameters)
    # run Estep on test data
    testEM <- estep(data=measPtTest, modelName = myEMtrain$modelName, parameters=myEMtrain$parameters)
    # get test log likelihood 
    loglikeEM[idx,] <- c(Kest,testEM$loglik,i)
  }
}

loglikeEM <- as.data.frame(loglikeEM)
loglikeEM$fold<- as.factor(loglikeEM$fold)

# average loglike over fold
foldAvgloglikeEM <- matrix(0,nrow=length(candidate_k),ncol=1)
for (myidx in seq(1,length(candidate_k))) {
  foldAvgloglikeEM[myidx] <- mean(loglikeEM$holdoutLogLike[loglikeEM$k==myidx])
}

# select EM model based on knee in log likelihood curve
bestK <- findKnee(foldAvgloglikeEM,15)
# use all data to select model parameters
# initialize with kmeans
mykmeans <- kmeans(measPt, centers=bestK, iter.max=10, nstart=1)

df = data.frame(mykmeans$cluster)
write_xlsx(df, "PixelGroupings_RGBOnly.xlsx")
