require('e1071')
require('foreach')
require('doMC')

source(paste0(getwd(), '/R/data processing/0 1 read data.R'))
source(paste0(getwd(), '/R/_utils.R'))

kmeans.features <- read.csv(paste0(getwd(),
                                   '/data/tidy/kmeans_features_c500_5k.csv'))
names(kmeans.features)[1] <- 'GalaxyID'
galaxies <- merge(kmeans.features , train.solutions, by='GalaxyID')

# set up a train and test sets
index <- 1:nrow(galaxies)
trainindex <- sample(index, trunc(length(index)*0.8))

train.set <- galaxies[trainindex,]
test.set <- galaxies[-trainindex,]

# set up cluster
#cl <- makeCluster(detectCores())
# export features to the cluster, they will we in scope of worker threads
#clusterExport(cl=cl, varlist=c("train.set", "test.set"))

#classes <- names(train.set[2002:ncol(train.set)])
classes <- names(train.set[2002:2004])
# a function that trains an svm model and makes predictions
run.svm <- function(class){
  # train svm with linear kernel
  model.svm <- svm(x = train.set[,2:2001],
                   y = train.set[,class],
                   kernel='linear')
  # save model
  print(paste('Class', class, 'trained.'))
  #save(model.svm, file=paste0(getwd(),'/data/models/svm_', class, '.rda'))
  # predict on the test set
  return(predict(model.svm, test.set[,2:2001]))
}

print('Starting SVM training and prediction...')
# actually run the parallel jobs
#results <- data.frame(mclapply(classes, run.svm,
#                               mc.cores = getOption('mc.cores', 3L)))
#stopCluster(cl)
registerDoMC(3)

res <- foreach(class = classes) %dopar% {
  model.svm <- svm(x = train.set[,2:2001],
                   y = train.set[,class],
                   kernel='linear')
  # save model
  print(paste('Class', class, 'trained.'))
  save(model.svm, file=paste0(getwd(),'/data/models/svm_', class, '.rda'))
  # predict on the test set
  predict(model.svm, test.set[,2:2001])
}

results <- data.frame(res)
print(paste('RMSE: ', rmse(as.matrix(results), test.set[,classes])))
write.csv(results, row.names=FALSE,
          file = paste0(getwd(),"/data/tidy/svmpred_linear_untuned.csv"))
