source("http://bioconductor.org/biocLite.R")
biocLite("EBImage")
filePath = "/Users/tatiana/Desktop/images_training_rev1"
fileNames <- list.files(filepath)

extractFeatures <- function(filepath)
{
  processedImage <- readImage(filepath)
  processedImage <- processedImage[120:324,100:324,]
  img = thresh(processedImage[,,1], w=100, h=100, offset=0.1)
  objects <- bwlabel(img)
  allshapeFeatures <- computeFeatures.shape(objects)
  allshapeFeatures <- as.data.frame(allshapeFeatures)
  ind <- which(allshapeFeatures$s.area == max(allshapeFeatures$s.area))
  shapeFeatures <- allshapeFeatures[ind,]
  
  #компактность
  density <- shapeFeatures[2]^2/shapeFeatures[1]
  #отношение меньшего радиуса к большему
  radRelation <- shapeFeatures[5]/shapeFeatures[6]
  
  allmomentFeatures <- computeFeatures.moment(objects)
  momentFeatures <- allmomentFeatures[ind,]
  eccentricity <- momentFeatures[4]
  
  allbasicFeatures <- computeFeatures.basic(objects,processedImage)
  basicFeatures <- allbasicFeatures[ind,]
  featuresVector <- c(density,eccentricity, radRelation, basicFeatures)
  featuresVector <- unname(featuresVector)
  featuresAmmount <- length(featuresVector)
  return(featuresVector)
}
makeFeaturesMatrix <- function()
{
  i = 1;
  fileNames <- list.files(filePath)
  output <- matrix(0, ncol = featuresAmmount, nrow =length(fileNames))
  for (fl in fileNames)
  {
    output[i,] <- unlist(extractFeatures(fl))
    i=i+1
  }
}

boundary = as.integer(nrow(training_solutions)*0.8)
trainSolutionsSet <- training_solutions[2:boundary,]
trainSet <- output[2:boundary,]

testSolutionSet <- training_solutions[(boundary+1):nrow(training_solutions),]
testSet <- output[(boundary+1):nrow(output),]
#
trainNorm <- scale(trainSet)
testNorm <- scale(testSet)
#models?
cvfit <- cv.glmnet(trainNorm, trainSolutionsSet[,2])
fit = glmnet(trainNorm, trainSolutionsSet[,2], alpha = 0.2, nlambda = 20)
coef.apprx = coef(fit, s = 0.5, exact = FALSE)

result <- predict(cvfit, newx = testNorm, type = "response")

tr <- as.data.frame(trainNorm)
test <- as.data.frame(testNorm)
my.gbm.try <- gbm(trainSolutionsSet[,2]~., tr, distribution = "gaussian", n.trees = 4000, interaction.depth = 5,shrinkage = 0.05, cv.folds = 5, verbose = TRUE)
result <- predict(my.gbm.try, newdata= test, type = "response")

#подсчет ошибки
error <- (result - testSolutionSet[,2])^2
error <- sum(error)/length(error)
sqrt(error)

