require('glmnet')
require('pls')
require('gbm')


serg.features <- read.csv(paste0(getwd(),'/data/features/sergeyFeatures.csv'),
                          sep=';')[-c(1,2),]

serg.features[,'filesNames'] = as.numeric(gsub('.jpg', '', serg.features[,'filesNames']))
names(serg.features)[length(serg.features)] <- 'GalaxyID'
train.set = merge(serg.features, train.solutions, by='GalaxyID')



my.cv.glmnet <- cv.glmnet(as.matrix(train.set[,2:85]), train.set[,86],
                          family="gaussian")

my.cv.glmnet <- cv.glmnet(as.matrix(train.set[,2:85]), train.set[,86],
                          family="gaussian")



my.gbm.try <- gbm(Class1.1~., train.set[,2:86], distribution = "gaussian",
                  n.trees = 4000, interaction.depth = 5,
                  shrinkage = 0.05, cv.folds = 5, verbose = TRUE)



#my.log.cv.glmnet <- cv.glmnet(as.matrix(train.set[,2:85]), log(train.set[,86]),
#                          family="gaussian")

