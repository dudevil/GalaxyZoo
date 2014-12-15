require(randomForest)
require('glmnet')

source(paste0(getwd(), '/R/_utils.R'))

sets <- split.dataset(load.galaxies('/data/tidy/kmeans_features_c1000_20k.csv'))

train.set <- sets$train.set
test.set <- sets$test.set

fit.glmnet <- cv.glmnet(data.matrix(train.set[2:2001]),
                     y=data.matrix(train.set[2002:ncol(train.set)]),
                     family = 'mgaussian',
                     alpha = 0.1, nlambda = 1000)

glm.pred <- predict(fit.glmnet, data.matrix(train.set[2:2001]), s='lambda.min')

glm.predict.test <- predict(fit.glmnet,
                            data.matrix(test.set[2:2001]),
                            s='lambda.min')

features <- sample(2:2001, size=500)

rf <- randomForest(x=train.set[features],
                   y=train.set$Class1.1,
                   nodesize=500)


fr.pred <- predict(rf, test.set[features])

rmse(fr.pred, test.set$Class1.1)
