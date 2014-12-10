require('DirichletReg')

source(paste0(getwd(), '/R/data processing/0 1 read data.R'))
source(paste0(getwd(), '/R/_utils.R'))

kmeans.features <- read.csv(paste0(getwd(),'/data/tidy/pooled_features_c500_p8_s1.csv'))

names(kmeans.features)[1] <- 'GalaxyID'
galaxies <- merge(kmeans.features , train.solutions, by='GalaxyID')

index <- 1:nrow(galaxies)
trainindex <- sample(index, trunc(length(index)*0.8))

train.set.all <- galaxies[trainindex, ]
test.set <- galaxies[-trainindex, ]

index <- 1:nrow(train.set.all)
valindex <- sample(index, trunc(length(index)*0.8))
train.set <- train.set.all[valindex, ]
valid.set <- train.set.all[-valindex, ]

Class1 <- DR_data(train.set[,2002:2004])

my.dirreg <- DirichReg(Class1 ~., data=train.set[,2:2002])


#my.gbm.try <- gbm(Class1.1~., train.set[,2:2002], distribution = "gaussian",
#                  n.trees = 4000, interaction.depth = 5,
#                  shrinkage = 0.1, cv.folds = 5, verbose = TRUE)

#index <- 1:nrow(train.set.all)
#valindex <- sample(index, trunc(length(index)*0.8))
#train.set <- train.set.all[valindex, ]
#valid.set <- train.set.all[-valindex, ]

#xbst.params <- list(max.depth = 2, eta = 1, objective = "reg:linear")
#dtrain <- xgb.DMatrix(as.matrix(train.set[,2:2002]), label = train.set$Class1.1)
#nround <- 10
#xgb.cv(param, dtrain, nround, nfold=5, metrics={'error'})

#my.xbst <- xgboost(xbst.params, data=as.matrix(train.set[,2:2002]),
                   label = train.set$Class1.1,
                   nrounds = 710)

#my.xbst.prediction <- predict(my.xbst, newdata = as.matrix(test.set[,2:2001]))

#rmse(test.set$Class1.1, my.xbst.prediction)

#xgb.cv(xbst.params, dtrain, nround, nfold=3, metrics={'error'})
