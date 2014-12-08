require('e1071')

source(paste0(getwd(), '/R/data processing/0 1 read data.R'))
source(paste0(getwd(), '/R/_utils.R'))


kmeans.features <- read.csv(paste0(getwd(),'/data/tidy/pooled_features_c500_p8_s1.csv'))

names(kmeans.features)[1] <- 'GalaxyID'
galaxies <- merge(kmeans.features , train.solutions, by='GalaxyID')

index <- 1:nrow(galaxies)
trainindex <- sample(index, trunc(length(index)*0.8))

train.set.all <- galaxies[trainindex, ]
test.set <- galaxies[-trainindex, ]

my.svm <- svm(Class1.1 ~., data=train.set[,2:2002], kernel='linear')
