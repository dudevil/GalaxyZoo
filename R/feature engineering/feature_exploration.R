require(data.table)

features <- fread(paste0(getwd(),'/data/features/kmeans_features_1000c.csv'))

feature_correlations <- cor(features[,2:ncol(features), with=F])

features_dev <- sapply(feature_correlations, sd)
