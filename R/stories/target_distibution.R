#
# Have a look at the resulting class distributions
#
#

require('ggplot2')
require('reshape2')

#load train.solutions
source(paste0(getwd(),'/R/data processing/0 1 read data.R'))

#n_galaxies = nrow(train.solutions)
#sum_classes = sapply(train.solutions[,-1], sum)/n_galaxies
#sum_classes = sum_classes / sum(sum_classes)
#sum_classes = data.frame(sum_classes)
#sum_classes['class'] = rownames(sum_classes)
classes <- train.solutions[,-1]
names(classes) <- gsub('Class', '', names(classes))
#classes['class'] <- names(classes)
classes.molten = melt(classes, id.vars=c(),
                      variable.names='Class',
                      value.name = 'Prob')

bar.classes = ggplot(data = classes.molten,
                      aes(x = class, y = prob))
bar.classes + geom_boxplot()
