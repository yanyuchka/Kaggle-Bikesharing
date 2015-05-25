#!/usr/bin/Rscript
# this script creates a baseline submission using Random Forests
# simply forked from 'Random Forest Benchmark' at Kaggle
# it will be properly commented later

require(ggplot2)
require(lubridate)
require(randomForest)

set.seed(1)

train <- read.csv("./data/train.csv")
test <- read.csv("./data/test.csv")


extract.features <- function(data) {
  features <- c("holiday",
                "workingday",
                "temp",
                "atemp",
                "humidity",
                "hour")
  data$hour <- hour(ymd_hms(data$datetime))
  return(data[,features])
}

train.fea <- extract.features(train)
test.fea  <- extract.features(test)

submission <- data.frame(datetime=test$datetime, count=NA)


for (i_year in unique(year(ymd_hms(test$datetime)))) {
  for (i_month in unique(month(ymd_hms(test$datetime)))) {
    test.locs   <- year(ymd_hms(test$datetime))==i_year & month(ymd_hms(test$datetime))==i_month
    test.subset <- test[test.locs,]
    train.locs  <- ymd_hms(train$datetime) <= min(ymd_hms(test.subset$datetime))
    cat("Year: ", i_year, "\tMonth: ", i_month, "\n")
    rf1 <- randomForest(extract.features(train[train.locs,]), train[train.locs,"casual"], ntree=100)
    print("casual done")
    rf2 <- randomForest(extract.features(train[train.locs,]), train[train.locs,"registered"], ntree=100)
    print("registered done")
    submission[test.locs, "count"] <- predict(rf1, extract.features(test.subset)) + predict(rf2, extract.features(test.subset))
  }
}

write.csv(submission, file = "to_submit_rf_R.csv", row.names=FALSE)


