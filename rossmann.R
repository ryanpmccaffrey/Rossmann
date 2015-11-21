  # set working directory, clear variables and load required packages
  setwd("/Users/ryanmccaffrey/Documents/Data Science/Kaggle/Rossmann Stores")
  rm(list=ls())
  require(data.table)
  require(xgboost)
  
  # read in training and test data sets
  train <- fread("train.csv", stringsAsFactors=T)
  test <- fread("test.csv", stringsAsFactors=T)
  store <- fread("store.csv", stringsAsFactors=T)
  
  # merge store data with train and test data sets
  train <- merge(train,store,by="Store")
  test <- merge(test,store,by="Store")
  
  # remove training data with zero sales and closed store
  train <- train[train$Sales>0,]
  train <- train[train$Open == 1,]
  
  # set na values to zero
  train[is.na(train)] <- 0
  test[is.na(test)] <- 0
  
  # convert 'date' from chr to date
  train[,Date:=as.Date(train$Date)]
  test[,Date:=as.Date(test$Date)]
  
  # break up train 'date' into day, month, year; convert char and factor variables to numeric ids
  train[,DayOfMonth:=as.integer(format(Date, "%d"))]
  train[,Month:=as.integer(format(Date, "%m"))]
  train[,Year:=as.integer(format(Date, "%Y"))]
  train[,Date:=NULL]
  train[,Customers:=NULL]
  train[,Store:=as.numeric(Store)]
  train[,StateHoliday:=as.numeric(as.factor(StateHoliday))]
  train[,SchoolHoliday:=as.numeric(as.factor(SchoolHoliday))]
  train[,StoreType:=as.numeric(as.factor(StoreType))]
  train[,Assortment:=as.numeric(as.factor(Assortment))]
  train[,PromoInterval:=as.numeric(as.factor(PromoInterval))]
  
  # break up test 'date' into day, month, year; convert char and factor variables to numeric ids
  test[,DayOfMonth:=as.integer(format(Date, "%d"))]
  test[,Month:=as.integer(format(Date, "%m"))]
  test[,Year:=as.integer(format(Date, "%Y"))]
  test[,Date:=NULL]
  test[,Customers:=NULL]
  test[,Store:=as.numeric(Store)]
  test[,StateHoliday:=as.numeric(as.factor(StateHoliday))]
  test[,SchoolHoliday:=as.numeric(as.factor(SchoolHoliday))]
  test[,StoreType:=as.numeric(as.factor(StoreType))]
  test[,Assortment:=as.numeric(as.factor(Assortment))]
  test[,PromoInterval:=as.numeric(as.factor(PromoInterval))]
  
  # convert 'competition open since' variables to 'number of months open'
  train[,CompetitionOpenForMonths:=((Year-CompetitionOpenSinceYear)*12 + (Month-CompetitionOpenSinceMonth))]
  train[,CompetitionOpenSinceYear:=NULL]
  train[,CompetitionOpenSinceMonth:=NULL]
  test[,CompetitionOpenForMonths:=((Year-CompetitionOpenSinceYear)*12 + (Month-CompetitionOpenSinceMonth))]
  test[,CompetitionOpenSinceYear:=NULL]
  test[,CompetitionOpenSinceMonth:=NULL]
  
  # if negative, set 'number of months open' variable to zero
  train$CompetitionOpenForMonths <- ifelse(train$CompetitionOpenForMonths<0,0,train$CompetitionOpenForMonths)
  test$CompetitionOpenForMonths <- ifelse(test$CompetitionOpenForMonths<0,0,test$CompetitionOpenForMonths)
  
  # sort test data by 'id' and then remove 'id' column
  test <- test[order(Id),]
  test[,Id:=NULL]
  
  # function for calculating root mean squre percentage error (evaluation criterion)
  rmspe<- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    elab<-exp(as.numeric(labels))-1
    epreds<-exp(as.numeric(preds))-1
    err <- sqrt(mean((epreds/elab-1)^2))
    return(list(metric = "rmspe", value = err))
  }
  
  k <- sample(nrow(train),10000)
  dval <- xgb.DMatrix(data=data.matrix(train)[k,-3],label=log(train$Sales+1)[k])
  dtrain <- xgb.DMatrix(data=data.matrix(train)[-k,-3],label=log(train$Sales+1)[-k])
  watchlist <- list(val=dval,train=dtrain)
  param <- list(  
    objective="reg:linear", 
    booster="gbtree",
    eta=0.02, # 0.06, #0.01,
    max_depth=10, #changed from default of 8
    subsample=0.9, # 0.7
    colsample_bytree=0.7 # 0.7
  )
  
  clf <- xgb.train(   
    params              = param, 
    data                = dtrain, 
    nrounds             = 3200, 
    verbose             = 0,
    early.stop.round    = 100,
    watchlist           = watchlist,
    maximize            = FALSE,
    feval               = rmspe
  )
  
  # write final prediction to csv file
  pred <- exp(predict(clf, data.matrix(test[,]))) -1
  submission <- data.frame(Id=1:length(pred),Sales=pred)
  write.csv(submission, "rossmann_xgb_eta02_maxd10_nrounds3200_esr100.csv",row.names=F)
