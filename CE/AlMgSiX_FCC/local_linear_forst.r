data <- read.csv("/home/gudrun/davidkl/Documents/GPAWTutorial/CE/AlMgSiX_FCC/data/almgsiX_data.csv")
X <- data[,1:ncol(data)]
y <- data[,ncol(data)]

X_train <- X[1:500,]
X_test <- X[500:nrow(X),]
y_train <- y[1:500]
y_test <- y[500:564]

forest <- grf::ll_regression_forest(X_train, y_train, tune.parameters = "all")

pred <- predict(forest, X_train)
error <- sqrt(mean((pred[,1] - y_train)^2))*1000.0

pred_test <- predict(forest, X_test)
error_test <- sqrt(mean((pred_test[,1] - y_test)^2))*1000.0

