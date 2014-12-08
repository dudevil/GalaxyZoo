rmse <- function(y, y_hat){
  return(sqrt(mean(y-y_hat)^2))
}
