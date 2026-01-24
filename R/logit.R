#' Train a Regularized Logistic Regression Model using glmnet
#'
#' This function trains a logistic regression model using Lasso regularization
#' via the glmnet package. It uses cross-validation to automatically find the
#' optimal regularization strength (lambda).
#'
#' @param train_vectorized The training feature matrix (e.g., a `dfm` from quanteda).
#'   This should be a sparse matrix.
#' @param Y The response variable for the training set. Should be a factor for
#'   classification.
#' @param test_vectorized The test feature matrix, which must have the same
#'   features as `train_vectorized`.
#'
#' @return A list containing two elements:
#'   \item{pred}{A vector of class predictions for the test set.}
#'   \item{model}{The final, trained `cv.glmnet` model object.}
#'
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict
#' @importFrom doParallel registerDoParallel
#'
#' @export
logit_model <- function(train_vectorized, Y, test_vectorized, parallel=F){
  message("\n--- Running Logistic Regression (logit) Function ---\n")


  cat("1. Training the glmnet model with 5-fold cross-validation...\n")
  cv_model <- cv.glmnet(
    x = train_vectorized,
    y = Y ,
    family = "multinomial", # This specifies logistic regression
    alpha = 1,           # This specifies Lasso regularization (great for text)
    nfolds = 5,          # Number of cross-validation folds
    parallel = parallel     # Tell glmnet to use the parallel core
  )
  cat("   - CV complete. Best lambda (lambda.min) found:", round(cv_model$lambda.min, 6), "\n")

  y_pred <- predict(cv_model,
                    newx = test_vectorized,
                    s = "lambda.min",
                    type = "class")
  results <- list(
    pred = y_pred,
    model = cv_model
  )
  message("--- Logit function complete. Returning results. ---\n")
  return(results)

}




