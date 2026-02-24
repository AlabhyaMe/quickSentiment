#' Run a Full Text Classification Pipeline on Preprocessed Text
#'
#' This function takes a data frame with pre-cleaned text and handles the
#' data splitting, vectorization, model training, and evaluation.
#'
#' @param vect_method A string specifying the vectorization method.
#'   Defaults to \code{"bag_of_words"}.
#'   \itemize{
#'     \item \code{"bag_of_words"} (Alias: \code{"bow"}) - Standard count of words.
#'     \item \code{"term_frequency"} (Alias: \code{"tf"}) - Normalized counts.
#'     \item \code{"tfidf"} (Alias: \code{"tf-idf"}) - Term Frequency-Inverse Document Frequency.
#'     \item \code{"binary"} - Presence/Absence (1/0).
#'   }
#' @param model_name A string specifying the model to train.
#'   Defaults to \code{"logistic_regression"}.
#'   \itemize{
#'     \item \code{"random_forest"} (Alias: \code{"rf"})
#'     \item \code{"xgboost"} (Alias: \code{"xgb"})
#'     \item \code{"logistic_regression"} (Alias: \code{"logit"}, \code{"glm"})
#'   }
#' @param df The input data frame.
#' @param text_column_name The name of the column containing the **preprocessed** text.
#' @param sentiment_column_name The name of the column containing the original target labels (e.g., ratings).
#' @param n_gram The n-gram size to use for BoW/TF-IDF. Defaults to 1.
#' @param parallel If TRUE, runs model training in parallel. Default FALSE.
#' @param tune Logical. If TRUE, the pipeline will perform hyperparameter tuning
#'    for the selected model. Defaults to FALSE. [NEW]
#' @return A list containing the trained model object, the DFM template,
#'   class levels, and a comprehensive evaluation report.
#' @importFrom parallel detectCores makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach registerDoSEQ
#' @importFrom pROC roc auc coords
#' @export
#' @examples
#' df <- data.frame(
#'   text = c("good product", "excellent", "loved it", "great quality",
#'            "bad service", "terrible", "hated it", "awful experience",
#'            "not good", "very bad", "fantastic", "wonderful"),
#'   y = c("P", "P", "P", "P", "N", "N", "N", "N", "N", "N", "P", "P")
#' )
#'
#'
#' out <- pipeline("bow", "naive_bayes", df, "text", "y")
#'
pipeline <- function(vect_method,
  model_name,
  df,
  text_column_name,
  sentiment_column_name,
  n_gram = 1,
  tune = FALSE,
  parallel=FALSE) {

    stopf <- function(...) stop(sprintf(...), call. = FALSE)

    # --- 1. CLEAN & TRANSLATE ARGUMENTS ---
    #to lower and trim ensures small typos don't break the function and allows for more flexible input
    vect_method <- tolower(trimws(vect_method))
    model_name  <- tolower(trimws(model_name))
    n_gram <- as.integer(n_gram)

    # ALIASING: VECTORIZERS
    if (vect_method == "bow") vect_method <- "bag_of_words"
    if (vect_method == "tf")  vect_method <- "term_frequency"
    if (vect_method == "tf-idf")  vect_method <- "tfidf"

    # ALIASING: Convert shortcuts to official names
    if (model_name == "rf")    model_name <- "random_forest"
    if (model_name == "xgb")   model_name <- "xgboost"
    if (model_name == "nb")   model_name <- "naive_bayes"
    if (model_name %in% c("logit","glm","logistic")) model_name <- "logistic_regression"


  # Ensure Vector method is in Input
    allowed_vect <- c("bag_of_words", "binary", "term_frequency", "tfidf")
    if (!vect_method %in% allowed_vect) {
      stopf("Vectorizer '%s' is not supported. Use: %s.",
            vect_method, paste(allowed_vect, collapse = ", "))
    }

  # Ensure Model name is in Input
    allowed_models <- c("logistic_regression", "random_forest", "xgboost", "naive_bayes")
    if (!model_name %in% allowed_models) {
      stopf("Model '%s' is not supported. Use: %s.",
            model_name, paste(allowed_models, collapse = ", "))
    }

  # Model list that can be used
    models_list <- list(
      logistic_regression = logit_model,
      random_forest       = rf_model,
      xgboost             = xgb_model,
      naive_bayes         = nb_model
    )


  if (!is.data.frame(df)) stopf("`df` must be a data.frame.")
  if (!text_column_name %in% names(df)) stopf("Column '%s' not found.", text_column_name)
  if (!sentiment_column_name %in% names(df)) stopf("Column '%s' not found.", sentiment_column_name)

  #FOR GLMNET ONLY
  if (isTRUE(parallel)) {
    # Dynamically detect cores and register
    n_cores <- parallel::detectCores() - 1
    cl <- parallel::makeCluster(n_cores)
    doParallel::registerDoParallel(cl)

    # Ensure the cluster stops even if the code crashes
    on.exit(parallel::stopCluster(cl), add = TRUE)
    on.exit(foreach::registerDoSEQ(), add = TRUE)
  }

  message(paste0("--- Running Pipeline: ", toupper(vect_method), " + ", toupper(model_name), " ---\n"))
  df$sentiment <- as.factor(df[[sentiment_column_name]])

  # --- 1. DATA CLEANING & PREP ---
  # Remove rows where the cleaned text is empty

  initial_rows <- nrow(df)
  df[[text_column_name]] <- trimws(as.character(df[[text_column_name]]))
  df[[text_column_name]][df[[text_column_name]] == ""] <- NA_character_

  df <- df[!is.na(df[[text_column_name]]) & !is.na(df$sentiment), , drop = FALSE]


  rows_dropped <- initial_rows - nrow(df)

  if (rows_dropped > 0) {
    warning(sprintf("Dropped %d row(s) with missing/empty text or missing labels.", rows_dropped),
            call. = FALSE)
  }

  if (nrow(df) < 5) stopf("Not enough rows after filtering (%d).", nrow(df))
  if (nlevels(df$sentiment) < 2) stopf("Need at least 2 sentiment classes after filtering.")

  # --- 2. TRAIN/TEST SPLIT ---
  target_vector <- df[[sentiment_column_name]]

  # Base R Stratified Split
  train_idx <- unlist(lapply(split(seq_along(target_vector), target_vector), function(idx) {
    sample(idx, size = round(0.8 * length(idx)))
  }))

  # Ensure it's a clean numeric vector
  train_idx <- as.numeric(train_idx)

  data_train <- df[train_idx, ]
  data_test <- df[-train_idx, ]

  message(paste0("Data split: ", nrow(data_train), " training rows, ", nrow(data_test), " test rows.\n"))

  # --- 3. VECTORIZATION ---
  # This now operates on the pre-cleaned text column
  message(sprintf("Vectorizing with %s (ngram=%d)...", toupper(vect_method), n_gram))

  fit <- BOW_train(data_train[[text_column_name]],
                   weighting_scheme = vect_method,
                   ngram_size = n_gram)
  X_train <- fit$dfm_template

  X_test <- BOW_test(data_test[[text_column_name]],fit)

  # Prepare the response variables (y)
  y_train <- data_train$sentiment
  y_test <- data_test$sentiment

  # --- 4. MODEL TRAINING & PREDICTION ---

  model_results <- models_list[[model_name]](
    X_train,
    y_train,
    X_test,
    parallel = parallel,
    tune=tune
    )
  if (is.null(model_results$model) || is.null(model_results$pred)) {
    stopf("Model function '%s' must return a list with elements `model` and `pred`.", model_name)
  }


  # --- 5. EVALUATION ---
  # A Evaluation
  predictions <- model_results$pred
  predictions_factor <- factor(predictions, levels = levels(y_test))

  # Using Custom function for evaluation
  evaluation <- evaluate_metrics(
    predicted = model_results$pred,
    actual = y_test
)

  # B ROC/AUC
  roc_guidance <- NULL

  # We only calculate ROC for Binary classification (2 levels)
  if (nlevels(y_test) == 2) {
    target_class <- levels(y_test)[2] # Usually the "Positive" class
    #  Extract strictly by name.
    # Since we fixed the models, we TRUST that 'probs' is a matrix with this column.
    pos_probs <- model_results$probs[, target_class]

    # Calculate ROC
    roc_obj <- pROC::roc(response = y_test, predictor = as.vector(pos_probs), quiet = TRUE)

    # Get the "Best" threshold using Youden's J statistic
    best_coords <- pROC::coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"), transpose = FALSE)

    roc_guidance <- list(
      auc = as.numeric(pROC::auc(roc_obj)),
      best_threshold = best_coords$threshold,
      roc_object = roc_obj # Save this if you want to plot it later
    )
  }
 # Map the model_name to our internal shorthand for the router
 internal_type_map <- c(
  "logistic_regression" = "logit",
  "random_forest" = "rf",
  "xgboost" = "xgb",
  "naive_bayes" = "nb"
)
  # --- 6. RETURN FINAL RESULTS ---
  final_output <- list(
      trained_model = model_results$model,
      model_type    = internal_type_map[model_name],
      best_lambda   = model_results$best_lambda, # <- for glmnet only
      dfm_template = fit,
      class_levels = levels(y_test),
      ngram_size_used = n_gram,
      vectorize_test_function = BOW_test,
      evaluation_report = evaluation,
      guidance = roc_guidance
      )

      # Print the final completion message
  message("\n======================================================")
  message(sprintf(" PIPELINE COMPLETE: %s + %s", toupper(vect_method), toupper(model_name)))
  message(sprintf(" Model AUC: %.3f", roc_guidance$auc))
  message(sprintf(" Recommended ROC Threshold: %.3f", roc_guidance$best_threshold))
  message("======================================================\n")

  return(final_output)
  }
