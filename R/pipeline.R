#' Run a Full Text Classification Pipeline on Preprocessed Text
#'
#' This function takes a data frame with pre-cleaned text and handles the
#' data splitting, vectorization, model training, and evaluation.
#'
#' @param vect_method A string specifying the vectorization method (e.g., "bow", "tfidf").
#' @param model_name A string specifying the model to train (e.g., "logit", "rf", "xgb").
#' @param df The input data frame.
#' @param text_column_name The name of the column containing the **preprocessed** text.
#' @param sentiment_column_name The name of the column containing the original target labels (e.g., ratings).
#' @param n_gram The n-gram size to use for BoW/TF-IDF. Defaults to 1.
#'
#' @return A list containing the trained model object, the DFM template,
#'   class levels, and a comprehensive evaluation report.
#'
#' @importFrom stats predict
#' @importFrom caret confusionMatrix
#'
#' @export
pipeline <- function(vect_method,
  model_name,
  df,
  text_column_name,
  sentiment_column_name,
  n_gram = 1) {

cat(paste0("--- Running Pipeline: ", toupper(vect_method), " + ", toupper(model_name), " ---\n"))
  df$sentiment <- as.factor(df[[sentiment_column_name]])
# --- 1. DATA CLEANING & PREP ---
# Remove rows where the cleaned text is empty
initial_rows <- nrow(df)
df[[text_column_name]][df[[text_column_name]] == ""] <- NA
df <- df[!is.na(df[[text_column_name]]), ]
rows_dropped <- initial_rows - nrow(df)
if (rows_dropped > 0) {
cat(paste0("WARNING: Dropped ", rows_dropped, " rows due to missing values in '", text_column_name, "'\n"))
}


# --- 2. TRAIN/TEST SPLIT ---
set.seed(42)
train_indices <- sample(1:nrow(df), size = 0.8 * nrow(df))
data_train <- df[train_indices, ]
data_test <- df[-train_indices, ]
cat(paste0("Data split: ", nrow(data_train), " training rows, ", nrow(data_test), " test rows.\n"))

# --- 3. VECTORIZATION ---
# This now operates on the pre-cleaned text column
X_train <- NULL
X_test <- NULL

if (vect_method %in% c("bow", "tf", "tfidf")) {
cat(paste("Vectorizing text using:", toupper(vect_method), "...\n"))

X_train <- BOW_train(data_train[[text_column_name]],
      weighting_scheme = vect_method,
      ngram_size = n_gram)
X_test <- BOW_test(data_test[[text_column_name]],
    X_train,
    weighting_scheme = vect_method,
    ngram_size = n_gram)

} else {
stop(paste("Vectorizer '", vect_method, "' is not supported!"))
}

# Prepare the response variables (y)
y_train <- data_train$sentiment
y_test <- data_test$sentiment

# --- 4. MODEL TRAINING & PREDICTION ---
model_results <- NULL

if (model_name == "logit") {
model_results <- logit_model(X_train, y_train, X_test)
} else if (model_name == "rf") {
model_results <- rf_model(X_train, y_train, X_test)
} else if (model_name == "xgb") {
model_results <- xgb_model(X_train, y_train, X_test)
} else {
stop(paste("Model '", model_name, "' is not supported!"))
}

# --- 5. EVALUATION ---
cat("\n--- Evaluating Model Performance ---\n")
predictions <- model_results$pred
predictions_factor <- factor(predictions, levels = levels(y_test))

evaluation <- caret::confusionMatrix(
data = predictions_factor,
reference = y_test,
mode = "everything"
)
print(evaluation)

# --- 6. RETURN FINAL RESULTS ---
final_output <- list(
trained_model = model_results$model,
dfm_template = X_train,
class_levels = levels(y_test),
ngram_size_used = n_gram,
vectorize_test_function = BOW_test,
evaluation_report = evaluation
)

return(final_output)
}
