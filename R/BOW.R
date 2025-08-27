#BOW.R
#' Train a Bag-of-Words Model 
#' @param doc A character vector of documents to be processed.
#' @param weighting_scheme A string specifying the weighting to apply. One of
#'   "count" (default), "tf", or "tfidf".
#' @param ngram_size An integer specifying the maximum n-gram size. For example,
#'   `ngram_size = 2` will create unigrams and bigrams. Defaults to 1.
#'
#' @return A quanteda `dfm` object to be used as a template for new data.
#'
#' @importFrom quanteda tokens tokens_ngrams tokens_select dfm dfm_weight dfm_tfidf
#' @importFrom magrittr %>%
#'
#' @export
BOW_train <- function(doc, weighting_scheme = "count", ngram_size=1) {
  cat(paste0("  - Fitting BoW model (", weighting_scheme, ") on training data...\n"))
  
  dfm <- doc %>%
    quanteda::tokens(remove_punct = TRUE, remove_numbers = TRUE) %>%
    quanteda::tokens_ngrams(n = 1:ngram_size) %>%
    quanteda::tokens_select(pattern = quanteda::stopwords("en"), selection = "remove") %>%
    quanteda::dfm() 
  
  # Apply weighting based on the chosen scheme
  if (weighting_scheme == "tf") {
    dfm <- dfm_weight(dfm, scheme = "prop")
  } else if (weighting_scheme == "tfidf") {
    dfm <- dfm_tfidf(dfm)
  }
  # If scheme is "count", we do nothing and return the raw counts.
  
  return(dfm)
}
#' Transform New Text into a Document-Feature Matrix
#'
#' This function takes a character vector of new documents and transforms it
#' into a DFM that has the exact same features as a pre-fitted training DFM,
#' ensuring consistency for prediction.
#'
#' @param doc A character vector of new documents to be processed.
#' @param training_dfm The `dfm` object created by `BOW_train` that will be
#'   used as a template for matching features.
#' @param weighting_scheme A string specifying the weighting to apply. Must be
#'   the same as used in `BOW_train`. One of "count", "tf", or "tfidf".
#' @param ngram_size The maximum n-gram size. Must be the same as used in
#'   `BOW_train`. Defaults to 1.
#'
#' @return A quanteda `dfm` object that is perfectly aligned with the training DFM.
#'
#' @importFrom quanteda tokens tokens_ngrams tokens_select dfm dfm_match featnames dfm_weight dfm_tfidf
#' @importFrom magrittr %>%
#'
#' @export
# The new, more flexible BOW_test function
BOW_test <- function(doc, training_dfm, weighting_scheme = "count",ngram_size=1) {
  cat(paste0("  - Applying BoW transformation (", weighting_scheme, ") to new data...\n"))
  
  # Create a raw DFM and match its features to the training template
  dfm_raw <- doc %>%
    quanteda::tokens(remove_punct = TRUE, remove_numbers = TRUE) %>%
    quanteda::tokens_ngrams(n = 1:ngram_size)%>%
    quanteda::tokens_select(pattern = quanteda::stopwords("en"), selection = "remove") %>%
    quanteda::dfm()
  
  dfm_matched <- quanteda::dfm_match(dfm_raw, features = quanteda::featnames(training_dfm))
  
  # Apply the SAME weighting scheme to the matched test DFM
  if (weighting_scheme == "tf") {
    dfm_final <- dfm_weight(dfm_matched, scheme = "prop")
  } else if (weighting_scheme == "tfidf") {
    dfm_final <- dfm_tfidf(dfm_matched)
  } else {
    dfm_final <- dfm_matched
  }
  
  return(dfm_final)
}
