#' Preprocess a Vector of Text Documents
#'
#' This function provides a comprehensive and configurable pipeline for cleaning
#' raw text data. It handles a variety of common preprocessing steps including
#' removing URLs and HTML, lowercasing, stopword removal, and lemmatization.
#'
#' @param doc_vector A character vector where each element is a document.
#' @param remove_brackets A logical value indicating whether to remove text in square brackets.
#' @param remove_urls A logical value indicating whether to remove URLs and email addresses.
#' @param remove_html A logical value indicating whether to remove HTML tags.
#' @param remove_nums A logical value indicating whether to remove numbers.
#' @param remove_emojis_flag A logical value indicating whether to remove common emojis.
#' @param to_lowercase A logical value indicating whether to convert text to lowercase.
#' @param remove_punct A logical value indicating whether to remove punctuation.
#' @param remove_stop_words A logical value indicating whether to remove English stopwords.
#' @param lemmatize A logical value indicating whether to lemmatize words to their dictionary form.
#'
#' @return A character vector of the cleaned and preprocessed text.
#'
#' @importFrom stringr str_replace_all str_remove_all str_squish
#' @importFrom quanteda tokens tokens_tolower tokens_select stopwords
#' @importFrom textstem lemmatize_strings
#' @importFrom magrittr %>%
#'
#' @export
pre_process <- function(doc_vector,
  remove_brackets = TRUE,
  remove_urls = TRUE,
  remove_html = TRUE,
  remove_nums = TRUE, # Changed default for consistency
  remove_emojis_flag = TRUE, # Changed default
  to_lowercase = TRUE,
  remove_punct = TRUE,
  remove_stop_words = TRUE,
  lemmatize = TRUE) {

# --- Stage 1: String-level cleaning ---
if (remove_brackets) {
doc_vector <- stringr::str_replace_all(doc_vector, "\\[[^\\]]*\\]", "")
}
if (remove_urls) {
doc_vector <- stringr::str_replace_all(doc_vector, "http\\S+|www\\S+|https\\S+|\\S*@\\S*\\s?", "")
}
if (remove_html) {
doc_vector <- stringr::str_replace_all(doc_vector, "<.*?>", "")
}
if (remove_emojis_flag) {
doc_vector <- stringr::str_remove_all(doc_vector, "[\\U0001F600-\\U0001F64F]|[\\U0001F300-\\U0001F5FF]|[\\U0001F680-\\U0001F6FF]|[\\U0001F1E0-\\U0001F1FF]")
}
doc_vector <- stringr::str_squish(doc_vector)

# --- Stage 2: Tokenization and token-level operations ---
# Note: The original function had a complex de-tokenize/re-tokenize step for lemmatization.
# A more efficient, standard workflow is to perform all string operations first.
if (to_lowercase) {
doc_vector <- base::tolower(doc_vector)
}
if (lemmatize) {
doc_vector <- textstem::lemmatize_strings(doc_vector)
}

# Now, tokenize and remove stopwords/punctuation/numbers
toks <- quanteda::tokens(doc_vector,
remove_punct = remove_punct,
remove_numbers = remove_nums)

if (remove_stop_words) {
toks <- quanteda::tokens_select(toks, pattern = quanteda::stopwords("en"), selection = "remove")
}

# --- Final step: Convert tokens back to a single string ---
processed_texts <- sapply(as.list(toks), paste, collapse = " ")
names(processed_texts) <- NULL

return(processed_texts)
}
