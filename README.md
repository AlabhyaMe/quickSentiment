#Get started with the installation
```bash
# install.packages("devtools")
devtools::install_github("AlabhyaMe/quickSentiment")
```

quickSentiment: A Fast and Flexible Pipeline for Text Classification in R
quickSentiment is an R package designed to streamline the process of text classification. It provides a complete, end-to-end workflow from text cleaning to model training, evaluation, and prediction. Built on a modular architecture, it allows users to easily experiment with different vectorization methods and high-performance machine learning models.

Key Features
Modular Pipeline: A single, powerful pipeline() function to run the entire training and evaluation process.

Multiple Vectorization Methods: Includes support for Bag-of-Words, Term Frequency (TF), and TF-IDF, with easy n-gram integration.

High-Performance Models: Comes with built-in support for three powerful and popular classification models:

Logistic Regression (via glmnet for speed and regularization)

Random Forest (via the high-performance ranger package)

XGBoost (via the industry-standard xgboost package)

Reproducible Predictions: The pipeline returns a self-contained artifact object, bundling the trained model and all necessary components to ensure that predictions on new data are consistent and reliable.

Installation
You can install the development version of quickSentiment from GitHub with:

```bash
# install.packages("devtools")
devtools::install_github("AlabhyaMe/quickSentiment")
```
Core Workflow: The Three Main Functions
The quickSentiment workflow is designed to be logical and flexible, centered around three key functions.

1. pre_process_final(): Cleaning Your Text
The first step is to clean your raw text data. This function is kept separate from the main pipeline to allow you to use the same cleaned text for multiple different tasks in the future (e.g., classification, topic modeling).

It handles a variety of common cleaning steps, including lowercasing, lemmatization, and removing URLs, HTML, and stopwords.

Usage:
```bash
library(quickSentiment)
library(readr)

# Load your data
my_data <- read_csv("path/to/your/data.csv")

# Create a new column of cleaned text
my_data$cleaned_text <- pre_process_final(my_data$reviewText)
```

2. pipeline(): The Main Engine for Training
This is the core function of the package. It takes your preprocessed data and handles everything else: splitting the data, vectorizing the text, training a model, and evaluating its performance.

You can easily swap out vectorization methods or models just by changing a string argument.

Usage:
```bash
# Train a TF-IDF Logistic Regression model with bigrams
pipeline_artifacts <- pipeline(
  df = my_data,
  text_column_name = "cleaned_text",
  sentiment_column_name = "rating", # The original column with ratings or labels
  vect_method = "tfidf",
  model_name = "logit",
  n_gram = 2
)
```
The function will print a detailed evaluation report from caret and return a list containing all the necessary "artifacts" for prediction.

3. prediction(): Scoring New, Unseen Data
Once your pipeline has run, you can use the pipeline_artifacts object it returned to make predictions on new data. This function is generic and automatically adapts to the type of model you trained (glmnet, ranger, or xgboost).

Usage:
```bash
# Create a vector of new, raw text to predict on
new_reviews <- c(
  "This was the best product I have ever purchased!",
  "A complete waste of money, I would not recommend this."
)

# The prediction function uses the artifacts to ensure a consistent workflow
final_predictions <- prediction(
  pipeline_object = pipeline_artifacts,
  new_raw_text_vector = new_reviews
)

print(final_predictions)
```
Behind the Hood: How it Works
quickSentiment is designed to be both user-friendly and powerful. This is achieved through a few key design choices:

The "Model Contract": Each model function (logit, rf_model, xgboost_model) follows a strict contract: it must return a list containing the predictions (pred) and the trained model object (model). This allows the main pipeline to treat them interchangeably, making the system highly modular and easy to extend with new models.

High-Performance Backends: The package doesn't reinvent the wheel. It stands on the shoulders of giants by using the best-in-class, high-performance packages for each task:

quanteda for fast and memory-efficient text processing.

glmnet for its incredibly fast, pathwise algorithm for regularized regression.

ranger for a multi-threaded, C++ implementation of Random Forest.

xgboost for its industry-leading speed and predictive accuracy.

Self-Contained Artifacts: The pipeline() function returns a single R object that contains everything needed for reproducible predictions: the trained model, the DFM vocabulary template, the preprocessing function, and the n-gram settings. This "all-in-one" object prevents common errors and ensures that new data is always processed in the exact same way as the training data.

Future Development
This package is under active development. Future plans include:

Adding more vectorization methods, such as pre-trained embeddings (Word2Vec, GloVe).

Integrating topic modeling capabilities.

Adding more machine learning models.

License
This package is licensed under the MIT License.

## 📚 Dataset Citation

The demo uses publicly available training data from:

> Madhav Kumar Choudhary. *Sentiment Prediction on Movie Reviews*. Kaggle.  
> [https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews](https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews)  
> Accessed on: 2025- 07-15

If you use this dataset in your own work, please cite the original creator as per Kaggle's [Terms of Use](https://www.kaggle.com/terms).


