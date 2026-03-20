This is a submission of version 0.3.2 


## Test environments
* local Windows install, R 4.5.3

## R CMD check results
0 errors | 0 warnings | 0 note



## Major package updates in 0.3.1:
- `pipeline()` no longer computes ROC or returns ROC guidelines
- `predict_sentiment()` no longer inherits a threshold directly from the pipeline. It now defaults to a standard 0.5 classification threshold, while still allowing users to manually pass the custom cutoff they feel most comfortable with.
- `evaluate_performance()` is now available. It allows user to calculate detailed metrics (including AUC, Precision, and Recall) targeted at any specific factor level
-  evaluate_performance() engine leverages R's S3 object-oriented system
-  evaluate_performance() result can be called in R's plot() function to instantly generate high-quality ROC and Precision-Recall curves
