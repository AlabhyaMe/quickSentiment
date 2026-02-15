## Resubmission
This is a resubmission. I have addressed the feedback from Benjamin Altmann:
* Expanded "TF-IDF" to "Term Frequency-Inverse Document Frequency" in DESCRIPTION.
* Replaced all instances of T/F with TRUE/FALSE in function arguments and examples.
* Replaced \dontrun{} with \donttest{} in examples.
* Replaced print()/cat() with message() in R/logit.R and R/prediction.R to allow suppression.
* Removed set.seed() from within functions in R/pipeline.R.

## Test environments
* local Windows 11 x64, R 4.4.1
* win-builder (devel and release)

## R CMD check results
0 ERRORs | 0 WARNINGs | 0 NOTEs