# quickSentiment 0.3.7

## Bug Fixes
* **TF-IDF Weighting Consistency:** Fixed a mathematical mismatch in `BOW_train` and `BOW_test`. Previously, `BOW_train` utilized `quanteda::dfm_tfidf` (which defaults to standard inverse document frequency), while `BOW_test` applied a manual smoothed IDF formula (`log((N+1)/(df+1)) + 1`). The pipeline now correctly and consistently applies the smoothed scikit-learn style IDF weights across both training and testing datasets using `quanteda::dfm_weight()`.

## Minor Improvements
* **Dependency Reduction:** Removed the `@importFrom` dependency on the `Matrix` package, as the updated IDF weighting logic now operates entirely natively within the `quanteda` ecosystem.
* Fixed a minor missing comma syntax error in the `switch` statement of `BOW_test`.
