This is a submission of version 0.3.3


## Test environments
* local Windows install, R 4.5.3

## R CMD check results
0 errors | 0 warnings | 0 note



## Major package updates in 0.3.3

- `evaluate_performance()` minor updates - Added automatic downsampling to evaluate_performance() for datasets over 1,000 rows, optimizing memory and plotting speed while strictly preserving optimal threshold coordinates.
- `evaluate_performance()` return a decile summary of performance metrics instead of a full data frame, making it easier to quickly assess model performance at key thresholds.
