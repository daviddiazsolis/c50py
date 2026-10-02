# Benchmark results (c50py 0.5.0, 23 OpenML datasets, 5-fold CV)

Leaves are rules for the rulesets. Boosted models and HGB have no single size.

## Mean over datasets

|                                          |   accuracy |   leaves / rules |   conditions/rule |   distinct vars/rule |   columns used |   fit time (s) |
|:-----------------------------------------|-----------:|-----------------:|------------------:|---------------------:|---------------:|---------------:|
| c50py tree (C5.0 defaults)               |      0.86  |           43.861 |             6.476 |                5.396 |         12.235 |          0.704 |
| c50py tree, cf tuned by CV               |      0.862 |           30.07  |             5.811 |                4.943 |         10.713 |          6.342 |
| c50py tree, winnow=True                  |      0.859 |           33.739 |             5.842 |                4.833 |          9.609 |          1.294 |
| c50py ruleset (C5RulesClassifier)        |      0.863 |           21.313 |             3.262 |                3.164 |         11.617 |          1.019 |
| c50py boosting, trials=10                |      0.879 |          nan     |           nan     |              nan     |        nan     |          7.573 |
| CART + one-hot (sklearn defaults)        |      0.848 |          152.113 |             8.737 |                6.455 |         15.852 |          0.025 |
| CART + one-hot, pruning tuned by CV      |      0.866 |           32.009 |             5.282 |                4.369 |          9.522 |         10.605 |
| HistGradientBoosting (native categories) |      0.877 |          nan     |           nan     |              nan     |        nan     |          0.514 |
| C5.0 original (R), tree                  |      0.862 |           33.191 |           nan     |              nan     |         10.511 |          0.092 |
| C5.0 original (R), rules=TRUE            |      0.868 |           17.887 |           nan     |              nan     |          9.83  |          0.097 |
| C5.0 original (R), trials=10             |      0.878 |          nan     |           nan     |              nan     |        nan     |          0.182 |


## Paired comparisons across datasets (Wilcoxon signed-rank)

Wins/ties/losses count datasets where the first model is more accurate by more than 0.05 points, within 0.05 points, or less accurate. The leaves ratio is the geometric mean of (leaves of the first) / (leaves of the second); below 1 means smaller.

| comparison                  |   mean accuracy difference | wins/ties/losses   |   p (accuracy) |   leaves ratio |   p (leaves) |
|:----------------------------|---------------------------:|:-------------------|---------------:|---------------:|-------------:|
| c50py vs C5.0_R             |                    -0.0018 | 9/4/10             |          0.306 |           1.24 |       0.0298 |
| c50py_rules vs C5.0_R_rules |                    -0.0048 | 8/3/12             |          0.073 |           1.15 |       0.0051 |
| c50py_boost vs C5.0_R_boost |                     0.0007 | 11/4/8             |          0.715 |                |              |
| c50py vs cart               |                     0.0119 | 15/1/7             |          0.019 |           0.36 |       0      |
| c50py_cv vs cart_cv         |                    -0.0037 | 9/1/13             |          0.661 |           0.99 |       0.7998 |
| c50py vs cart_cv            |                    -0.0058 | 9/1/13             |          0.548 |           1.31 |       0.1485 |
| c50py_rules vs cart_cv      |                    -0.0025 | 9/2/12             |          0.783 |           0.72 |       0.0327 |
| c50py_rules vs c50py        |                     0.0033 | 14/2/7             |          0.131 |           0.55 |       0      |
| c50py_winnow vs c50py       |                    -0.0008 | 11/5/7             |          0.711 |           0.77 |       0.0007 |
| c50py_cv vs c50py           |                     0.0021 | 11/7/5             |          0.287 |           0.76 |       0.0064 |
| c50py vs hgb                |                    -0.0176 | 4/3/16             |          0.007 |                |              |
| c50py_rules vs hgb          |                    -0.0142 | 6/1/16             |          0.022 |                |              |
| c50py_boost vs hgb          |                     0.0012 | 8/1/14             |          0.426 |                |              |


## By type of data: mean accuracy

Datasets per type: {'mixed': 14, 'categorical only': 9}

| model                                    |   categorical only |   mixed |
|:-----------------------------------------|-------------------:|--------:|
| c50py tree (C5.0 defaults)               |              0.939 |   0.809 |
| c50py tree, cf tuned by CV               |              0.942 |   0.81  |
| c50py tree, winnow=True                  |              0.938 |   0.808 |
| c50py ruleset (C5RulesClassifier)        |              0.942 |   0.812 |
| c50py boosting, trials=10                |              0.943 |   0.837 |
| CART + one-hot (sklearn defaults)        |              0.928 |   0.797 |
| CART + one-hot, pruning tuned by CV      |              0.934 |   0.822 |
| HistGradientBoosting (native categories) |              0.938 |   0.839 |
| C5.0 original (R), tree                  |              0.937 |   0.813 |
| C5.0 original (R), rules=TRUE            |              0.945 |   0.819 |
| C5.0 original (R), trials=10             |              0.945 |   0.835 |


## By type of data: mean leaves / rules

| model                               |   categorical only |   mixed |
|:------------------------------------|-------------------:|--------:|
| c50py tree (C5.0 defaults)          |               40   |    46.3 |
| c50py tree, cf tuned by CV          |               38.4 |    24.7 |
| c50py tree, winnow=True             |               38.8 |    30.5 |
| c50py ruleset (C5RulesClassifier)   |               24.2 |    19.4 |
| CART + one-hot (sklearn defaults)   |               78.4 |   199.5 |
| CART + one-hot, pruning tuned by CV |               51.5 |    19.5 |
| C5.0 original (R), tree             |               41.4 |    27.9 |
| C5.0 original (R), rules=TRUE       |               25   |    13.3 |


## Per dataset: accuracy

| dataset         |     n |   p |   p_cat | type             |   c50py |   c50py_cv |   c50py_winnow |   c50py_rules |   c50py_boost |   cart |   cart_cv |   hgb |   C5.0_R |   C5.0_R_rules |   C5.0_R_boost |
|:----------------|------:|----:|--------:|:-----------------|--------:|-----------:|---------------:|--------------:|--------------:|-------:|----------:|------:|---------:|---------------:|---------------:|
| Australian      |   690 |  14 |       8 | mixed            |   0.868 |      0.87  |          0.864 |         0.872 |         0.868 |  0.819 |     0.854 | 0.864 |    0.864 |          0.858 |          0.87  |
| adult           | 10000 |  14 |       8 | mixed            |   0.846 |      0.852 |          0.848 |         0.853 |         0.844 |  0.812 |     0.849 | 0.863 |    0.856 |          0.856 |          0.862 |
| anneal          |   898 |  18 |      12 | mixed            |   0.915 |      0.923 |          0.927 |         0.937 |         0.958 |  0.993 |     0.992 | 0.99  |    0.935 |          0.954 |          0.954 |
| bank-marketing  | 10000 |  16 |       9 | mixed            |   0.896 |      0.902 |          0.897 |         0.901 |         0.902 |  0.871 |     0.903 | 0.905 |    0.903 |          0.904 |          0.903 |
| breast-cancer   |   286 |   9 |       9 | categorical only |   0.724 |      0.755 |          0.731 |         0.724 |         0.71  |  0.661 |     0.703 | 0.685 |    0.731 |          0.738 |          0.727 |
| car             |  1728 |   6 |       6 | categorical only |   0.973 |      0.974 |          0.973 |         0.981 |         0.981 |  0.966 |     0.966 | 0.995 |    0.969 |          0.977 |          0.979 |
| churn           |  5000 |  20 |       4 | mixed            |   0.944 |      0.945 |          0.945 |         0.948 |         0.955 |  0.914 |     0.94  | 0.96  |    0.945 |          0.945 |          0.953 |
| colic           |   368 |  26 |      19 | mixed            |   0.834 |      0.84  |          0.851 |         0.826 |         0.872 |  0.807 |     0.867 | 0.861 |    0.851 |          0.845 |          0.862 |
| credit-approval |   690 |  15 |       9 | mixed            |   0.843 |      0.835 |          0.846 |         0.836 |         0.854 |  0.823 |     0.865 | 0.859 |    0.861 |          0.861 |          0.864 |
| credit-g        |  1000 |  20 |      13 | mixed            |   0.723 |      0.736 |          0.738 |         0.729 |         0.73  |  0.684 |     0.717 | 0.766 |    0.737 |          0.743 |          0.75  |
| cylinder-bands  |   540 |  35 |      17 | mixed            |   0.707 |      0.7   |          0.685 |         0.737 |         0.791 |  0.741 |     0.739 | 0.83  |    0.717 |          0.704 |          0.791 |
| dresses-sales   |   500 |  12 |      11 | mixed            |   0.582 |      0.588 |          0.61  |         0.592 |         0.596 |  0.536 |     0.612 | 0.59  |    0.598 |          0.61  |          0.596 |
| eucalyptus      |   736 |  19 |       5 | mixed            |   0.63  |      0.628 |          0.601 |         0.628 |         0.659 |  0.622 |     0.632 | 0.667 |    0.629 |          0.644 |          0.658 |
| hepatitis       |   155 |  19 |      13 | mixed            |   0.787 |      0.787 |          0.794 |         0.794 |         0.852 |  0.774 |     0.794 | 0.794 |    0.781 |          0.813 |          0.832 |
| kr-vs-kp        |  3196 |  36 |      36 | categorical only |   0.993 |      0.993 |          0.992 |         0.994 |         0.994 |  0.995 |     0.994 | 0.993 |    0.993 |          0.994 |          0.994 |
| lymph           |   148 |  18 |      15 | mixed            |   0.764 |      0.75  |          0.723 |         0.737 |         0.852 |  0.77  |     0.757 | 0.804 |    0.722 |          0.743 |          0.805 |
| mushroom        |  8124 |  21 |      21 | categorical only |   1     |      1     |          1     |         0.999 |         1     |  1     |     1     | 1     |    1     |          0.998 |          1     |
| nursery         | 10000 |   8 |       8 | categorical only |   0.99  |      0.991 |          0.99  |         0.992 |         0.998 |  0.995 |     0.995 | 0.919 |    0.99  |          0.992 |          0.996 |
| sick            |  3772 |  27 |      21 | mixed            |   0.989 |      0.989 |          0.988 |         0.985 |         0.988 |  0.989 |     0.991 | 0.989 |    0.988 |          0.983 |          0.987 |
| soybean         |   683 |  35 |      35 | categorical only |   0.921 |      0.921 |          0.906 |         0.917 |         0.93  |  0.915 |     0.911 | 0.941 |    0.918 |          0.93  |          0.928 |
| splice          |  3190 |  60 |      60 | categorical only |   0.937 |      0.945 |          0.943 |         0.939 |         0.95  |  0.921 |     0.947 | 0.964 |    0.937 |          0.939 |          0.951 |
| tic-tac-toe     |   958 |   9 |       9 | categorical only |   0.951 |      0.946 |          0.951 |         0.976 |         0.981 |  0.953 |     0.94  | 0.993 |    0.936 |          0.975 |          0.972 |
| vote            |   435 |  16 |      16 | categorical only |   0.959 |      0.959 |          0.959 |         0.959 |         0.945 |  0.943 |     0.947 | 0.949 |    0.959 |          0.959 |          0.959 |


## Per dataset: leaves / rules

| dataset         |   c50py |   c50py_cv |   c50py_winnow |   c50py_rules |   cart |   cart_cv |   C5.0_R |   C5.0_R_rules |
|:----------------|--------:|-----------:|---------------:|--------------:|-------:|----------:|---------:|---------------:|
| Australian      |    19.6 |       12   |           17   |          10.8 |   75   |       7.2 |     13.2 |            9   |
| adult           |    79   |       27.4 |           54.4 |          26.2 | 1156   |      27.8 |     38.2 |           21   |
| anneal          |    22.8 |       28   |           26.2 |          13   |   15.8 |      15.2 |     31.2 |           12.2 |
| bank-marketing  |   209.2 |       53.4 |           55   |          74.2 |  658.6 |      22.8 |     76.8 |           31.4 |
| breast-cancer   |     7.6 |        4.2 |            5   |           5.6 |   72.4 |       7.6 |      7   |            4.4 |
| car             |    47.8 |       49.8 |           47.8 |          34   |   89.4 |      81.6 |     43.6 |           37.8 |
| churn           |    56.2 |       48.2 |           52.2 |          24.8 |  240.8 |      34.4 |     39   |           20.6 |
| colic           |     6.8 |        7   |            5.8 |           4   |   33.2 |       5.4 |      3   |            3   |
| credit-approval |    20.4 |       14.8 |           15.8 |           9.6 |   72   |       5   |      8   |            7   |
| credit-g        |    64.8 |       27.8 |           55.4 |          24.8 |  153.6 |      27.2 |     52.8 |           18.4 |
| cylinder-bands  |    40.4 |       39.6 |           34.6 |          18.2 |   60.8 |      31   |     41.8 |           13.6 |
| dresses-sales   |    10.8 |        9.8 |            2.2 |           4   |  113.6 |      23.4 |      6.6 |            4.4 |
| eucalyptus      |    80.2 |       44.6 |           78.6 |          38.8 |  143   |      28.4 |     45.6 |           23.8 |
| hepatitis       |     5   |        3   |            2.8 |           4.4 |   17.4 |       3.4 |      6.2 |            4.2 |
| kr-vs-kp        |    26.4 |       26.2 |           24.2 |          13.4 |   48.6 |      37.2 |     26.4 |           13.4 |
| lymph           |    11.6 |        8.8 |            6.4 |           8.2 |   21.2 |      14   |     11.4 |            8.2 |
| mushroom        |     7.8 |        7.8 |            7.6 |           6.4 |   14.2 |      14   |      8   |            6.8 |
| nursery         |   137.2 |      142   |          137.2 |          73.8 |  208.8 |     203.2 |    134.8 |           78.4 |
| sick            |    21.8 |       21.8 |           20.8 |          11.2 |   31.8 |      27.2 |     17.2 |            9.2 |
| soybean         |    38.6 |       39.8 |           37.4 |          30.2 |   64.2 |      44   |     51.4 |           31.4 |
| splice          |    57.2 |       36.2 |           51.8 |          33.6 |  121   |      22.4 |     60.8 |           29.8 |
| tic-tac-toe     |    33.6 |       35.6 |           33.6 |          17.6 |   64   |      46.6 |     36.4 |           20   |
| vote            |     4   |        3.8 |            4.2 |           3.4 |   23.2 |       7.2 |      4   |            3.4 |


## Per dataset: mean distinct columns per rule

| dataset         |   c50py |   c50py_cv |   c50py_rules |   cart |   cart_cv |
|:----------------|--------:|-----------:|--------------:|-------:|----------:|
| Australian      |    5.7  |       5.02 |          3.33 |   6.24 |      2.54 |
| adult           |    6.7  |       5.43 |          3.94 |   9.46 |      5.02 |
| anneal          |    5.4  |       5.75 |          2.18 |   4.96 |      4.94 |
| bank-marketing  |    7.55 |       5.99 |          4.71 |   7.97 |      4.13 |
| breast-cancer   |    3.29 |       2.31 |          2.38 |   6.15 |      2.73 |
| car             |    5.26 |       5.29 |          4.34 |   5.68 |      5.63 |
| churn           |    5.47 |       5.15 |          3.48 |   8.71 |      4.78 |
| colic           |    3.04 |       2.94 |          2    |   4.94 |      2.32 |
| credit-approval |    5.43 |       5.06 |          3.17 |   6.08 |      2.46 |
| credit-g        |    7.39 |       6.08 |          4.7  |   7.25 |      4.29 |
| cylinder-bands  |    8.29 |       8.21 |          3.31 |   7.62 |      6.37 |
| dresses-sales   |    3.05 |       2.5  |          1.69 |   7.51 |      3.22 |
| eucalyptus      |    7.03 |       5.96 |          3.85 |   6.34 |      3.85 |
| hepatitis       |    2.22 |       1.43 |          1.63 |   4.39 |      1.58 |
| kr-vs-kp        |    7.61 |       7.51 |          3.91 |   8.7  |      7.84 |
| lymph           |    4.55 |       3.66 |          2.62 |   4.63 |      3.88 |
| mushroom        |    3.81 |       3.81 |          2.01 |   3.74 |      3.69 |
| nursery         |    6.32 |       6.34 |          4.82 |   6.45 |      6.44 |
| sick            |    4.94 |       4.94 |          2.46 |   4.44 |      4.41 |
| soybean         |    6.03 |       6.12 |          3.19 |   7.78 |      6.53 |
| splice          |    7.31 |       6.57 |          4.41 |   7.95 |      5.1  |
| tic-tac-toe     |    5.5  |       5.55 |          3.24 |   6.57 |      5.92 |
| vote            |    2.22 |       2.08 |          1.42 |   4.9  |      2.82 |
