Rent the Runway Recommendation Engine
-
To view the code used for this project, please see ![Recommendations.py](https://github.com/befowle/Recommend_The_Runway/blob/master/Recommendations.py)

  - ![Data Source:](https://cseweb.ucsd.edu/~jmcauley/datasets.html)
    - Decomposing fit semantics for product size recommendation in metric spaces, Rishabh Misra, Mengting Wan, Julian McAuley, RecSys, 2018

Objective
-
- To predict item ratings
- To create content-based item recommendation engine

Data Processing
-
- Drop NAN values, remove strings from weight and convert to integer, remove spaces from column names, drop extraneous columns

- Engineer generalized categories for item types (from 68 categories to 7)

- Create separate dataframes for user data features and item features



EDA
-

K-Nearest Neighbors Baseline Model
-

Rating Prediction: Singular Value Decomposition
-

Content-Based Recommendation Engine
-

