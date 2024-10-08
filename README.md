# This project is a Competitive Model Comparison(CMC)
Built on top of sklearn's models.

## Models:
> 1. Logistic Regression
> 2. SVM
> 3. KNN
> 4. Decision Tree
> 5. Random Forest
> 6. Extra Trees
> 7. Gradient Boosting
> 8. AdaBoost
>

## Preprocessing:
> 1. Drop nulls
> 2. Encode
> 3. Scale

## Dependancies:

```
import pandas as pd\r
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
```

## Usage:

Importing the modules from the package
>from plumber import mlpipe, mlht

### CMC (pre hyperparameter tuning)
Instantiating the CMC

```
pipe = mlpipe(df=df,target_column=target_column,test_size=0.2,random_state=1,display_analytics=True)
```

Running the CMC

```
pipe.run_pipeline(drop_max_na_col_in=True,drop_threshold_in=0.25)
```

Get feature importance of user model or best model. If a user model has been selected, it will prioritize that.

```
pipe.get_feature_importance()
```

Choose a model of your liking

```
my_model=pipe.user_model_return()
```

Visualize a decision tree

```
pipe.visualize_decision_tree()
```

### Hyperparameter Tuning

Instantiating the hyperparameter tuning object. (cores decides how many CPU cores you provide for cross validation, default cross validation set to 10Fold CV)

```
ht = mlht(model=my_model, pipeline=pipe, cores=2)
```

Run the tuning

```
ht.run_tuning()
```

Get the tuned model for deployment

```
tm = ht.tuned_model
```
