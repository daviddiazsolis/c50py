import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("""# Comprehensive Guide to C5.0 Decision Trees with `c50py`

This tutorial provides a deep dive into **`c50py`**, a modern Python implementation of Quinlan's C5.0 algorithm.

We will cover:
1.  **Why C5.0?** Key advantages over standard CART trees (scikit-learn).
2.  **Native Categorical Support**: How `c50py` handles high-cardinality features and **automatically merges categories**.
3.  **Robustness**: Handling missing values without imputation.
4.  **Interpretability**: Extracting and tracing rules.
5.  **Boosting**: Improving performance with C5.0-style boosting.
6.  **Benchmarking**: Comparing performance against scikit-learn.
"""),

    nbf.v4.new_markdown_cell("""## 1. Setup and Installation

First, ensure `c50py` is installed.
"""),

    nbf.v4.new_code_cell("""!pip install c50py graphviz pandas scikit-learn matplotlib"""),

    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from c50py import C5Classifier, C5Regressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import time

# Set random seed for reproducibility
np.random.seed(42)"""),

    nbf.v4.new_markdown_cell("""## 2. The Power of Native Categorical Support

One of the strongest features of C5.0 is its ability to handle categorical variables **natively**. 

Standard CART implementations (like scikit-learn's) require One-Hot Encoding (OHE). For a feature with $K$ categories, OHE creates $K$ binary columns. This leads to:
*   **Sparse Data**: Inefficient memory usage.
*   **Deep Trees**: The tree must make many splits (Is it 'A'? No. Is it 'B'? No...) to isolate a group.
*   **Loss of Context**: The relationship between categories is lost.

**C5.0**, on the other hand, splits categorical features into **subsets**. A single node can split a feature like `Color` into `{Red, Blue}` vs `{Green, Yellow}`. This is much more powerful and interpretable.

### Demonstration: High Cardinality Feature
Let's create a synthetic dataset with a high-cardinality categorical feature to see this in action.
"""),

    nbf.v4.new_code_cell("""# Generate synthetic data
n_samples = 1000
# Feature 1: "City" (High Cardinality - 20 categories)
cities = [f"City_{i}" for i in range(20)]
# Feature 2: "Age" (Numeric)
ages = np.random.randint(18, 70, size=n_samples)

# Assign target based on groups of cities
# Group A: City_0 to City_9 -> High probability of Class 1
# Group B: City_10 to City_19 -> High probability of Class 0
X_cat = np.random.choice(cities, size=n_samples)
y = []
for city, age in zip(X_cat, ages):
    city_idx = int(city.split('_')[1])
    prob = 0.8 if city_idx < 10 else 0.2
    # Add some noise/interaction with age
    if age > 50: prob += 0.1
    y.append(1 if np.random.rand() < prob else 0)

df_syn = pd.DataFrame({'City': X_cat, 'Age': ages})
y_syn = np.array(y)

print("Data Sample:")
print(df_syn.head())
print(f"Unique Cities: {df_syn['City'].nunique()}")"""),

    nbf.v4.new_markdown_cell("""### Training C5.0 with Native Categoricals

We simply pass the dataframe. We can specify `categorical_features` indices or names.
"""),

    nbf.v4.new_code_cell("""# Initialize C5.0
# We tell it that 'City' is categorical.
# infer_categorical=True can also detect object columns automatically.
clf_c5 = C5Classifier(feature_names=list(df_syn.columns), categorical_features=["City"])

t0 = time.time()
clf_c5.fit(df_syn.values, y_syn)
print(f"C5.0 Training Time: {time.time() - t0:.4f}s")

# Visualize the tree structure
# Notice how it groups cities!
clf_c5.print_tree()"""),

    nbf.v4.new_markdown_cell("""**Observation**: Look at the output above. You should see a split like:
`if City in {City_0, City_1, ...}:`
This single node captures the logic that would take *many* nodes in a standard CART tree.

### Comparison with Scikit-Learn (One-Hot Encoding)
Now let's see what scikit-learn does with this data. We must One-Hot Encode first.
"""),

    nbf.v4.new_code_cell("""# One-Hot Encoding for sklearn
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_ohe = enc.fit_transform(df_syn[['City']])
X_num = df_syn[['Age']].values
X_sklearn = np.hstack([X_ohe, X_num])

clf_sk = DecisionTreeClassifier(random_state=42, max_depth=5) # Limit depth to keep it readable
clf_sk.fit(X_sklearn, y_syn)

# Let's look at the depth and node count
print(f"Sklearn Tree Depth: {clf_sk.get_depth()}")
print(f"Sklearn Node Count: {clf_sk.get_n_leaves()}")

# It's much harder to read this tree because 'City' is split into 20 binary features.
"""),

    nbf.v4.new_markdown_cell("""## 3. Missing Value Handling

Real-world data is messy. `c50py` handles missing values (`NaN` or `None`) natively using **fractional case propagation**, the same strategy as the original C5.0.

*   **Training**: If a value is missing at a split, the instance is sent down **both** branches with a weight proportional to the probability of that branch.
*   **Prediction**: The prediction is a weighted average of the results from both branches.

This avoids the need for arbitrary imputation (like filling with mean/median) which can distort data.
"""),

    nbf.v4.new_code_cell("""# Introduce missing values
df_missing = df_syn.copy()
# Randomly drop 20% of 'Age'
mask = np.random.rand(n_samples) < 0.2
df_missing.loc[mask, 'Age'] = np.nan

print(f"Missing values in Age: {df_missing['Age'].isna().sum()}")

# C5.0 handles this automatically
clf_miss = C5Classifier(feature_names=["City", "Age"], categorical_features=["City"])
clf_miss.fit(df_missing.values, y_syn)

print("Training successful with missing values!")
# We can even trace a prediction for a sample with missing data
sample_missing = df_missing.iloc[np.where(mask)[0][0]]
print(f"Sample with missing Age: \\n{sample_missing}")
print(f"Prediction: {clf_miss.predict([sample_missing.values])[0]}")"""),

    nbf.v4.new_markdown_cell("""## 4. Interpretability: Rules and Graphviz

Decision trees are loved for interpretability. `c50py` provides tools to make this even better.

### Export to Graphviz
"""),

    nbf.v4.new_code_cell("""import graphviz

# Export the tree we trained earlier
dot_data = clf_c5.export_graphviz(feature_names=["City", "Age"], class_names=["Class 0", "Class 1"], format="dot")
graph = graphviz.Source(dot_data)
graph
# If running locally, you can use graph.render("tree") to save a PDF/PNG"""),

    nbf.v4.new_markdown_cell("""### Extracting Rules
Sometimes a list of rules is easier to read than a diagram.
"""),

    nbf.v4.new_code_cell("""rules = clf_c5.export_rules(feature_names=["City", "Age"], class_names=["Class 0", "Class 1"])
for r in rules[:5]:
    print(r)"""),

    nbf.v4.new_markdown_cell("""### Rule Tracing
You can ask the model *why* it made a specific prediction for a specific sample.
"""),

    nbf.v4.new_code_cell("""# Trace the first sample
sample = df_syn.iloc[0].values
trace = clf_c5.predict_rule([sample], feature_names=["City", "Age"])
print(f"Sample: {sample}")
print(f"Reasoning: {trace[0]}")"""),

    nbf.v4.new_markdown_cell("""## 5. Boosting

C5.0 is famous for its boosting implementation (similar to Adaboost). You can enable this simply by setting `trials > 1`.
"""),

    nbf.v4.new_code_cell("""# Train a boosted ensemble with 10 trees
clf_boost = C5Classifier(trials=10, feature_names=["City", "Age"], categorical_features=["City"])
clf_boost.fit(df_syn.values, y_syn)

print(f"Boosted Ensemble Size: {len(clf_boost.ensemble_)} trees")
print(f"Accuracy: {clf_boost.score(df_syn.values, y_syn):.4f}")"""),

    nbf.v4.new_markdown_cell("""## 6. Benchmark: Titanic Dataset

Let's put it all together on a real dataset: Titanic. We will compare `c50py` vs `sklearn`.
"""),

    nbf.v4.new_code_cell("""# Load Titanic Data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df_titanic = pd.read_csv(url)

# Preprocessing
df_titanic = df_titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
df_titanic['Age'] = df_titanic['Age'].fillna(df_titanic['Age'].median()) # Fill numeric for sklearn
df_titanic['Embarked'] = df_titanic['Embarked'].fillna(df_titanic['Embarked'].mode()[0])
df_titanic = df_titanic.dropna()

X = df_titanic.drop(columns=['Survived'])
y = df_titanic['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Sklearn (Requires OHE) ---
categorical_cols = ['Sex', 'Embarked']
numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Simple manual OHE for demonstration
X_train_ohe = pd.get_dummies(X_train, columns=categorical_cols)
X_test_ohe = pd.get_dummies(X_test, columns=categorical_cols)
# Align columns
X_train_ohe, X_test_ohe = X_train_ohe.align(X_test_ohe, join='left', axis=1, fill_value=0)

clf_sk = DecisionTreeClassifier(max_depth=5, random_state=42)
t0 = time.time()
clf_sk.fit(X_train_ohe, y_train)
sk_time = time.time() - t0
sk_acc = clf_sk.score(X_test_ohe, y_test)

# --- C5.0 (Native) ---
# We pass the original dataframe (numpy array of objects)
# We need to specify which columns are categorical
cat_features = ['Sex', 'Embarked']
# Note: Pclass is numeric in sklearn but could be categorical. Let's keep it numeric for parity.

clf_c5 = C5Classifier(feature_names=list(X.columns), categorical_features=cat_features)
t0 = time.time()
clf_c5.fit(X_train.values, y_train)
c5_time = time.time() - t0
c5_acc = clf_c5.score(X_test.values, y_test)

print("--- Results ---")
print(f"Sklearn (CART) | Accuracy: {sk_acc:.4f} | Time: {sk_time:.4f}s")
print(f"c50py (C5.0)   | Accuracy: {c5_acc:.4f} | Time: {c5_time:.4f}s")

# Let's try Boosting
clf_c5_boost = C5Classifier(trials=10, feature_names=list(X.columns), categorical_features=cat_features)
clf_c5_boost.fit(X_train.values, y_train)
c5_boost_acc = clf_c5_boost.score(X_test.values, y_test)
print(f"c50py (Boost)  | Accuracy: {c5_boost_acc:.4f}")
"""),

    nbf.v4.new_markdown_cell("""## Conclusion

`c50py` offers a powerful alternative to standard decision trees in Python, bringing:
1.  **Cleaner Trees**: Thanks to native categorical grouping.
2.  **Robustness**: Built-in missing value handling.
3.  **Performance**: Competitive accuracy, often superior with boosting.
4.  **Insight**: Easy-to-read rules and graphs.

Give it a try on your next tabular dataset!
""")
]

with open('examples/c50py_comprehensive_tutorial.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully!")
