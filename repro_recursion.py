import pandas as pd
import numpy as np
import time
from c50py import C5Classifier

# Generate synthetic data
n_samples = 1000
np.random.seed(42)
# Feature 1: "City" (High Cardinality - 20 categories)
cities = [f"City_{i}" for i in range(20)]
# Feature 2: "Age" (Numeric)
ages = np.random.randint(18, 70, size=n_samples)

# Assign target based on groups of cities
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

# Initialize C5.0
clf_c5 = C5Classifier(feature_names=list(df_syn.columns), categorical_features=["City"])

print("Starting fit...")
t0 = time.time()
try:
    clf_c5.fit(df_syn.values, y_syn)
    print(f"C5.0 Training Time: {time.time() - t0:.4f}s")
    clf_c5.print_tree()
except RecursionError:
    print("Caught RecursionError!")
except Exception as e:
    print(f"Caught unexpected error: {e}")
