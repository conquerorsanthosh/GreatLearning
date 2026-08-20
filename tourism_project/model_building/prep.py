import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(RAW_PATH)

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# Check for duplicates
print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Check for null values — treat if any are found
print("\nNull values per column:")
nulls = df.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() > 0 else "None found.")
# No nulls in this dataset, so no imputation needed here.

# df.describe() — numerical and categorical — find anomalies (min/max)
print("\nNumerical summary:")
print(df.describe())

print("\nCategorical summary:")
print(df.describe(include="object"))

# Gender column rectification: "Fe Male" -> "Female"
print("\nGender before:", df["Gender"].unique())
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
print("Gender after:", df["Gender"].unique())

# Marital Status rectification: merge "Unmarried" into "Single"
# Note: purchase rates differ between these (Single 36% vs Unmarried 24%
# in the raw data) — merging trades that signal away, but keeping to spec.
print("\nMaritalStatus before:", df["MaritalStatus"].unique())
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
print("MaritalStatus after:", df["MaritalStatus"].unique())

# Basic anomaly cleanup — DurationOfPitch and NumberOfTrips have implausible
# max values (127 minutes, 22 trips/year) relative to the rest of the
# distribution. Treat as data-entry errors: set to NaN, then fill with median.
for col, cap in [("DurationOfPitch", 60), ("NumberOfTrips", 20)]:
    n_flagged = (df[col] > cap).sum()
    print(f"\n{col}: {n_flagged} rows above {cap}, will be treated as anomalies.")
    df.loc[df[col] > cap, col] = df[col].median()

# Scan remaining categorical columns for anything else unusual
print("\nRemaining categorical value checks:")
for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {df[col].unique()}")

print("\nFinal shape after cleaning:", df.shape)
print("\nFirst 5 rows after cleaning:")
print(df.head())


X = df.drop(columns=["ProdTaken", "CustomerID", "Unnamed: 0"], errors="ignore")
y = df["ProdTaken"]

# stratify=y keeps the imbalanced purchase ratio (~19% positive) consistent
# across both splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Xtrain:", Xtrain.shape, "| Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape, "| ytest:", ytest.shape)
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
