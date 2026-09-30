# ============================================================
# LIFE EXPECTANCY PREDICTION USING MACHINE LEARNING
# STEP 3 & 4: DATASET UNDERSTANDING AND DATA PREPROCESSING
# ============================================================

# ============================================================
# STEP 3: DATASET LOADING AND INITIAL UNDERSTANDING
# ============================================================

# Explanation:
# In this section, the pandas library is imported to read and
# analyze the CSV dataset. The dataset is then loaded into a
# pandas DataFrame for further processing.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "Life Expectancy Data.csv"

df = pd.read_csv(file_path)

# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

# Explanation:
# The Life Expectancy dataset is stored in the project folder.
# The read_csv() function is used to load the CSV file into
# a pandas DataFrame.

file_path = "Life Expectancy Data.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")


# ------------------------------------------------------------
# 2. DISPLAY FIRST FIVE RECORDS
# ------------------------------------------------------------

# Explanation:
# The head() function displays the first five rows of the dataset.
# This helps us understand the structure and type of information
# present in the dataset.

print("\nFirst 5 rows of the dataset:")
print(df.head())


# ------------------------------------------------------------
# 3. CHECK DATASET SIZE
# ------------------------------------------------------------

# Explanation:
# The shape attribute returns the number of rows and columns.
# This helps us understand the size of the dataset.

print("\nDataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 4. DISPLAY COLUMN NAMES
# ------------------------------------------------------------

# Explanation:
# The column names are displayed to identify all variables
# available in the dataset.

print("\nOriginal column names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 5. CHECK DATA TYPES AND NON-NULL VALUES
# ------------------------------------------------------------

# Explanation:
# The info() function provides information about the dataset,
# including column names, data types, and non-null values.
# This helps us identify numerical and categorical variables
# and understand the presence of missing values.

print("\nDataset information:")
df.info()


# ------------------------------------------------------------
# 6. STATISTICAL SUMMARY
# ------------------------------------------------------------

# Explanation:
# The describe() function provides statistical information
# about numerical variables, including mean, standard deviation,
# minimum, maximum, and quartile values.

print("\nStatistical summary:")
print(df.describe())


# ------------------------------------------------------------
# 7. CHECK MISSING VALUES
# ------------------------------------------------------------

# Explanation:
# The isnull().sum() function calculates the number of missing
# values in each column. This is important because missing data
# needs to be handled before developing machine learning models.

print("\nMissing values before preprocessing:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 8. CHECK DUPLICATE ROWS
# ------------------------------------------------------------

# Explanation:
# The duplicated().sum() function checks whether completely
# duplicated records are present in the dataset.

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# ============================================================
# STEP 4: DATA CLEANING AND PREPROCESSING
# ============================================================

# Explanation:
# Before building machine learning models, the dataset needs
# to be cleaned and prepared. In this step, we clean column
# names, remove rows with missing target values, and handle
# missing values in input features.


# ------------------------------------------------------------
# 9. CLEAN COLUMN NAMES
# ------------------------------------------------------------

# Explanation:
# Some column names contain unnecessary spaces at the beginning
# or end. The strip() function removes these extra spaces and
# makes the column names consistent.

df.columns = df.columns.str.strip()

print("\nCleaned column names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 10. REMOVE ROWS WITH MISSING TARGET VALUES
# ------------------------------------------------------------

# Explanation:
# Life expectancy is the target variable of this project.
# Since the target value is missing for some records, those
# records cannot be directly used for supervised learning.
# Therefore, rows with missing Life expectancy values are removed.

df = df.dropna(subset=["Life expectancy"])

print("\nDataset shape after removing missing target values:")
print(df.shape)


# ------------------------------------------------------------
# 11. IDENTIFY NUMERICAL COLUMNS
# ------------------------------------------------------------

# Explanation:
# Numerical columns contain quantitative values and can be
# processed using numerical imputation techniques.

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns


# ------------------------------------------------------------
# 12. HANDLE MISSING VALUES IN NUMERICAL FEATURES
# ------------------------------------------------------------

# Explanation:
# Missing values in numerical input features are replaced with
# their respective median values. Median imputation is preferred
# because it is less affected by extreme values and outliers
# compared with mean imputation.
#
# The target variable, Life expectancy, is excluded because
# missing target values were already removed in the previous step.

feature_numerical_columns = [
    column for column in numerical_columns
    if column != "Life expectancy"
]

for column in feature_numerical_columns:
    df[column] = df[column].fillna(df[column].median())


# ------------------------------------------------------------
# 13. HANDLE MISSING VALUES IN CATEGORICAL FEATURES
# ------------------------------------------------------------

# Explanation:
# Categorical columns contain text-based values. If any missing
# categorical values are present, they are replaced with the
# most frequently occurring value using the mode.

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


# ------------------------------------------------------------
# 14. VERIFY MISSING VALUES AFTER PREPROCESSING
# ------------------------------------------------------------

# Explanation:
# After handling missing values, we check the dataset again.
# This verifies whether any missing values remain.

print("\nMissing values after preprocessing:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 15. CHECK DUPLICATES AFTER PREPROCESSING
# ------------------------------------------------------------

# Explanation:
# We perform a final duplicate check after preprocessing to
# ensure that duplicate records are not present.

print("\nDuplicate rows after preprocessing:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 16. FINAL DATASET INFORMATION
# ------------------------------------------------------------

# Explanation:
# The final dataset shape is displayed to understand how many
# records and features are available after preprocessing.

print("\nFinal dataset shape:")
print(df.shape)

print("\nData preprocessing completed successfully!")
# ============================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Explanation:
# Exploratory Data Analysis is performed to understand the
# distribution of the target variable and the relationships
# between different features and Life expectancy.
#
# Visualizations help us identify patterns, trends, relationships,
# and possible outliers in the dataset.


# ------------------------------------------------------------
# 17. IMPORT VISUALIZATION LIBRARIES
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# 18. DISTRIBUTION OF LIFE EXPECTANCY
# ------------------------------------------------------------

# Explanation:
# A histogram is used to understand how Life expectancy values
# are distributed across the dataset. It helps us identify the
# central tendency, spread, and shape of the target variable.

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Life expectancy"],
    bins=30,
    kde=True
)

plt.title("Distribution of Life Expectancy")
plt.xlabel("Life Expectancy")
plt.ylabel("Number of Countries/Observations")
plt.tight_layout()

plt.savefig("outputs/life_expectancy_distribution.png")
plt.show()


# ------------------------------------------------------------
# 19. CORRELATION HEATMAP
# ------------------------------------------------------------

# Explanation:
# A correlation heatmap is used to understand the strength and
# direction of relationships between numerical variables.
#
# Correlation values range from -1 to +1.
# Values closer to +1 indicate a strong positive relationship,
# while values closer to -1 indicate a strong negative relationship.

numerical_df = df.select_dtypes(
    include=["int64", "float64"]
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    numerical_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png")
plt.show()


# ------------------------------------------------------------
# 20. LIFE EXPECTANCY VS SCHOOLING
# ------------------------------------------------------------

# Explanation:
# Schooling represents the average number of years of schooling.
# This scatter plot helps us understand whether education level
# has a relationship with Life expectancy.

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Schooling",
    y="Life expectancy"
)

plt.title("Life Expectancy vs Schooling")
plt.xlabel("Schooling")
plt.ylabel("Life Expectancy")

plt.tight_layout()

plt.savefig("outputs/life_expectancy_vs_schooling.png")
plt.show()


# ------------------------------------------------------------
# 21. LIFE EXPECTANCY VS ADULT MORTALITY
# ------------------------------------------------------------

# Explanation:
# Adult Mortality represents the probability of dying between
# certain adult age groups. This plot helps us examine its
# relationship with Life expectancy.

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Adult Mortality",
    y="Life expectancy"
)

plt.title("Life Expectancy vs Adult Mortality")
plt.xlabel("Adult Mortality")
plt.ylabel("Life Expectancy")

plt.tight_layout()

plt.savefig("outputs/life_expectancy_vs_adult_mortality.png")
plt.show()


# ------------------------------------------------------------
# 22. LIFE EXPECTANCY VS GDP
# ------------------------------------------------------------

# Explanation:
# GDP is an economic indicator. This scatter plot helps us
# examine whether economic conditions are associated with
# differences in Life expectancy.

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="GDP",
    y="Life expectancy"
)

plt.title("Life Expectancy vs GDP")
plt.xlabel("GDP")
plt.ylabel("Life Expectancy")

plt.tight_layout()

plt.savefig("outputs/life_expectancy_vs_gdp.png")
plt.show()


# ------------------------------------------------------------
# 23. LIFE EXPECTANCY BY DEVELOPMENT STATUS
# ------------------------------------------------------------

# Explanation:
# The dataset contains a Status variable indicating whether
# a country is Developed or Developing.
#
# A boxplot is used to compare the distribution of Life expectancy
# between these two groups.

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Status",
    y="Life expectancy"
)

plt.title("Life Expectancy by Development Status")
plt.xlabel("Development Status")
plt.ylabel("Life Expectancy")

plt.tight_layout()

plt.savefig("outputs/life_expectancy_by_status.png")
plt.show()


print("\nEDA completed successfully!")