# quick sanity check (not part of the pipeline): join feature store with label store and fit a simple model
import glob
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

label_files = glob.glob("datamart/gold/label_store/*")
feature_files = glob.glob("datamart/gold/feature_store/*")
labels_df = spark.read.parquet(*label_files)
features_df = spark.read.parquet(*feature_files)

# label is observed at mob 6, features are at the loan application date (6 months earlier)
labels_df = labels_df.withColumn("snapshot_date", F.add_months(col("snapshot_date"), -6))
df = labels_df.join(features_df, on=["Customer_ID", "snapshot_date"], how="inner")
pdf = df.toPandas()
print("label rows:", labels_df.count(), "rows after joining with features:", len(pdf))

feature_columns = [c for c in pdf.columns if c not in ["loan_id", "Customer_ID", "label", "label_def", "snapshot_date"]]
print("number of features:", len(feature_columns))
print("bad rate:", round(pdf["label"].mean(), 4))

# out of time split: train on 2023 applications, test on 2024
train = pdf[pdf["snapshot_date"] < pd.Timestamp("2024-01-01").date()]
test = pdf[pdf["snapshot_date"] >= pd.Timestamp("2024-01-01").date()]
X_train, y_train = train[feature_columns].astype(float), train["label"]
X_test, y_test = test[feature_columns].astype(float), test["label"]
print("train rows:", len(train), "test rows:", len(test))

# logistic regression (impute + scale fitted on train only) and gradient boosting (handles null itself)
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
lr.fit(X_train, y_train)
print("logistic regression  train auc:", round(roc_auc_score(y_train, lr.predict_proba(X_train)[:, 1]), 3),
      "test auc:", round(roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]), 3))

gb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=40, l2_regularization=1.0, random_state=42)
gb.fit(X_train, y_train)
print("gradient boosting    train auc:", round(roc_auc_score(y_train, gb.predict_proba(X_train)[:, 1]), 3),
      "test auc:", round(roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1]), 3))

spark.stop()
