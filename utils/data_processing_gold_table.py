import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    if not os.path.exists(filepath):
        print('no data, skipped:', filepath)
        return None
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # no loan is at this mob in this month (e.g. the first 6 months), nothing to save
    if df.count() == 0:
        print('no loans at mob', mob, 'in', snapshot_date_str + ', skipped')
        return df

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_features_gold_table(snapshot_date_str, silver_attributes_directory, silver_financials_directory, silver_clickstream_directory, gold_feature_store_directory, spark, lookback_months=6):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver tables (attributes and financials are captured at loan application = snapshot_date)
    attributes_filepath = silver_attributes_directory + "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    financials_filepath = silver_financials_directory + "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    if not os.path.exists(attributes_filepath) or not os.path.exists(financials_filepath):
        print('no data, skipped:', attributes_filepath)
        return None
    attributes_df = spark.read.parquet(attributes_filepath)
    financials_df = spark.read.parquet(financials_filepath)
    print('loaded from:', attributes_filepath, 'row count:', attributes_df.count())
    print('loaded from:', financials_filepath, 'row count:', financials_df.count())

    # join attributes with financials, drop PII (Name, SSN)
    df = attributes_df.select("Customer_ID", "snapshot_date", "Age", "Occupation")
    df = df.join(financials_df, on=["Customer_ID", "snapshot_date"], how="left")

    # feature: one hot encode occupation
    occupations = ["Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur", "Journalist", "Lawyer",
                   "Manager", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer"]
    for occupation in occupations:
        df = df.withColumn("occupation_" + occupation.lower(), F.when(col("Occupation") == occupation, 1).otherwise(0))

    # feature: encode categorical columns as numbers (unknown stays null)
    df = df.withColumn("credit_mix_score", F.when(col("Credit_Mix") == "Bad", 0).when(col("Credit_Mix") == "Standard", 1).when(col("Credit_Mix") == "Good", 2))
    df = df.withColumn("pays_min_amount", F.when(col("Payment_of_Min_Amount") == "Yes", 1).when(col("Payment_of_Min_Amount") == "No", 0))
    df = df.withColumn("high_spent", F.when(col("Payment_Behaviour").startswith("High"), 1).when(col("Payment_Behaviour").startswith("Low"), 0))
    df = df.withColumn("payment_value_size", F.when(col("Payment_Behaviour").contains("Small"), 1).when(col("Payment_Behaviour").contains("Medium"), 2).when(col("Payment_Behaviour").contains("Large"), 3))

    # feature: ratios
    df = df.withColumn("debt_to_income", col("Outstanding_Debt") / col("Annual_Income"))
    df = df.withColumn("emi_to_salary", col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))
    df = df.withColumn("invest_to_salary", col("Amount_invested_monthly") / col("Monthly_Inhand_Salary"))
    df = df.withColumn("balance_to_salary", col("Monthly_Balance") / col("Monthly_Inhand_Salary"))

    # feature: existing loans by type, Type_of_Loan looks like "Auto Loan, Student Loan, and Payday Loan"
    loan_types = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", "Mortgage Loan",
                  "Not Specified", "Payday Loan", "Personal Loan", "Student Loan"]
    for loan_type in loan_types:
        column_name = "has_" + loan_type.lower().replace(" ", "_").replace("-", "_")
        df = df.withColumn(column_name, F.when(col("Type_of_Loan").contains(loan_type), 1).otherwise(0))

    # feature: clickstream of the last months up to snapshot_date (only past data, no leakage)
    clickstream_df = spark.read.parquet(silver_clickstream_directory + "*.parquet")
    clickstream_df = clickstream_df.withColumn("months_ago", F.months_between(F.lit(snapshot_date), col("snapshot_date")).cast(IntegerType()))
    clickstream_df = clickstream_df.filter((col("months_ago") >= 0) & (col("months_ago") < lookback_months))

    # for each fe_x: value in the snapshot month, average of last 3 months, average of last 6 months
    agg_exprs = [F.count("*").alias("clickstream_months")]
    for i in range(1, 21):
        fe = "fe_" + str(i)
        agg_exprs.append(F.max(F.when(col("months_ago") == 0, col(fe))).alias(fe + "_last"))
        agg_exprs.append(F.avg(F.when(col("months_ago") < 3, col(fe))).alias(fe + "_avg_3m"))
        agg_exprs.append(F.avg(col(fe)).alias(fe + "_avg_6m"))
    clickstream_df = clickstream_df.groupBy("Customer_ID").agg(*agg_exprs)

    df = df.join(clickstream_df, on="Customer_ID", how="left")

    # select columns to save: drop the raw text columns, keep keys + numeric features
    df = df.drop("Occupation", "Type_of_Loan", "Credit_Mix", "Credit_History_Age", "Payment_of_Min_Amount", "Payment_Behaviour")

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count(), 'columns:', len(df.columns))

    return df
