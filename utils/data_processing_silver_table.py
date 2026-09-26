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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    if not os.path.exists(filepath):
        print('no data, skipped:', filepath)
        return None
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_silver_clickstream_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    if not os.path.exists(filepath):
        print('no data, skipped:', filepath)
        return None
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }
    for i in range(1, 21):
        column_type_map["fe_" + str(i)] = IntegerType()

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_attributes_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    if not os.path.exists(filepath):
        print('no data, skipped:', filepath)
        return None
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: Age comes as text like "31_", "-500", "8678" -> remove "_" and only keep realistic ages
    df = df.withColumn("Age", F.regexp_replace(col("Age").cast(StringType()), "_", "").cast(IntegerType()))
    df = df.withColumn("Age", F.when((col("Age") >= 1) & (col("Age") <= 100), col("Age")))

    # clean data: "_______" means occupation is unknown
    df = df.withColumn("Occupation", F.when(col("Occupation") != "_______", col("Occupation")))

    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_financials_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    if not os.path.exists(filepath):
        print('no data, skipped:', filepath)
        return None
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: some numbers come as text with "_" e.g. "52312.68_" -> 52312.68
    # values starting with "_" (e.g. "_", "__10000__") are placeholders for missing -> null
    dirty_numeric_columns = ["Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
                             "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance"]
    for column in dirty_numeric_columns:
        df = df.withColumn(column, F.when(col(column).cast(StringType()).startswith("_"), None)
                                    .otherwise(F.regexp_replace(col(column).cast(StringType()), "_", "")))

    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # clean data: impossible values e.g. 1756 bank accounts, 5789% interest, -100 loans -> null
    valid_range = {
        "Annual_Income": (0, 5000000),
        "Num_Bank_Accounts": (0, 30),
        "Num_Credit_Card": (0, 30),
        "Interest_Rate": (0, 100),
        "Num_of_Loan": (0, 20),
        "Num_of_Delayed_Payment": (0, 100),
        "Num_Credit_Inquiries": (0, 1000),
    }
    for column, (low, high) in valid_range.items():
        df = df.withColumn(column, F.when((col(column) >= low) & (col(column) <= high), col(column)))

    # clean data: garbage categories "_" and "!@9#%8" -> null
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix").isin("Bad", "Standard", "Good"), col("Credit_Mix")))
    df = df.withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour").contains("spent"), col("Payment_Behaviour")))

    # augment data: "10 Years and 9 Months" -> 129 months
    years = F.regexp_extract(col("Credit_History_Age"), r"(\d+) Year", 1).cast(IntegerType())
    months = F.regexp_extract(col("Credit_History_Age"), r"(\d+) Month", 1).cast(IntegerType())
    df = df.withColumn("Credit_History_Months", years * 12 + months)

    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
