# Databricks notebook source
# MAGIC %md
# MAGIC In this Notebook we are going to setup our access to kaggle account, so that we can access data and load the desired data in future notebooks

# COMMAND ----------

# DBTITLE 1,Installing Kaggle
# MAGIC %pip install kaggle spark-nlp==4.0.0

# COMMAND ----------

# DBTITLE 1,Importing Required Libraries
import re
import os
import json

# COMMAND ----------

# DBTITLE 1,Setting Env Variables
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0]
username_sql = re.sub('\W', '_', username)
tmpdir = f"/dbfs/tmp/{username}/"
tmpdir_dbfs = f"/tmp/{username}"
database_name = f"gaming_{username_sql}"
database_location = f"{tmpdir}gaming"

# COMMAND ----------

# DBTITLE 1,Setting Credentials
import os

# setting usename
os.environ['kaggle_username'] = 'yash5raj' # enter your usename for kaggle account
os.environ['kaggle_key'] = '137e8df3003faa1f564248e5f43a90f3' # paster your kaggle API key for accessing the account

# GENERALY USIGN THE CREDENTIALS LIKE ABOVE IS NOT REALLY A SECURE WAY OF DOING IT, RATHE THAN WE CAN USE 'SECRET' FUNCTIONALITY PROIDED BY DATABRICKS
# os.environ['kaggle_username'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_username")
# os.environ['kaggle_key'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_key")

os.environ['tmpdir'] = tmpdir

# COMMAND ----------

# DBTITLE 1,Creating Database objects
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name} location '{database_location}'")
spark.sql(f"USE {database_name}")
