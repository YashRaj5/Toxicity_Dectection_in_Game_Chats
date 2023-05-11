# Databricks notebook source
# DBTITLE 1,Running setup configuration
# MAGIC %run ./00_setup

# COMMAND ----------

database_name

# COMMAND ----------

# DBTITLE 1,Reinitiate database environment
spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")
spark.sql(f"CREATE DATABASE {database_name} location 'dbfs:/tmp/yash.raj/gaming/'")
spark.sql(f"USE {database_name}")

# COMMAND ----------

# DBTITLE 1,Writing Data to Delta Lake
# MAGIC %md
# MAGIC Key features of Delta Lake include:
# MAGIC
# MAGIC * ACID Transactions: Ensures data integrity and read consistency with complex, concurrent data pipelines.
# MAGIC * Unified Batch and Streaming Source and Sink: A table in Delta Lake is both a batch table, as well as a streaming source and sink. Streaming data ingest, batch historic backfill, and interactive queries all just work out of the box.
# MAGIC * Schema Enforcement and Evolution: Ensures data cleanliness by blocking writes with unexpected.
# MAGIC * Time Travel: Query previous versions of the table by time or version number.
# MAGIC * Deletes and upserts: Supports deleting and upserting into tables with programmatic APIs.
# MAGIC * Open Format: Stored as Parquet format in blob storage.
# MAGIC * Audit History: History of all the operations that happened in the table.
# MAGIC * Scalable Metadata management: Able to handle millions of files are scaling the metadata operations with Spark.

# COMMAND ----------

# import pandas as pd
# df = spark.createDataFrame(pd.read_csv('/dbfs/tmp/yash.raj/train.csv'))

# creating a dataframe for referencing 'train.csv' data
trainDF = spark.read.csv('dbfs:/tmp/yash.raj/train.csv', header = True, multiLine=True, escape='"')
# creating table for train data
trainDF.write.format("delta").mode("overwrite").saveAsTable("toxicity_training")

# reading a datagrame for referencing 'test.csv' data
testDF = spark.read.csv(f'dbfs:/tmp/yash.raj/test.csv', header=True, multiLine=True, escape='"')
testDF.write.format("delta").mode("overwrite").saveAsTable("toxicity_test")

# COMMAND ----------

# DBTITLE 1,Loading Game data into Delta Lake
for file in ['match','match_outcomes','player_ratings','players','chat','cluster_regions']:
    df = spark.read.csv(f"dbfs:/tmp/yash.raj/{file}.csv",header=True,escape='"',multiLine=True)
    df.write.format("delta").mode("overwrite").saveAsTable(f"toxicity_{file}")

# COMMAND ----------

# MAGIC %md
# MAGIC Due to the table only having chat messages, we can disable column level statistics for faster queries and streaming jobs. Note: These settings should only be used when tuning specific performance of a table and not generally used.

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE toxicity_chat SET TBLPROPERTIES
# MAGIC (
# MAGIC  'delta.checkpoint.writeStatsAsStruct' = 'false',
# MAGIC  'delta.checkpoint.writeStatsAsJson' = 'false'
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploring the data

# COMMAND ----------

# DBTITLE 1,Region group count of players & messages
# MAGIC %sql
# MAGIC SELECT region,
# MAGIC   count(distinct account_id) `# of players`,
# MAGIC   count(key) `# of messages`
# MAGIC FROM Toxicity_chat
# MAGIC JOIN Toxicity_players
# MAGIC ON Toxicity_chat.match_id = Toxicity_players.match_id
# MAGIC JOIN Toxicity_match
# MAGIC ON Toxicity_match.match_id = Toxicity_players.match_id
# MAGIC JOIN Toxicity_cluster_regions
# MAGIC ON Toxicity_match.cluster = Toxicity_cluster_regions.cluster
# MAGIC GROUP BY region
# MAGIC ORDER BY count(account_id) desc, count(account_id) desc

# COMMAND ----------

# DBTITLE 1,Number of messages sent per account
# MAGIC %sql
# MAGIC SELECT account_id,
# MAGIC   count(key) `# of messages` FROM Toxicity_chat
# MAGIC JOIN Toxicity_players
# MAGIC ON Toxicity_chat.match_id = Toxicity_players.match_id
# MAGIC AND Toxicity_chat.slot = Toxicity_players.player_slot
# MAGIC GROUP BY account_id
# MAGIC ORDER BY count(key) desc
