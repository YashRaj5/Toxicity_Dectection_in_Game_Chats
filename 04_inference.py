# Databricks notebook source
# MAGIC %md
# MAGIC In this notebook we are going to:
# MAGIC * Load a model from MLflow
# MAGIC * Productionalize a streaing & batch inference pipeline
# MAGIC * Explore the impact of toxicity

# COMMAND ----------

# DBTITLE 1,Configuring the Environment
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Importing Libraries
from pyspark.sql.functions import col, struct
from pyspark.sql.types import *
import mlflow

# COMMAND ----------

# DBTITLE 1,Loading classification model form MLflow Model Registry
# MAGIC %md
# MAGIC The MLflow Model Registry component is a centralized model store, a set of APIs, and UI, used to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (e.g. which MLflow experiment and run produced the model), model versioning, stage transitions (e.g. from staging to production), and annotations.

# COMMAND ----------

model_name='Toxicity MultiLabel Classification'
stage = None
 
model = mlflow.spark.load_model(f'models:/{model_name}/{stage}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Productionalizing ML Pipelines with Batch or Streaming
# MAGIC Note: in this solution accelerator, we are storing the data back into Delta Lake, but we could just as easily push out events or alerts based on the inference results.
# MAGIC <img src = "https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/data-pipelines.png">
# MAGIC ##### Structured Streaming for One API that handles Batch & Streaming
# MAGIC [Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html) is a scalable and fault-tolerant stream processing engine built on the Spark SQL engine. You can express your streaming computation the same way you would express a batch computation on static data. The Spark SQL engine will take care of running it incrementally and continuously and updating the final result as streaming data continues to arrive. You can use the Dataset/DataFrame API in Scala, Java, Python or R to express streaming aggregations, event-time windows, stream-to-batch joins, etc.
# MAGIC
# MAGIC [Here](https://docs.databricks.com/spark/latest/structured-streaming/index.html) is a link that shows examples and code for streaming with Kafka, Kinesis, and other popular sources.

# COMMAND ----------

# DBTITLE 1,Reading the Stream
raw_comments = spark.readStream.format("Delta")\
  .table("Toxicity_Chat")\
  .withColumnRenamed('key', 'comment_text')\
  .repartition(5000)

# COMMAND ----------

# DBTITLE 1,Inference on Streaming Dataframe
comments_pred = model.transform(raw_comments)\
  .withColumnRenamed('key', 'comment_text')\
  .drop('document', 'token', 'universal_embeddings')\
  .withColumn('predicted',col('class.result'))

# COMMAND ----------

# DBTITLE 1,Writing Stream
# For the sake of the accelerator, we clean up any previous checkpoint and start the stream. We write the output of comments_pred to the delta table "Toxicity_Chat_Pred"

# Initialize a user-specific checkpoint
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
checkpoint = f"/toxicity_accelerator/{user}/_checkpoints/Toxicity_Chat_Pred"
dbutils.fs.rm(checkpoint, True)

# COMMAND ----------

# # The trigger makes the Structured Streaming pipeline run once
# comments_pred.writeStream\
#   .trigger(once=True)\
#   .format("Delta")\
#   .option("checkpointLocation", checkpoint)\
#   .option("mergeSchema", "true")\
#   .table("Toxicity_Chat_Pred") \
#   .awaitTermination() # set awaitTermination to block subsequent blocks from execution

# COMMAND ----------

# DBTITLE 1,Dataframe API
# The dataframe api is an optional way to do batch inference. The below cell will recreate the same results as the streaming job above.
spark.sql("USE gaming_yash_raj")
spark.sql("DROP TABLE IF EXISTS Toxicity_Chat_Pred")

chatDF = spark.table("Toxicity_Chat").withColumnRenamed('key', 'comment_text').repartition(5000)
chatDF = model.transform(chatDF)\
  .withColumn('predicted',col('class.result'))\
  .drop('document', 'token', 'universal_embeddings', 'class')
 
chatDF.write.format("delta").mode("overwrite").saveAsTable("Toxicity_Chat_Pred")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display new table with inferred labesl

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT match_id, comment_text, slot, time, unit, predicted FROM Toxicity_Chat_Pred LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now that we have the pipeline and the silver table with the predicted labels, we can move onto combing the labeled data with our game data
# MAGIC <img src = "https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/delta-lake-silver.png">

# COMMAND ----------

# DBTITLE 1,Exploring the impact of toxicity on 50K Dota 2 Matches
# MAGIC %md
# MAGIC Toxicity tables Relationship Diagram
# MAGIC <img src = "https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/toxicity-erd.png">

# COMMAND ----------

# DBTITLE 0,Regional Analysis
# MAGIC %md
# MAGIC ### Regional Analysis
# MAGIC Top 5 Regions based on the number of toxic messages

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT region,
# MAGIC   Round(count(distinct account_id)) `# of toxic players`,
# MAGIC   Round(count(comment_text)) `# of toxic messages`
# MAGIC FROM Toxicity_chat_pred
# MAGIC JOIN Toxicity_players
# MAGIC ON Toxicity_chat_pred.match_id = Toxicity_players.match_id
# MAGIC JOIN Toxicity_match
# MAGIC ON Toxicity_match.match_id = Toxicity_players.match_id
# MAGIC JOIN Toxicity_cluster_regions
# MAGIC ON Toxicity_match.cluster = Toxicity_cluster_regions.cluster
# MAGIC WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC GROUP BY region
# MAGIC ORDER BY count(account_id) desc, count(account_id) desc
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ### Message Analysis
# MAGIC Number of messages per label/type of toxicity

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'Toxic', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC UNION
# MAGIC SELECT 'Non-Toxic', count(*) FROM Toxicity_chat_pred WHERE size(predicted) > 0

# COMMAND ----------

# MAGIC %sql
# MAGIC -- speeding up the above query
# MAGIC OPTIMIZE Toxicity_chat_pred
# MAGIC -- ZORDER BY match_id

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'Toxic', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC UNION
# MAGIC SELECT 'Non-Toxic', count(*) FROM Toxicity_chat_pred WHERE size(predicted) > 0

# COMMAND ----------

# MAGIC %md
# MAGIC Before query was taking 2.26 minutes for execution and afte optimizing it is taking 
# MAGIC 1.68 seconds
# MAGIC And even the size of table before was around 60 MB with around 5000 files and now it is around 38 MB and with only 1 file

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Toxic AS Label_Type, Message_Count from (
# MAGIC   SELECT 'Toxic', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC   UNION
# MAGIC   SELECT 'Obscene', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'obscene')
# MAGIC   UNION
# MAGIC   SELECT 'Insult', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'insult')
# MAGIC   UNION
# MAGIC   SELECT 'Threat', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'threat')
# MAGIC   UNION
# MAGIC   SELECT 'Identity_hate', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'identity_hate')
# MAGIC   UNION
# MAGIC   SELECT 'Severe_toxic', count(*) AS Message_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'severe_toxic')
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Number of messages per 1,2,3,4,5 labels
# MAGIC SELECT size(predicted) AS Number_of_Labels, count(*) AS Message_Count FROM Toxicity_chat_pred WHERE size(predicted) > 0 GROUP BY size(predicted) ORDER BY size(predicted) ASC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Match Analysis
# MAGIC We can see of the 50k match dataset, 58% of the matches contained some form of toxicity. Below is the % per label. Keep in mind the Toxic label is included with all other labels.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Toxic AS Label_Type, Match_Count, Round((Match_Count/(SELECT count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred))*100) AS Percent_of_total_matches from (
# MAGIC   SELECT 'Toxic', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC   UNION
# MAGIC   SELECT 'Obscene', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'obscene')
# MAGIC   UNION
# MAGIC   SELECT 'Insult', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'insult')
# MAGIC   UNION
# MAGIC   SELECT 'Threat', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'threat')
# MAGIC   UNION
# MAGIC   SELECT 'Identity_hate', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'identity_hate')
# MAGIC   UNION
# MAGIC   SELECT 'Severe_toxic', count(DISTINCT match_id) AS Match_Count FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'severe_toxic')
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC Number of Toxic Messages based on match time ranges

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Game_Lobby AS Timeline,
# MAGIC   Number_of_toxic_messages
# MAGIC   FROM (
# MAGIC     SELECT 'Game_Lobby', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time < 0
# MAGIC     UNION 
# MAGIC     SELECT 'Less_Than_5_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time < 300
# MAGIC     UNION 
# MAGIC     SELECT '5-10_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time BETWEEN 300 AND  600
# MAGIC     UNION
# MAGIC     SELECT '10-20_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time BETWEEN 600 AND  1200
# MAGIC     UNION
# MAGIC     SELECT '20-30_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time BETWEEN 1200 AND 1800
# MAGIC     UNION
# MAGIC     SELECT '30-40_minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time BETWEEN 1800 AND 2400
# MAGIC     UNION
# MAGIC     SELECT '40+minutes', count(comment_text) `Number_of_toxic_messages` FROM Toxicity_chat_pred WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic') AND time > 2400)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Player Analysis
# MAGIC Top 10 Players with the highest count of toxic messages

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT  account_id,
# MAGIC   count(comment_text) `# of messages`
# MAGIC FROM Toxicity_chat_pred
# MAGIC JOIN Toxicity_players
# MAGIC ON Toxicity_chat_pred.match_id = Toxicity_players.match_id
# MAGIC AND Toxicity_chat_pred.slot = Toxicity_players.player_slot
# MAGIC WHERE array_contains(Toxicity_chat_pred.predicted, 'toxic')
# MAGIC GROUP BY account_id
# MAGIC ORDER BY count(comment_text) desc
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC Any of these queries we could now save as our gold layer tables for consumption by the business or analysts

# COMMAND ----------

spark.sql("DROP DATABASE gaming_yash_raj CASCADE")
