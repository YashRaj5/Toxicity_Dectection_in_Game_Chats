# Databricks notebook source
# DBTITLE 1,Tasks performed in this notebook
# MAGIC %md
# MAGIC * Configuring the environment
# MAGIC * Explore the training data
# MAGIC * Prep the training data
# MAGIC * Understnad sentence embeddings
# MAGIC * Build embedding and classification pipelines
# MAGIC * Track model training with MLflow
# MAGIC   * Trian
# MAGIC   * Tune
# MAGIC   * Evaluate
# MAGIC   * Register

# COMMAND ----------

# MAGIC %md
# MAGIC * **Install libraries:**
# MAGIC   * Maven Coordinates:
# MAGIC     * CPU: com.johnsnowlabs.nlp:spark-nlp_2.12:4.0.0
# MAGIC     * GPU: com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:4.0.0

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Importing libraries
import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
 
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.sql.functions import lit,when,col,array,array_contains,array_remove,regexp_replace,size,when
from pyspark.sql.types import ArrayType,DoubleType,StringType
 
from pyspark.ml.evaluation import MultilabelClassificationEvaluator
 
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploring the Training Dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM Toxicity_training;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM Toxicity_training WHERE toxic = 0 LIMIT 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM Toxicity_training WHERE toxic = 1 LIMIT 1;

# COMMAND ----------

# DBTITLE 1,Training data prep
dataPrepDF = spark.table("Toxicity_training")\
  .withColumnRenamed("toxic","toxic_true")\
  .withColumnRenamed("severe_toxic","severe_toxic_true")\
  .withColumnRenamed("obscene","obscene_true")\
  .withColumnRenamed("threat","threat_true")\
  .withColumnRenamed("insult","insult_true")\
  .withColumnRenamed("identity_hate","identity_hate_true")\
  .withColumn('toxic',when(col('toxic_true') == '1','toxic').otherwise(0))\
  .withColumn('severe_toxic',when(col('severe_toxic_true') == '1','severe_toxic').otherwise(0))\
  .withColumn('obscene',when(col('obscene_true') == '1','obscene').otherwise(0))\
  .withColumn('threat',when(col('threat_true') == '1','threat').otherwise(0))\
  .withColumn('insult',when(col('insult_true') == '1','insult').otherwise(0))\
  .withColumn('identity_hate',when(col('identity_hate_true') == '1','identity_hate').otherwise(0))\
  .withColumn('labels',array_remove(array('toxic','severe_toxic','obscene','threat','insult','identity_hate'),'0')\
              .astype(ArrayType(StringType())))\
  .drop('toxic','severe_toxic','obscene','threat','insult','identity_hate')\
  .withColumn('label_true', array(
    col('toxic_true').cast(DoubleType()),
    col('severe_toxic_true').cast(DoubleType()),
    col('obscene_true').cast(DoubleType()),
    col('threat_true').cast(DoubleType()),
    col('insult_true').cast(DoubleType()),
    col('identity_hate_true').cast(DoubleType()))
  )
  
train, val = dataPrepDF.randomSplit([0.8,0.2],42)

# COMMAND ----------

display(dataPrepDF)

# COMMAND ----------

# DBTITLE 1,Displaying Training Data
display(train.limit(1).filter(size(col('labels')) == 0))

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Pipeline

# COMMAND ----------


