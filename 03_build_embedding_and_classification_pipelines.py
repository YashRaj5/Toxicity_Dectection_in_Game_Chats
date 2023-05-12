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

# MAGIC %md
# MAGIC The models built in this notebook uses Spark NLP to classify toxic comments and is an adaptation of a [demo notebook](https://nlp.johnsnowlabs.com/2021/01/21/multiclassifierdl_use_toxic_sm_en.html) published by John Snow Labs.
# MAGIC
# MAGIC Spark NLP is an open source library that is built on top of Apache Spark™ and Spark ML. A few benefits of using Spark NLP include:
# MAGIC * Start-of-the-Art: pre-trained algorithms available out-of-the-box
# MAGIC * Efficient: single processing framework mitigates serializing/deserializing overhead
# MAGIC * Enterprise Ready: successfully deployed by many large enterprises.
# MAGIC
# MAGIC Further information on Spark-NLP and more can be found here.
# MAGIC * Transformers documentation used in pipeline
# MAGIC * Annotators documentation used in pipeline
# MAGIC
# MAGIC
# MAGIC Lets jump in and build our pipeline.
# MAGIC * Document Assembler creates the first annotation of type Document from the contents of our dataframe. This is used by the annotators in subsequent steps.
# MAGIC * Embeddings map words to vectors. A great explanation on this topic can be found here. The embeddings serve as an input for our classifier.
# MAGIC <img src = "https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/nlp_pipeline.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.
# MAGIC
# MAGIC The Conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet) is working on tools to help improve the online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful, or otherwise likely to make someone leave a discussion). So far they’ve built a range of publicly available models served through the Perspective API, including toxicity. But the current models still make errors, and they don’t allow users to select which types of toxicity they’re interested in finding (e.g. some platforms may be fine with profanity, but not with other types of toxic content).
# MAGIC
# MAGIC Automatically detect identity hate, insult, obscene, severe toxic, threat, or toxic content in SM comments using our out-of-the-box Spark NLP Multiclassifier DL. We removed the records without any labels in this model. (only 14K+ comments were used to train this model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining the Document Assemble and Embedding Stages

# COMMAND ----------

# MAGIC %md
# MAGIC As per John Snow Labs documentation:
# MAGIC The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the STS benchmark for semantic similarity, and the results can be seen in the example notebook made available. The [universal-sentence-encoder](https://sparknlp.org/2020/04/17/tfhub_use.html) model is trained with a deep averaging network (DAN) encoder.

# COMMAND ----------

document_assembler = DocumentAssembler() \
  .setInputCol("comment_text") \
  .setOutputCol("document")
 
universal_embeddings = UniversalSentenceEncoder.pretrained() \
  .setInputCols(["document"]) \
  .setOutputCol("universal_embeddings")  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Classifier Stage
# MAGIC [MultiClassifier DL Approach](https://nlp.johnsnowlabs.com/docs/en/annotators#multiclassifierdl-multi-label-text-classification) is Mult-label Text Classification. MulitClassifierDL uses a Bidirectional GRU with Convolution model that was build inside TensorFlow

# COMMAND ----------

threshold = 0.7
batchSize = 32
maxEpochs = 10
learningRate = 1e-3
 
ClassifierDL = MultiClassifierDLApproach() \
  .setInputCols(["universal_embeddings"]) \
  .setOutputCol("class") \
  .setLabelColumn("labels") \
  .setMaxEpochs(maxEpochs) \
  .setLr(learningRate) \
  .setBatchSize(batchSize) \
  .setThreshold(threshold) \
  .setOutputLogsPath('./') \
  .setEnableOutputLogs(False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining the model pipeline

# COMMAND ----------

EndToEndPipeline = Pipeline(stages=[
  document_assembler,
  universal_embeddings,
  ClassifierDL
])

# COMMAND ----------

# MAGIC %md
# MAGIC # Mulitlabel Classification Training
# MAGIC * Create experiment and enable autologging for spark.
# MAGIC * Train and evaluate the model, logging model metrics along the way.
# MAGIC * End the tracked run. Results will be viewable in the experiments tab on the top right of the UI.
# MAGIC <img src = "https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/mlflow.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating Experiment & Start Autologging
# MAGIC We want experiments to persist outside of this notebook and to allow others to collaborate with their work on the same project.
# MAGIC * Create experiment in users folder to hold model artifacts and parameters
# MAGIC
# MAGIC Note: When running this code for production, change the experiment path to a location outside of a user's personal folder.

# COMMAND ----------

# Create an experiment with a name that is unique and case sensitive
client = MlflowClient()
 
mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')}/Toxicity_Classification")
 
mlflow.spark.autolog()

# COMMAND ----------

# print(dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user'))

# COMMAND ----------

# MAGIC %md
# MAGIC Under your user folder, you will find an experiment created to log the runs with parameters and metrics. The model will also be logged in the model registry during the run, similar to the image on the right.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and Evaluate the Model
# MAGIC
# MAGIC Note: This implementation uses the basic fit & transform without cross validation or parameter grids.

# COMMAND ----------

with mlflow.start_run():
  
  mlflow.log_param('threshold',threshold)
  mlflow.log_param('batchSize',batchSize)
  mlflow.log_param('maxEpochs',maxEpochs)
  mlflow.log_param('learningRate',learningRate)
  
  model = EndToEndPipeline.fit(train)
  
  mlflow.spark.log_model(model,"spark-model",registered_model_name='Toxicity MultiLabel Classification', pip_requirements=["spark-nlp"])
  
  #supports "f1" (default), "weightedPrecision", "weightedRecall", "accuracy"
  evaluator = MultilabelClassificationEvaluator(labelCol="label_true",predictionCol="label_pred")
  
  predictions = model.transform(val)\
   .withColumn('label_pred', array(
       when(array_contains(col('class.result'),'toxic'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'severe_toxic'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'obscene'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'threat'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'insult'),1).otherwise(0).cast(DoubleType()),
       when(array_contains(col('class.result'),'identity_hate'),1).otherwise(0).cast(DoubleType())
     )
   )
  
  score = evaluator.evaluate(predictions)
  mlflow.log_metric('f1',score)
  print(score)
  
mlflow.end_run()

# COMMAND ----------

print('HeHe')
