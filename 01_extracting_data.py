# Databricks notebook source
# MAGIC %md
# MAGIC Toxicity can have a large impact on player engagement and satisfaction. Game companies are working on ways to address forms of toxicity in their platforms. One of the most common interactions with toxicity is in chat boxes or in-game messaging systems. As companies are becoming more data driven, the opportunity to detect toxicity using the data at hand is present, but technically challenging. This solution accelerator is a head start on deploying a ML-enhanced data pipeline to address toxic messages in real time.

# COMMAND ----------

# MAGIC %md
# MAGIC In these series of notebooks we are going to use multi-label classification to detect and analyze toxicity in data.
# MAGIC
# MAGIC In support of this goal, we will:
# MAGIC
# MAGIC * Load toxic-comment training data from Jigsaw and game data from Dota 2.
# MAGIC * Create one pipeline for streaming and batch to detect toxicity in near real-time and/or on an ad-hoc basis. This pipeline can then be used for managing tables for reporting, ad hoc queries, and/or decision support.
# MAGIC * Label text chat data using Multi-Label Classification.
# MAGIC * Create a dashboard for monitoring the impact of toxicity.

# COMMAND ----------

# MAGIC %md
# MAGIC Jigsaw Dataset
# MAGIC * The dataset used in this accelerator is from Jigsaw. Jigsaw is a unit within Google that does work to create a safer internet. Some of the areas that Jigsaw focuses on include: disinformation, censorship, and toxicity.
# MAGIC
# MAGIC * Jigsaw posted this dataset on Kaggle three years ago for the toxic comment classification challenge.  
# MAGIC
# MAGIC * This is a multilabel classification problem that includes the following labels: Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate 

# COMMAND ----------

# MAGIC %md
# MAGIC **DOTA 2 Matches Dataset**
# MAGIC * This dataset is from is a multiplayer online battle arena (MOBA) video game developed and published by Valve.
# MAGIC
# MAGIC * Dota 2 is played in matches between two teams of five players, with each team occupying and defending their own separate base on the map.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Setting Up the Enviroment

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Downloading the data
# MAGIC In this step, we will:
# MAGIC
# MAGIC * Download the Jigsaw dataset
# MAGIC * Unzip the Jigsaw dataset
# MAGIC * Download the Dota 2 game dataset
# MAGIC * Unzip the Dota 2 game dataset

# COMMAND ----------

# DBTITLE 1,Making directory for raw data
# MAGIC %sh
# MAGIC mkdir -p /root/.kaggle/

# COMMAND ----------

# DBTITLE 1,Adding credentials to directory
# MAGIC %sh
# MAGIC echo """{\"username\":\"$kaggle_username\",\"key\":\"$kaggle_key\"}""" > /root/.kaggle/kaggle.json
# MAGIC chmod 600 /root/.kaggle/kaggle.json

# COMMAND ----------

# MAGIC %md
# MAGIC **This part of the code didn't work, so I had to explicitly download the data**, 
# MAGIC but for reference purposes i am adding the code

# COMMAND ----------

# DBTITLE 1,Downloading the data
# MAGIC %sh -e
# MAGIC rm -rf $tmpdir
# MAGIC  
# MAGIC cd $tmpdir
# MAGIC  
# MAGIC kaggle competitions download -p "$tmpdir" -c jigsaw-toxic-comment-classification-challenge 

# COMMAND ----------

# DBTITLE 1,Unzipping the downloaded the data
# MAGIC %sh -e
# MAGIC cd $tmpdir
# MAGIC unzip -o jigsaw-toxic-comment-classification-challenge.zip
# MAGIC unzip -o train.csv.zip
# MAGIC unzip -o test.csv.zip

# COMMAND ----------

# DBTITLE 1,Copying the downloaded data into the desired location
# tmpdir = '/dbfs/tmp/yash.raj/' # directory to create
# creating the directory explicityly
dbutils.fs.mkdirs('yash.raj')

# COMMAND ----------

dbutils.fs.ls('dbfs:/tmp/yash.raj/')

# COMMAND ----------

# performing the copy of data
ls = ['sample_submission', 'test_labels', 'train', 'test']
for _ in ls:
    dbutils.fs.cp("dbfs:/dbfs/tmp/yash.raj/{0}.csv".format(_), "dbfs:/tmp/yash.raj/")
    print(f"{_} is copied")

# COMMAND ----------

# DBTITLE 1,Downloading Dota data
# MAGIC %sh -e
# MAGIC kaggle datasets download -p $tmpdir -d devinanzelmo/dota-2-matches

# COMMAND ----------

# DBTITLE 1,Unzipping the Dota Match Data
# MAGIC %sh -e
# MAGIC cd $tmpdir
# MAGIC unzip -o dota-2-matches.zip 

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/tmp/yash.raj/'))
