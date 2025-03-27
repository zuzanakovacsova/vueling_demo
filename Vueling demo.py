# Databricks notebook source
# MAGIC %md
# MAGIC # Vueling workshop
# MAGIC ## Build a simple end-to-end pipeline with Databricks
# MAGIC In this notebook, we will ingest some data related to flight operations, do simple ETL, train a machine learning model and build a dashboard. 
# MAGIC
# MAGIC The goal is to:
# MAGIC - Ingest data, join with other data sources.
# MAGIC - Train an ML model to predict delay.
# MAGIC - Build a dashboard to visualise the data.
# MAGIC - Create a Genie space to explore the data.
# MAGIC - Create a workflow to refresh the notebook and dashboard.
# MAGIC - Add workflow to an asset bundle.

# COMMAND ----------

# MAGIC %md
# MAGIC You need a cluster with Databricks ML runtime to run this notebook.

# COMMAND ----------

# DBTITLE 1,Set up
#Adding widget to disable model predictions when running notebook 
dbutils.widgets.dropdown("run_ml", "yes", ["yes", "no"])



# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data ingestion
# MAGIC First, we need to ingest some data. We will use several data sources and you can also add your own. The main table we will use is Airline performance data. This data can be accessed from your Databricks workspace.
# MAGIC
# MAGIC Navigate to the "Marketplace" tab and search for "airline" products. We will be using the "Airline Performance Data" dataset.
# MAGIC ![](/Workspace/Users/zuzana.kovacsova@databricks.com/vueling_images/Screenshot 2025-03-27 at 16.54.56.png)
# MAGIC
# MAGIC After agreeing with the licencing terms, you can make this dataset accessible in your workspace by clicking on "Get instance access". The data will be made accessible in a catalog. 
# MAGIC
# MAGIC
# MAGIC **Only one user has to make a copy of this data, other users can access the same catalog after appropriate permissions are set in Unity Catalog.**

# COMMAND ----------

# MAGIC %md
# MAGIC Other data sources that are used in this notebook include a table of Airport codes, accesible [here](https://datahub.io/core/airport-codes)
# MAGIC and a database of airline names, found [here](https://www.kaggle.com/datasets/open-flights/airline-database?resource=download)
# MAGIC
# MAGIC If you want, you can add other data sources. Some interesting datasets related to aviation are available from Eurocontrol: [url](https://ansperformance.eu/data/)

# COMMAND ----------

# MAGIC %md
# MAGIC ____________________

# COMMAND ----------

# MAGIC %md
# MAGIC We only have read access to the data from Databricks Marketplace, so we will have to move the data to our own catalog. You can create a catalog and a schema using the statement below, or from the UI.

# COMMAND ----------

# MAGIC %md
# MAGIC Note: The following ETL is using simple SQL statements as compared to a DLT pipeline which would be be used in a real scenario. The reason for not doing DLT is simplicity of demoing; and DLT pipelines also don't allow you to mix SQL and Python in one notebook.

# COMMAND ----------

# DBTITLE 1,Create catalog and schema
# MAGIC %sql
# MAGIC -- Create catalog and schema
# MAGIC -- CREATE CATALOG IF NOT EXISTS <your catalog>
# MAGIC -- CREATE SCHEMA IF NOT EXISTS <your catalog>.airport_demo 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check the mumber of records in the flights_small table ingested from Marketplace. I also encourage you to explore this data in Unity Catalog.

# COMMAND ----------

# DBTITLE 1,Check number of records in Flights dataset
# MAGIC %sql
# MAGIC SELECT count(*) FROM `zuzana_k_airline_performance`.`v01`.`flights_small`;

# COMMAND ----------

# MAGIC %md
# MAGIC We will create our own copy of the data. As the dataset is very large, we will randomly downsample it by 80%. 

# COMMAND ----------

# DBTITLE 1,Copy data from delta sharing location
# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE dbdemos_zuzana_k.airport_demo.flights_small_raw AS SELECT * from zuzana_k_airline_performance.v01.flights_small
# MAGIC TABLESAMPLE (20 PERCENT);

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will add data that we downloaded from the links above; airport codes data and airline names data. You can either upload through the UI or programatically.

# COMMAND ----------

# MAGIC %md
# MAGIC I uploaded the airline names table via the UI; you can add a table by navigating to your schema and selecting "Create table" from the "Create" dropdown in the upper right corner. I named my table "airlines".
# MAGIC
# MAGIC
# MAGIC ![](/Workspace/Users/zuzana.kovacsova@databricks.com/vueling_images/Screenshot 2025-03-27 at 17.15.08.png)

# COMMAND ----------

# MAGIC %md
# MAGIC I imported the airport codes .csv programatically - I first uploaded the file onto my workspace and from there I read it as a .csv.

# COMMAND ----------

# DBTITLE 1,Import airport code data from your workspace directory

import pandas as pd
airport_codes = pd.read_csv("/Workspace/Users/zuzana.kovacsova@databricks.com/airport-codes.csv")
airport_codes.head()


# COMMAND ----------

# DBTITLE 1,Save into a Delta table in your schema
df = spark.createDataFrame(airport_codes)
df.write.format("delta").mode("overwrite").saveAsTable("dbdemos_zuzana_k.airport_demo.airport_codes")

# COMMAND ----------

# MAGIC %md
# MAGIC You can now see the two tables in Unity Catalog.
# MAGIC
# MAGIC You can add more data sources. If ingesting from REST API, once can parse the JSON response into a dataframe and save the data as in step 16. 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Explore your data
# MAGIC %sql
# MAGIC SELECT DISTINCT UniqueCarrier FROM dbdemos_zuzana_k.airport_demo.flights_small_raw

# COMMAND ----------

# MAGIC %sql
# MAGIC -- add some more queries to explore the dataset or ingest more data sources.

# COMMAND ----------

# MAGIC %md
# MAGIC We will join the airline names table onto our main flights table to get the full name of each airline. I am not doing any more data cleaning or exploration but I encourage you to see if any values or columns can be dropped.

# COMMAND ----------

# DBTITLE 1,Join airline table onto flights_small table to find names of airlines
# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE dbdemos_zuzana_k.airport_demo.flights_small_silver AS SELECT flight.Year, flight.Month, flight.DayofMonth, flight.DayOfWeek, flight.DepTime, flight.CRSDepTime, flight.ArrTime, flight.CRSArrTime, flight.FlightNum, flight.TailNum, flight.ActualElapsedTime, flight.CRSElapsedTime, flight.AirTime, flight.ArrDelay, flight.DepDelay, flight.Origin, flight.Dest, flight.Distance, flight.TaxiIn, flight.TaxiOut, flight.Cancelled, flight.CancellationCode, flight.Diverted, flight.IsArrDelayed, flight.IsDepDelayed, airlines.Name as airline_name from dbdemos_zuzana_k.airport_demo.flights_small_raw as flight LEFT JOIN dbdemos_zuzana_k.airport_demo.airlines as airlines
# MAGIC ON flight.UniqueCarrier = airlines.IATA;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC We will take this silver data and use it to train a simple classification model in AutoML. To speed up training, we will downsample it even further. 

# COMMAND ----------

# DBTITLE 1,Create smaller table for AutoML experiment
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE dbdemos_zuzana_k.airport_demo.flights_small_training_data AS SELECT * FROM dbdemos_zuzana_k.airport_demo.flights_small_silver TABLESAMPLE (30 PERCENT)

# COMMAND ----------

# MAGIC %md
# MAGIC Before we jump into training an ML model, let's add some aggregations for a golden layer table that will show us how many delayed flights various airlines have.

# COMMAND ----------

# DBTITLE 1,Create Golden table with some aggregations
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE dbdemos_zuzana_k.airport_demo.airline_performance
# MAGIC AS 
# MAGIC SELECT airline_name, count(IsArrDelayed) as number_of_flights, count(IsArrDelayed) FILTER (WHERE IsArrDelayed = 'YES') as delayed_flights, delayed_flights/number_of_flights*100 as delay_percentage
# MAGIC from dbdemos_zuzana_k.airport_demo.flights_small_silver
# MAGIC GROUP BY airline_name; 

# COMMAND ----------

# DBTITLE 1,Our aggregated table
# MAGIC %sql
# MAGIC SELECT * FROM dbdemos_zuzana_k.airport_demo.airline_performance;

# COMMAND ----------

# MAGIC %md
# MAGIC We can enrich our data by using custom, user defined functions. I have a table with airport codes, which I want to use to encode the departure and arrival cities in the flights table. 
# MAGIC
# MAGIC My SQL function will be stored in Unity Catalog.

# COMMAND ----------

# DBTITLE 1,Create a simple function to look up airport codes
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION dbdemos_zuzana_k.airport_demo.lookup_airport_name(airport_code STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'This function looks up the provided airport code and returns the name of the airport.'
# MAGIC LANGUAGE SQL
# MAGIC RETURN
# MAGIC SELECT MAX(name)
# MAGIC FROM dbdemos_zuzana_k.airport_demo.airport_codes
# MAGIC WHERE iata_code = airport_code;

# COMMAND ----------

# DBTITLE 1,Apply the function to a column
# MAGIC %sql
# MAGIC SELECT Origin, dbdemos_zuzana_k.airport_demo.lookup_airport_name(Origin) as decoded_origin from dbdemos_zuzana_k.airport_demo.flights_small_silver
# MAGIC limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. AutoML Model training
# MAGIC We will now train a model to predict whether a flight will be delayed. I do not expect the model perform very well as the attributes in `flights_small_training_data` does not have much information about delays. 

# COMMAND ----------

# MAGIC %md
# MAGIC Create AutoML experiment to automatically train several models. Navigate to "Experiments" and start training a classification model. You will need a cluster with ML runtime available. Select the `flights_small_training_data` dataset as input, set the "IsArrDelayed" column as Prediction target. Under advanced configuration, select which metric to use for model evaluation (F1 can be used, or accuracy) and set 15 minutes to timeout. This limits how long the experiment takes. On the right, select which columns should be used as features - select which ones you think have an influence on delay status. 

# COMMAND ----------

# MAGIC %md
# MAGIC The result of your experiment will be a list of a few models with different evaluation metrics. The model with the best performance will also have a "Source" notebook. Click on "Viev notebook for best model" to see this notebook, and have a look at the data exploration notebook. 
# MAGIC
# MAGIC Then, click on the name of the best run. You will be taken to a page where you can explore further model metrics and other artifacts. In the top right is the option to "Register model". Select this option to register the model in unity catalog, this will allow us to use this model to predict delay status and also share it with others, if desired. Register the model in unity catalog and enter a new model name. It has to be in the format `<your catalog>.<your schema>.<model name>.` Registering the model takes a few minutes.

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can also retrieve our model and use it to predict some data.

# COMMAND ----------

# DBTITLE 1,Get parameter from widget
run_ml = dbutils.widgets.get("run_ml")

# COMMAND ----------

# DBTITLE 1,Retrieve model and predict delay
from mlflow import MlflowClient
import mlflow

if run_ml == "yes":
  client = MlflowClient()
  # replace with the name of your model
  model_version_uri = "models:/dbdemos_zuzana_k.airport_demo.predict_aircraft_delay@champion"
  model = mlflow.pyfunc.load_model(model_version_uri)
  # Example input features
  import pandas as pd 
  features = pd.DataFrame({
      'Month': [2],
      'DayofMonth': [5],
      'FlightNum': [2082],
      'Origin': ['BOS'],
      'Dest': ['PHX'],
      'airline_name': ['Phoenix Airways'],
  })
  # Ensure correct type
  features = features.astype({
      'Month': 'int32',
      'DayofMonth': 'int32',
      'FlightNum': 'int32',
      'Origin': 'str',
      'Dest': 'str',
      'airline_name': 'str'
  })
  # Predict delay
  prediction = model.predict(features)
  print(prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. BI
# MAGIC We will create a dashboard from our data. Navigate to your aggregate table `airline_performance`. In the top right, select Create dashboard. This will open a new dashboard for yo with the data displayed in a table. You can add more visualisations and add more data. Rename the dashboard by clicking on the name in the centre top. Each dashboard has a "Canvas" and a "Data" tab. If you click on "Data", you will be taken to a page where you can add more data sources. Currently, your dashboard has only access to the `airline_performance` dataset. Add the silver layer data by clicking on "Add data". 

# COMMAND ----------

# MAGIC %md
# MAGIC Navigate back to the "Canvas" tab and add visualisations by selecting the option from the toolbar on the bottom of the screen. You can also ask the AI assistant to plot some visualisation for you. Some prompts for visualisations include: 
# MAGIC - Number of delays per airline
# MAGIC - Which month has the most delays 
# MAGIC - Which city saw the most delayed flights

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/Workspace/Users/zuzana.kovacsova@databricks.com/vueling_images/Screenshot 2025-03-27 at 19.16.50.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Some questions, such as "Which city gets the most delays", are better explored in the SQL query editor. 

# COMMAND ----------

# MAGIC %md
# MAGIC Once your dashboard is set up, you can also create a Genie space. Navigate to the "Genie" tool on the left navigation bar and select "New". Add your dayaseys, `airline_performance` and `flights_small_silver`. 
# MAGIC
# MAGIC Next, we will set up the Genie space. In "Instructions", add some general instructions on how you want Genie to behave. This could be: 
# MAGIC > When someone asks whether an aircraft is delayed, use the "isArrDelayed" attribute. The values in this column are "YES" or "NO". 
# MAGIC
# MAGIC Depending on which data you are using, you might want to add more instructions. Start asking Genie questions about your dataset and see how it behaves. Use the "Show code" functionality to check whether the code is correct. You can add it to the "SQL queries" tab to add it as a stored querie from which Genie will learn and will use it the next time a user asks the same question.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Orchestrating in a workflow
# MAGIC The last step will be adding this notebook and the dashboard to a simple workflow. In the left navigation bar, select "Job Runs" and then "Create Job". In tasks, you can add this notebook to be refreshed. 

# COMMAND ----------

# MAGIC %md
# MAGIC We are using a widget to define the value of "run_ml". We want to disable this in the workflow. To do that, use the "parameters" option in the Workflows UI.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/Workspace/Users/zuzana.kovacsova@databricks.com/vueling_images/Screenshot 2025-03-27 at 19.36.38.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, add refreshing the dashboard as the next step, ensure it has refresh_notebook as a dependency. Run the pipeline and explore the workflows UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Store workflow in an asset bundle
# MAGIC Now that we have a pipeline, we can store it in an asset bundle to be able to deploy it as code. In this notebook, open the terminal by selecting the terminal widget in the bottom right corner. This will start a session on your cluster. First, we will create a default asset bundle template with `databricks bundle init`. After following the prompts, this will create a new asset bundle in your home directory with the selected resources in the `/resources` directory. The definition of the bundle can be seen in the `databricks.yml `file. 
# MAGIC
# MAGIC Next, we will add the new workflow as an resource to the bundle. Copy the Job ID number from the Job details page in the workflow UI. Next, in the terminal, `cd` into the asset bundle directory and run the following command: 
# MAGIC `databricks bundle generate job --existing-job-id <your job id>`
# MAGIC
# MAGIC
# MAGIC This will add the job as a new resource to the `/resources` folder.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC In this pipeline, we explored data from the Databricks marketplace, trained an ML model, built a dashboard an a Genie space, and linked it up in a job. How else can this data be used, what other data sources can we add? 
