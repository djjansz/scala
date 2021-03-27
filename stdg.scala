//make an array that holds integer elements 0 to 99
val range1 = spark.range(100).toDF("number")
// view the data
range1.collect()
//create an array that holds only the even numbers from range1
val divisBy2 = range1.where("number % 2 = 0")
// view the data
divisBy2.collect()
// read the data and show the first three rows
val flightData2015 = spark.read.option("inferSchema", "true").option("header", "true").csv("/user/sjf/data/flight_data/csv/2015-summary.csv")
flightData2015.show(3)
//explain the plan that Spark has to execute a sort by the count variable
flightData2015.sort("count").explain()
//change the number of shuffle partitions from the default of 200
spark.conf.set("spark.sql.shuffle.partitions", "5")
//sort by count and take the first two rows
flightData2015.sort("count").take(2)
// method call to make a DataFrame into a view that can be queried with SQL
flightData2015.createOrReplaceTempView("v_flight_data_2015")
//the SQL way of doing things
val sqlWay = spark.sql("""
SELECT DEST_COUNTRY_NAME, count(1)
FROM v_flight_data_2015
GROUP BY DEST_COUNTRY_NAME
""")
// show the output from the SQL way of doing things
sqlWay.collect()
//show the explain plan from the SQL way of doing things
sqlWay.explain
// the DataFrame way of doing things
val dataFrameWay = flightData2015.groupBy("DEST_COUNTRY_NAME").count()
//show the output of the dataFrame way of doing things
dataFrameWay.collect()
// show the explain plan from the dataFrame way of doing things
dataFrameWay.explain
// select the max count using spark SQL
spark.sql("SELECT max(count) from v_flight_data_2015").take(1)
//import and use the max function to work on a data frame
import org.apache.spark.sql.functions.max
flightData2015.select(max("count")).take(1)
// the Spark SQL way of showing the top five destination countries by the number of flights taken there
val topFiveSql = spark.sql("""
SELECT DEST_COUNTRY_NAME, sum(count) as destination_total
FROM v_flight_data_2015
GROUP BY DEST_COUNTRY_NAME
ORDER BY sum(count) DESC
LIMIT 5
""")
topFiveSql.show()
topFiveSql.explain()
// the dataFrame way of showing the top five destination countries by the number of flights taken there
import org.apache.spark.sql.functions.desc
flightData2015.groupBy("DEST_COUNTRY_NAME").sum("count").withColumnRenamed("sum(count)", "destination_total").sort(desc("destination_total")).limit(5).show()
// show the explain plan for the dataFrame way of doing things
flightData2015.groupBy("DEST_COUNTRY_NAME").sum("count").withColumnRenamed("sum(count)", "destination_total").sort(desc("destination_total")).limit(5).explain()
import spark.implicits._
// read a parquet file to create a dataFrame - 
val flightsDF = spark.read.parquet("/user/sjf/data/flight_data/parquet/2010_Summary.parquet")
// create a Spark Dataset which is a type-safe version of Spark's structured API
case class Flight(DEST_COUNTRY_NAME: String,ORIGIN_COUNTRY_NAME: String,count: BigInt)
val flights = flightsDF.as[Flight]
// view the dataFrame - it looks more list-like with square brackets
flightsDF.collect()
// view the dataset - it looks more tuple-like with round brackets
flights.collect()
//manipulating the Spark Dataset with functions that come with Spark
flights.filter(flight_row => flight_row.ORIGIN_COUNTRY_NAME == "Canada").map(flight_row => flight_row).take(5)
//calling take on a Dataset at different points in the manipulation
flights.take(5).filter(flight_row => flight_row.ORIGIN_COUNTRY_NAME != "Canada").map(fr => Flight(fr.DEST_COUNTRY_NAME, fr.ORIGIN_COUNTRY_NAME, fr.count + 5))
// create a dataFrame from structured streaming data - there are a lot of csv files in the by_day folder each representing one day of daily retail sales data
val staticDataFrame = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/user/sjf/data/retail_data/by_day/*.csv")
staticDataFrame.createOrReplaceTempView("retail_data")
val staticSchema = staticDataFrame.schema
// read and write streaming data which can be thought of as inputing and outputting a series of files each repesenting one day, minute, or second
import org.apache.spark.sql.functions.{window, column, desc, col}
//change the defualt number of shuffle partitions from 200 down to 5
spark.conf.set("spark.sql.shuffle.partitions", "5")
// streaming code uses the readStream method
val streamingDataFrame = spark.readStream.schema(staticSchema).option("maxFilesPerTrigger", 1).format("csv").option("header", "true").load("/user/sjf/data/retail_data/by_day/*.csv")
// view whether this is set up to stream data
streamingDataFrame.isStreaming // returns true
// create a streaming dataFrame
val purchaseByCustomerPerHour = streamingDataFrame.selectExpr("CustomerId","(UnitPrice * Quantity) as total_cost","InvoiceDate").groupBy($"CustomerId", window($"InvoiceDate", "1 day")).sum("total_cost")
// write to a streaming dataFrame - this step takes a long time to complete
purchaseByCustomerPerHour.writeStream.format("memory").queryName("customer_purchases").outputMode("complete").start()
// run queries against the streaming data
spark.sql("""SELECT * FROM customer_purchases ORDER BY `sum(total_cost)` DESC""").show(5)
// show the column names and data types of the staticDataFrame
staticDataFrame.printSchema()
// we need to transform the staticDataFrame so that it has all numeric values for the ML algorithm
import org.apache.spark.sql.functions.date_format
staticDataFrame.show(100)
//replace null with 0 and make the day_of_week column which shows the day of the week as in Monday, ..., Sunday
val preppedDataFrame = staticDataFrame.na.fill(0).withColumn("day_of_week", date_format($"InvoiceDate", "EEEE")).coalesce(5)
preppedDataFrame.show(100)
// split the data into a training and test sets
val trainDataFrame = preppedDataFrame.where("InvoiceDate < '2011-07-01'")
val testDataFrame = preppedDataFrame.where("InvoiceDate >= '2011-07-01'")
// view the row count of the training and test sets
trainDataFrame.count()
testDataFrame.count()
// use the StringIndexer and OneHotEncoder to transform Monday, Tuesday, ..., Sunday into ..., into 1, 2, .., 7
import org.apache.spark.ml.feature.StringIndexer
val indexer = new StringIndexer().setInputCol("day_of_week").setOutputCol("day_of_week_index")
import org.apache.spark.ml.feature.OneHotEncoder
val encoder = new OneHotEncoder().setInputCol("day_of_week_index").setOutputCol("day_of_week_encoded")
// machine learning algorithms take input as a vector type 
import org.apache.spark.ml.feature.VectorAssembler
val vectorAssembler = new VectorAssembler().setInputCols(Array("UnitPrice", "Quantity", "day_of_week_encoded")).setOutputCol("features")
// set this up into a pipeline so that any future data can go through the same process
import org.apache.spark.ml.Pipeline
val transformationPipeline = new Pipeline().setStages(Array(indexer, encoder, vectorAssembler))
// fit our transformer to the training dataset
val fittedPipeline = transformationPipeline.fit(trainDataFrame)
// use the fitted pipeline to transform the data in a consistent way
val transformedTraining = fittedPipeline.transform(trainDataFrame)
// put a copy of the data into memory so that it can be quickly accessed repeatably
transformedTraining.cache()
// import the model and instantiate it
import org.apache.spark.ml.clustering.KMeans
val kmeans = new KMeans().setK(20).setSeed(1L)
// fit the model on the training data
val kmModel = kmeans.fit(transformedTraining)
// compute the cost diagnostics on our training dataset - the resulting cost on the data is high given that we did not properly pre-process and scale the data
kmModel.computeCost(transformedTraining)
// use an RDD to parallelize some numbers
val rdd1=spark.sparkContext.parallelize(Seq(1, 2, 3,4,5,6,7,8,9,10)).toDF()
// show the values in the rdd
rdd1.collect()
