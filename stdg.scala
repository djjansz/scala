//---------------------------------------------------------------------------------//
//                       _______________  ______                                   //
//                      / ___/_  __/ __ \/ ____/                                   //
//                      \__ \ / / / / / / / __                                     //
//                     ___/ // / / /_/ / /_/ /                                     //
//                    /____//_/ /_____/\____/                                      //
//---------------------------------------------------------------------------------//
// an array that holds integer elements 0 to 99
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
// Spark as a progrmming language - performing addition in Spark (not in Scala or Python)
val df = spark.range(15).toDF("number")
df.show()
val df2=df.select(df.col("number") + 10)
df2.show()
// create a row using a range
spark.range(15).toDF().show()
// declare a column to be an IntegerType column
import org.apache.spark.sql.types._
val b = IntegerType
// create a dataframe from a json file
val df = spark.read.format("json").load("/user/sjf/data/flight_data/json/2015-summary.json")
// show the schema
df.printSchema()
// show the schema right on import - shows it horizontally instead of vertically
spark.read.format("json").load("/user/sjf/data/flight_data/json/2015-summary.json").schema
// create and inforce a schema upon the data
import org.apache.spark.sql.types.{StructField, StructType, StringType, LongType}
import org.apache.spark.sql.types.Metadata
val myManualSchema = StructType(Array(
  StructField("DEST_COUNTRY_NAME", StringType, true),
  StructField("ORIGIN_COUNTRY_NAME", StringType, true),
  StructField("count", LongType, false,
    Metadata.fromJson("{\"hello\":\"world\"}"))
))
val df = spark.read.format("json").schema(myManualSchema).load("/user/sjf/data/flight_data/json/2015-summary.json")
//creating a new column from scratch
import org.apache.spark.sql.functions.{col, column}
col("someColumnName")
column("someColumnName")
$"myColumn"
//refer to a specific column
df.col("count")
// an expression - a transformation on one or more values of a record in a DataFrame
(((col("someCol") + 5) * 200) - 6) < col("otherCol")
// in Scala
import org.apache.spark.sql.functions.expr
expr("(((someCol + 5) * 200) - 6) < otherCol")
//use the columns property to see all columns of the dataFrame
spark.read.format("json").load("/user/sjf/data/flight_data/json/2015-summary.json").columns
// see the first row of the DataFrame imported from the json file
df.first()
// creating a row manually
import org.apache.spark.sql.Row
val myRow = Row("Hello", null, 1, false)
// you can't use the show() or collect() action on a row or column, instead you access and show data by position
myRow(0)
//coerce the third element into an integer
myRow.getInt(2) 
// create a dataFrame from a json file
val df = spark.read.format("json").load("/user/sjf/data/flight_data/json/2015-summary.json")
//register the dataFrame as a temporary view to be queried by SQL
df.createOrReplaceTempView("dfTable")
// manually create dataFrame from an RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructField, StructType, StringType, LongType}
val myManualSchema = new StructType(Array(
  new StructField("some", StringType, true),
  new StructField("col", StringType, true),
  new StructField("names", LongType, false)))
val myRows = Seq(Row("Hello", null, 1L))
val myRDD = spark.sparkContext.parallelize(myRows)
val myDf = spark.createDataFrame(myRDD, myManualSchema)
myDf.show()
// turn a sequence into a row of a dataFrame
val myDF = Seq(("Hello", 2, 1L)).toDF("col1", "col2", "col3")
myDF.show()
// select a sepecific column of your dataFrame
df.select("DEST_COUNTRY_NAME", "ORIGIN_COUNTRY_NAME").show(2)
// referring to coumns in different ways
import org.apache.spark.sql.functions.{expr, col, column}
df.select($"DEST_COUNTRY_NAME").show(2)
// rename the column to destination
df.select(expr("DEST_COUNTRY_NAME AS destination")).show(2)
// rename the column to destination and then back to DEST_COUNTRY_NAME
df.select(expr("DEST_COUNTRY_NAME as destination").alias("DEST_COUNTRY_NAME")).show(2)
// showing a column beside itself
df.selectExpr("DEST_COUNTRY_NAME as newColumnName", "DEST_COUNTRY_NAME").show(2)
// selectExpr is the shorthand for the popular select(expr())
df.selectExpr("*", "(DEST_COUNTRY_NAME = ORIGIN_COUNTRY_NAME) as withinCountry").show(100)
// in SQL this would be SELECT avg(count) as avg_count,count(distinct(DEST_COUNTRY_NAME)) as distinct_countries FROM dfTABLE LIMIT 2
df.selectExpr("avg(count) as average_count", "count(distinct(DEST_COUNTRY_NAME)) as distinct_countries").show(2)
// 1 as One using Spark literals
import org.apache.spark.sql.functions.lit
df.select(expr("*"), lit(1).as("One")).show(2)
// 1 as One, with the withColumn method
df.withColumn("One", lit(1)).show(2)
// create a new column named withinCountry 
df.withColumn("withinCountry", expr("ORIGIN_COUNTRY_NAME == DEST_COUNTRY_NAME")).show(100)
// create a new column DEST_COUNTRY_NAME as Destination
df.withColumn("Destination", expr("DEST_COUNTRY_NAME")).columns
// rename DEST_COUNTRY_NAME to dest
df.withColumnRenamed("DEST_COUNTRY_NAME", "dest").columns
// in Scala
import org.apache.spark.sql.functions.expr
//create a long column name with reserved characters like spaces and dashes
val dfWithLongColName = df.withColumn("This Long Column-Name",expr("ORIGIN_COUNTRY_NAME"))
// handling long column names with backticks
dfWithLongColName.selectExpr("`This Long Column-Name`","`This Long Column-Name` as `new col`").show(2)
// create a view dfTableLong
dfWithLongColName.createOrReplaceTempView("dfTableLong")
// select just the one column and use the columns method to show that
dfWithLongColName.select(col("This Long Column-Name")).columns
//drop the ORIGIN_COUNTRY_NAME column
df.drop("ORIGIN_COUNTRY_NAME").columns
// drop two columns
dfWithLongColName.drop("ORIGIN_COUNTRY_NAME", "DEST_COUNTRY_NAME").show()
// convert an integer variable to long with the cast method
df.withColumn("count2", col("count").cast("long")).schema
// use the filter method to filter rows
df.filter(col("count") < 2).show(20)
// use the where method to filter
df.where("count < 2").show(20)
// chaining the where method is better than specifiying multiple AND filters
df.where(col("count") < 2).where(col("ORIGIN_COUNTRY_NAME") =!= "Croatia").show(20)
// in Scala
df.select("ORIGIN_COUNTRY_NAME", "DEST_COUNTRY_NAME").distinct().count()
// use the distinct method on a dataframe to extract the unique values
df.select("ORIGIN_COUNTRY_NAME").distinct().count()
// taking a sample of rows - this method has a 1% chance of picking a row and it picks five of the 256 rows
val seed = 12345
val withReplacement = false
val fraction = 0.01
df.sample(withReplacement, fraction, seed).show()
// sample the dataFrame with a random split - often used to create training, validation and test dataFrames
val dataFrames = df.randomSplit(Array(0.25, 0.75), seed)
//show the row count of the smaller dataFrame
dataFrames(0).count()
//show that row count of the larger dataFrame
dataFrames(1).count()
// in Scala
import org.apache.spark.sql.Row
val schema = df.schema
val newRows = Seq(
  Row("New Country", "Other Country", 5L),
  Row("New Country 2", "Other Country 3", 1L)
)
//create an array that is a flattened version of the table
val parallelizedRows = spark.sparkContext.parallelize(newRows)
//show the values of the parrallized array
parallelizedRows.collect()
// make a new dataFrame from the flattened array and schema description
val newDF = spark.createDataFrame(parallelizedRows, schema)
//use the union method to append the df with the newDF
df.union(newDF).where("count = 1").where($"ORIGIN_COUNTRY_NAME" =!= "United States").show() 
// the different ways to sort a dataFrame
df.sort("count").show(5)
df.orderBy("count", "DEST_COUNTRY_NAME").show(5)
df.orderBy(col("count"), col("DEST_COUNTRY_NAME")).show(5)
// use the asc or desc functions to sort by ascending or descending (asc_nulls_first or desc_nulls_last are other possibilities)
import org.apache.spark.sql.functions.{desc, asc_nulls_first}
df.orderBy(expr("count desc")).show(2)
df.orderBy(desc("count"), asc_nulls_first("DEST_COUNTRY_NAME")).show(2)
// it is sometimes advisable to use the sortWithinPartitions
spark.read.format("json").load("/user/sjf/data/flight_data/json/*-summary.json").sortWithinPartitions("count").show()
//use the limit methond to show the top ten rows
df.limit(10).show()
// show the six largest counts
df.orderBy(expr("count desc")).limit(6).show()
// this shows number of partitions accross the cluster
df.rdd.getNumPartitions 
// repartition will cause another shuffle to occur
df.repartition(5)
// repartition based on a column that would typically be used as an index
df.repartition(col("DEST_COUNTRY_NAME"))
// optionally sepcify the number of partitions
df.repartition(5, col("DEST_COUNTRY_NAME")).coalesce(2)
// in Scala
val collectDF = df.limit(10)
//take flattens your data and requires an integer
collectDF.take(5)
// prints out the dataFrame 
collectDF.show()
//show only the first five rows
collectDF.show(5, false)
//collect rows to the driver - this can crash the driver 
collectDF.collect()
// read the DataFrame that we will use for analysis
val df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/user/sjf/data/retail_data/by_day/2010-12-01.csv")
// view the column names and data types, this is similar to DESCRIBE Table in Spark SQL
df.printSchema()
// convert the dataFrame to a veiew and then DESCRIBE that view
df.createOrReplaceTempView("dfTable")
spark.sql("""DESCRIBE dfTable""").show()
//converting to Spark data types with the lit function - the rows are infinite here
import org.apache.spark.sql.functions.lit
df.select(lit(5), lit("five"), lit(5.0)).show(100)
//showing a literal being created in Spark SQL - only one row is created here
spark.sql("""SELECT 5, "five", 5.0""").show()
//there are different ways of specifying equality - the first example shows the equalTo() method
import org.apache.spark.sql.functions.col
df.where(col("InvoiceNo").equalTo(536365)).select("InvoiceNo", "Description").show(5, true) 
// Scala uses the triple equal sign to express equality
import org.apache.spark.sql.functions.col
df.where(col("InvoiceNo") === 536365).select("InvoiceNo", "Description").show(5, false)
// the simplest method is to specify the predicate as an expression in a string
df.where("InvoiceNo = 536365").show(5, false)
// using not equal to as a string expression
df.where("InvoiceNo <> 536365").show(5, false)
// specifying a boolean column as part of a filter
val priceFilter = col("UnitPrice") > 600
val descripFilter = col("Description").contains("POSTAGE")
df.where(col("StockCode").isin("DOT")).where(priceFilter.or(descripFilter)).show()
//specifyng a boolean column to get the unit prices for the expensive items using a Spark dataFrame
val DOTCodeFilter = col("StockCode") === "DOT"
val priceFilter = col("UnitPrice") > 600
val descripFilter = col("Description").contains("POSTAGE")
df.withColumn("isExpensive", DOTCodeFilter.and(priceFilter.or(descripFilter))).where("isExpensive").select("unitPrice", "isExpensive").show(5)
//specifying a boolean column to get the unit prices for the expensive items using Spark SQL
spark.sql("""SELECT UnitPrice, (StockCode = 'DOT' AND 
(UnitPrice > 600 OR instr(Description, "POSTAGE") >= 1)) as isExpensive 
FROM dfTable
WHERE (StockCode = 'DOT' AND
(UnitPrice > 600 OR instr(Description, "POSTAGE") >= 1))""").show(5)
// read the DataFrame that we will use for analysis
val df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/user/sjf/data/retail_data/by_day/2010-12-01.csv")
// view the column names and data types, this is similar to DESCRIBE Table in Spark SQL
df.printSchema()
// convert the dataFrame to a veiew and then DESCRIBE that view
df.createOrReplaceTempView("dfTable")
spark.sql("""DESCRIBE dfTable""").show()
//converting to Spark data types with the lit function - the rows are infinite here
import org.apache.spark.sql.functions.lit
df.select(lit(5), lit("five"), lit(5.0)).show(100)
//showing a literal being created in Spark SQL - only one row is created here
spark.sql("""SELECT 5, "five", 5.0""").show()
//there are different ways of specifying equality - the first example shows the equalTo() method
import org.apache.spark.sql.functions.col
df.where(col("InvoiceNo").equalTo(536365)).select("InvoiceNo", "Description").show(5, true) 
// Scala uses the triple equal sign to express equality
import org.apache.spark.sql.functions.col
df.where(col("InvoiceNo") === 536365).select("InvoiceNo", "Description").show(5, false)
// the simplest method is to specify the predicate as an expression in a string
df.where("InvoiceNo = 536365").show(5, false)
// using not equal to as a string expression
df.where("InvoiceNo <> 536365").show(5, false)
// specifying a boolean column as part of a filter
val priceFilter = col("UnitPrice") > 600
val descripFilter = col("Description").contains("POSTAGE")
df.where(col("StockCode").isin("DOT")).where(priceFilter.or(descripFilter)).show()
//specifyng a boolean column to get the unit prices for the expensive items using a Spark dataFrame
val DOTCodeFilter = col("StockCode") === "DOT"
val priceFilter = col("UnitPrice") > 600
val descripFilter = col("Description").contains("POSTAGE")
df.withColumn("isExpensive", DOTCodeFilter.and(priceFilter.or(descripFilter))).where("isExpensive").select("unitPrice", "isExpensive").show(5)
//specifying a boolean column to get the unit prices for the expensive items using Spark SQL
spark.sql("""SELECT UnitPrice, (StockCode = 'DOT' AND 
(UnitPrice > 600 OR instr(Description, "POSTAGE") >= 1)) as isExpensive 
FROM dfTable
WHERE (StockCode = 'DOT' AND
(UnitPrice > 600 OR instr(Description, "POSTAGE") >= 1))""").show(5)
//express filters using the dataFrame interface
import org.apache.spark.sql.functions.{expr, not, col}
df.withColumn("isExpensive", not(col("UnitPrice").leq(250)))
.filter("isExpensive")
.select("Description", "UnitPrice").show(5)
// expressing a filter as a SQL statement
df.withColumn("isExpensive", expr("NOT UnitPrice <= 250"))
.filter("isExpensive")
.select("Description", "UnitPrice").show(5)
//to the power of 2 - imagine that we miscounted the stock and the real supply is equal to fabricatedQuantity = (quantity*unit_price)^2 + 5
import org.apache.spark.sql.functions.{expr, pow}
val fabricatedQuantity = round(pow(col("Quantity") * col("UnitPrice"), 2) + 5,1)
df.select(expr("CustomerId"),expr("Quantity"),expr("round(UnitPrice,1)").alias("UnitPrice"), fabricatedQuantity.alias("realQuantity")).show(2)
//using a SQL expression with the dataFrame interface
df.selectExpr("CustomerId","Quantity","UnitPrice","round((POWER((Quantity * UnitPrice), 2.0) + 5),1) as realQuantity").show(2)
//Doing the same thing in Spark SQL 
spark.sql("""SELECT customerId, Quantity, UnitPrice,ROUND((POWER((Quantity * UnitPrice), 2.0) + 5),1) as realQuantity FROM dfTable""").show(2)
// in Scala
import org.apache.spark.sql.functions.{round, bround}
//bround rounds down when it's right on the border
import org.apache.spark.sql.functions.lit
df.select(round(lit("2.5")), bround(lit("2.5"))).show(2)
//  correlation between quantity and unit price is -.041, so when quantity goes up, then the unit price should go down a little bit
import org.apache.spark.sql.functions.{corr}
df.stat.corr("Quantity", "UnitPrice") // correlation calculated using the dataframe stat method
df.select(corr("Quantity", "UnitPrice")).show() // correlation calculated using the corr function for SQL
// the describe method in scala is similar to the summary function in R - it calculates the count, mean, stdev, min and max
df.describe().show()
// Quantity is on the top going from left to right while StockCode goes up and down on the side in this crosstab
df.stat.crosstab("StockCode", "Quantity").show()
// freqItems shows the StockCodes for the most popular items as ranked by the Quantity sold
df.stat.freqItems(Seq("StockCode", "Quantity")).show()
// the monotonically_increasing_id function in Scala is similar to the internal variable _n_ in the SAS DATA Step
import org.apache.spark.sql.functions.monotonically_increasing_id
df.select($"StockCode",monotonically_increasing_id().alias("monotinc")).show(10)
// capitalize every initial word in a string, similar to Excel's PROPCASE function
import org.apache.spark.sql.functions.{initcap}
df.select($"Description",initcap(col("Description"))).show(2, false)
// make an all upper case columna and an all lower case column
import org.apache.spark.sql.functions.{lower, upper}
df.select(col("Description"),lower(col("Description")),upper(lower(col("Description")))).show(2)
// adding or removing spaces around a string with left/right pad/trim
import org.apache.spark.sql.functions.{lit, ltrim, rtrim, rpad, lpad, trim}
df.select(
ltrim(lit("    HELLO    ")).as("ltrim"),
rtrim(lit("    HELLO    ")).as("rtrim"),
trim(lit("    HELLO    ")).as("trim"),
lpad(lit("HELLO"), 3, " ").as("lp"),
rpad(lit("HELLO"), 10, " ").as("rp")).show(2)
// 
import org.apache.spark.sql.functions.regexp_replace
//create a Seq[String] = List(black, white, red, green, blue)
val simpleColors = Seq("black", "white", "red", "green", "blue")
// String = BLACK|WHITE|RED|GREEN|BLUE
val regexString = simpleColors.map(_.toUpperCase).mkString("|")
// the | signifies `OR` in regular expression syntax - where we replace the colors with the word COLOR
df.select(regexp_replace(col("Description"), regexString, "COLOR").alias("color_clean"),col("Description")).show(10,false)
spark.sql("""SELECT regexp_replace(Description, 'BLACK|WHITE|RED|GREEN|BLUE', 'COLOR') as color_clean, Description FROM dfTable""").show(10,false)
// L gets replaced by 1, E gets replaced by 3, and T gets replaced by 7
import org.apache.spark.sql.functions.translate
df.select(translate(col("Description"), "LEET", "1337").alias("color_clean"), col("Description")).show(2,false)
spark.sql("""SELECT translate(Description, 'LEET', '1337'), Description FROM dfTable""").show(2,false)
// using regexp_extract to extract the colors Black, White, Red, Green, or Blue, from the column
import org.apache.spark.sql.functions.regexp_extract
val regexString = simpleColors.map(_.toUpperCase).mkString("(", "|", ")")
df.select(regexp_extract(col("Description"), regexString, 1).alias("color_clean"),col("Description")).show(10,false)
spark.sql("""SELECT regexp_extract(Description, '(BLACK|WHITE|RED|GREEN|BLUE)', 1) as color_clean, Description FROM dfTable""").show(10,false)
// Using the contains method to filter out only rows that contain the text BLACK or WHITE
val containsBlack = col("Description").contains("BLACK")
val containsWhite = col("DESCRIPTION").contains("WHITE")
df.withColumn("hasSimpleColor", containsBlack.or(containsWhite)).where("hasSimpleColor").select("Description").show(50, false)
spark.sql("""SELECT Description FROM dfTable WHERE instr(Description, 'BLACK') >= 1 OR instr(Description, 'WHITE') >= 1""").show(50,false)
// filter out only the rows that contain the text BLACK or WHITE in a more dynamic way
val simpleColors = Seq("black", "white", "red", "green", "blue")
val selectedColumns = simpleColors.map(color => {col("Description").contains(color.toUpperCase).alias(s"is_$color")}):+expr("*") 
df.select(selectedColumns:_*).where(col("is_white").or(col("is_black"))).select("Description").show(30, false)
// Dates and Times - only InvoiceDate is a timestamp in this dataFrame
df.printSchema()
// create a dataFrame with 10 rows of the same thing which is the current date and the current time
import org.apache.spark.sql.functions.{current_date, current_timestamp}
val dateDF = spark.range(10).withColumn("today", current_date()).withColumn("now", current_timestamp())
dateDF.createOrReplaceTempView("dateTable")
dateDF.show(10)
dateDF.printSchema()
// today -5 days, today + 5 days
import org.apache.spark.sql.functions.{date_add, date_sub}
dateDF.select(date_sub(col("today"), 5), date_add(col("today"), 5)).show(1)
spark.sql("""SELECT date_sub(today, 5), date_add(today, 5) FROM dateTable""").show(1)
// finding the days between two dates and the months between two dates
import org.apache.spark.sql.functions.{datediff, months_between, to_date}
dateDF.withColumn("week_ago", date_sub(col("today"), 7)).select(datediff(col("week_ago"), col("today"))).show(1)
dateDF.select(to_date(lit("2016-01-01")).alias("start"),to_date(lit("2017-05-22")).alias("end")).select(months_between(col("start"), col("end"))).show(1)
spark.sql("""SELECT to_date('2016-01-01'), months_between('2016-01-01', '2017-01-01'),datediff('2016-01-01', '2017-01-01') FROM dateTable""").show(1)
// the to_date functions converts a string into a date
import org.apache.spark.sql.functions.{to_date, lit}
spark.range(5).withColumn("date", lit("2017-01-01")).select(to_date(col("date"))).show(1)
// this shows the null that is generated when the to_date is ommitted
dateDF.select(to_date(lit("2016-20-12")),to_date(lit("2017-12-11"))).show(1)
// the dateFormat as an arugument for the to_date function to tell Spark that the format is year, day, month
import org.apache.spark.sql.functions.to_date
val dateFormat = "yyyy-dd-MM" 
val cleanDateDF = spark.range(1).select(to_date(lit("2017-12-11"), dateFormat).alias("date"),to_date(lit("2017-20-12"), dateFormat).alias("date2"))
cleanDateDF.createOrReplaceTempView("dateTable2")
cleanDateDF.show(1)
spark.sql("""SELECT to_date(date, 'yyyy-dd-MM'), to_date(date2, 'yyyy-dd-MM'), to_date(date) FROM dateTable2""").show(1)
// changing the date to a timestamp - the timestamp function requires that a format be specified
import org.apache.spark.sql.functions.to_timestamp
cleanDateDF.select(to_timestamp(col("date"), dateFormat)).show()
spark.sql("""SELECT to_timestamp(date, 'yyyy-dd-MM'), to_timestamp(date2, 'yyyy-dd-MM') FROM dateTable2""").show()
// filtering with dates - the filter is interpreted as yyyy-mm-dd here
cleanDateDF.filter(col("date2") > lit("2017-12-20")).show()
// filtering as a string which Spark parses to a literal does not work on date variables
cleanDateDF.filter(col("date2") > "'2017-12-20'").show()
// the coalese function selects the first non-null values from a set of columns
import org.apache.spark.sql.functions.coalesce
df.select(coalesce(col("Description"), col("CustomerId"))).show()
// drop if any of the rows are all NULL
df.na.drop("all", Seq("StockCode", "InvoiceNo")).show(1)
// replace NULL values in columns of type string
df.na.fill("All Null values become this string").show(1)
// replace NULL values with 5
df.na.fill(5, Seq("StockCode", "InvoiceNo"))
// using a Scala map to fill in NULL values
val fillColValues = Map("StockCode" -> 5, "Description" -> "No Value")
df.na.fill(fillColValues).show(1)
// complex structures - the struct is a set of columns that are dealt together
import org.apache.spark.sql.functions.struct
val complexDF = df.select(struct("Description", "InvoiceNo").alias("complex"))
complexDF.createOrReplaceTempView("complexDF")
// Using the dot notation and the getfield method to work with the struct
complexDF.select("complex.Description").show(1)
complexDF.select(col("complex").getField("Description")).show(1,false)
//show the complex Scala map as a column
spark.sql("""SELECT map(Description, InvoiceNo) as complex_map FROM dfTable WHERE Description IS NOT NULL""").show(5,false)
//select all values in the strct with *
complexDF.select("complex.*").show(1,false)
// using an array to split text like the words in a book
import org.apache.spark.sql.functions.split
df.select(split(col("Description"), " ")).show(20,false)
df.select(split(col("Description"), " ").alias("array_col")).selectExpr("array_col[0]").show(20,false)
// create an exploded column that contains one row per word in a Description
spark.sql("""SELECT Description, InvoiceNo, exploded FROM (SELECT *, split(Description, " ") as splitted FROM dfTable) LATERAL VIEW explode(splitted) as exploded""").show(10)
// Check whether each row contains white using arrays
import org.apache.spark.sql.functions.array_contains
df.select(array_contains(split(col("Description"), " "), "WHITE")).show(20,false)
// user-defined function to raise to the power of three
val udfExampleDF = spark.range(5).toDF("num")
def power3(number:Double):Double = number * number * number
power3(2.0)
// import the user defined functions package and register the power3udf function in metadata
import org.apache.spark.sql.functions.udf
val power3udf = udf(power3(_:Double):Double)
udfExampleDF.select(power3udf(col("num"))).show()
// register the power3 function as a Spark SQL function
spark.udf.register("power3", power3(_:Double):Double)
udfExampleDF.selectExpr("power3(num)").show(2)
spark.sql("""SELECT power3(Quantity) as Quantity3,Quantity from dfTable""").show(10,false)
// read in the data, repartitoining the data to have fewer partitions as these are small files
val df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/user/sjf/data/retail_data/all/*.csv").coalesce(5)
df.cache() // cache the results for faster access
df.createOrReplaceTempView("dfTable") 
// the dataframe count method is an action that is eagerly evaluated (not lazily)
df.count() == 541909
// the count aggregation function
import org.apache.spark.sql.functions.count
// the count function does not count nulls if a column is specified
df.select(count("Description")).show() // 540,455
// total row count - count star counts all rows including those that have all columns null
spark.sql("""SELECT COUNT(*) FROM dfTable""").show() // 541,909
//unique row count - count distinct star counts number of distinct rows
spark.sql("""SELECT COUNT(DISTINCT *) FROM dfTable""").show() //401,604
// the countDistinct aggregation function
import org.apache.spark.sql.functions.countDistinct
df.select(countDistinct("StockCode")).show() // 4,070
spark.sql("""SELECT COUNT(DISTINCT StockCode) as distinct_stockcodes FROM dfTable""").show() // 4,070
// approx_count_distinct(col) will save time versus count(disctinct col), although it is not as accurate
import org.apache.spark.sql.functions.approx_count_distinct
df.select(approx_count_distinct("StockCode", 0.1)).show() // 3,364
spark.sql("""SELECT approx_count_distinct(StockCode, 0.1) FROM DFTABLE""").show() // 3,364
// the first and last aggregation functions
import org.apache.spark.sql.functions.{first, last}
df.select(first("StockCode"), last("StockCode")).show() 
spark.sql("""SELECT first(StockCode), last(StockCode) FROM dfTable""").show()
// the min and max aggregation functions
import org.apache.spark.sql.functions.{min, max}
df.select(min("Quantity"), max("Quantity")).show() 
spark.sql("""SELECT min(Quantity), max(Quantity) FROM dfTable""").show()
// the sum aggregation function
import org.apache.spark.sql.functions.sum
df.select(sum("Quantity")).show() // 5,176,450
spark.sql("""SELECT sum(Quantity) FROM dfTable""").show() // 5,176,450
// the sumDistinct aggregation function
import org.apache.spark.sql.functions.sumDistinct
df.select(sumDistinct("UnitPrice")).show() // 611,388.39
spark.sql("""SELECT SUM(Distinct UnitPrice) FROM dfTable""").show() // 611,388.39
spark.sql("""SELECT SUM(UnitPrice) FROM dfTable""").show() // 2,498,803.97
// the avg aggregation function
import org.apache.spark.sql.functions.{sum, count, avg, expr}
// the avg is equal to the mean which is equal to the sum divided by the count
df.select(count("Quantity").alias("total_transactions"),sum("Quantity").alias("total_purchases"),avg("Quantity").alias("avg_purchases"),expr("mean(Quantity)").alias("mean_purchases")).selectExpr("total_purchases/total_transactions","avg_purchases","mean_purchases").show()
// the variance and standard deviation aggregation functions
import org.apache.spark.sql.functions.{var_pop, stddev_pop} // pop is short for population - the population stdev divides by n
import org.apache.spark.sql.functions.{var_samp, stddev_samp} // samp is short for sample - the sample stdev divides by n-1
df.select(var_pop("Quantity"), var_samp("Quantity")).show() // 47,559.3036 and 47,559.3914 - the sample variance is larger
spark.sql("""SELECT var_pop(Quantity), var_samp(Quantity) FROM dfTable""").show() // 47,559.3036 and 47,559.3914 - the sample variance is larger
df.select(stddev_pop("Quantity"), stddev_samp("Quantity")).show() //   218.0810 and  218.0812 - the sample standard deviation is larger
spark.sql("""SELECT stddev_pop(Quantity), stddev_samp(Quantity) FROM dfTable""").show()  //   218.0810 and  218.0812 - the sample standard deviation is larger
// the skewness and kurtosis aggregate functions
import org.apache.spark.sql.functions.{skewness, kurtosis}
df.select(skewness("Quantity"), kurtosis("Quantity")).show() // -0.2642 and 119,768.05496
spark.sql("""SELECT skewness(Quantity), kurtosis(Quantity) FROM dfTable""").show() // -0.2642 and 119,768.05496
// the correlation and covariance aggregation functions 
import org.apache.spark.sql.functions.{corr, covar_pop, covar_samp}
df.select(corr("InvoiceNo", "Quantity")).show() //.000491
spark.sql("""SELECT corr(InvoiceNo, Quantity) FROM dfTable""").show() //.000491
df.select(covar_pop("InvoiceNo", "Quantity"),covar_samp("InvoiceNo", "Quantity")).show() //1,052.7261 and 1,052.7281 - the sample covariance is larger
spark.sql("""SELECT covar_pop(InvoiceNo, Quantity),covar_samp(InvoiceNo, Quantity) FROM dfTable""").show() //1,052.7261 and 1,052.7281 - the sample covariance is larger
// in Scala
import org.apache.spark.sql.functions.{collect_set, collect_list}
df.agg(collect_set("Country"), collect_list("Country")).show() //collect_set de-dupes the data and creates a unique list while collect_list does not
df.select(countDistinct("Country")).show() // 38
spark.sql("""SELECT collect_set(Country) FROM dfTable""").show(1,false) // returns an array that holds the 38 countries
spark.sql("""SELECT collect_list(Country) FROM dfTable""").show(1) // returns a giant array that extends past column 8 million of the text file if show(1,false) is added
// the groupBy aggregation function
df.groupBy("InvoiceNo", "CustomerId").count().show() //shows the row count for each combination of InvoiceNo and CustomerId
// grouping with expressions
import org.apache.spark.sql.functions.count
df.groupBy("InvoiceNo").agg(count("Quantity").alias("quan"),expr("count(Quantity)")).show()
// grouping with maps
df.groupBy("InvoiceNo").agg("Quantity"->"avg", "Quantity"->"stddev_pop").sort("InvoiceNo").show()
spark.sql("""SELECT  InvoiceNo, avg(Quantity), stddev_pop(Quantity) FROM dfTable GROUP BY InvoiceNo ORDER BY InvoiceNo""").show()
// convert the InvoiceDate to a date column in preparation of working with Window functions
import org.apache.spark.sql.functions.{col, to_date}
val dfWithDate = df.withColumn("date", to_date(col("InvoiceDate"), "MM/d/yyyy H:mm"))
dfWithDate.show(10,false)
dfWithDate.createOrReplaceTempView("dfWithDate")
// Window functions
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.col
// partition by describes how we will break up the group, order determines ordering within a partition, the towsBetween states which rows will be included based on its reference to the current input row
val windowSpec = Window.partitionBy("CustomerId", "date").orderBy(col("Quantity").desc).rowsBetween(Window.unboundedPreceding, Window.currentRow)
import org.apache.spark.sql.functions.max
val maxPurchaseQuantity = max(col("Quantity")).over(windowSpec)
// the dense_rank() function is used to find the date at which the customer purchased the most quantity
import org.apache.spark.sql.functions.{dense_rank, rank}
val purchaseDenseRank = dense_rank().over(windowSpec) //dense rank has no breaks in rank, for example rank 1,1,2,3 (there is a tie for first place)
val purchaseRank = rank().over(windowSpec) // rank has breaks in the rank, for example rank 1,1,3,4 (there is a tie for first place)
import org.apache.spark.sql.functions.col
dfWithDate.where("CustomerId IS NOT NULL").orderBy("CustomerId").select(col("CustomerId"),col("date"),col("Quantity"),purchaseRank.alias("quantityRank"),purchaseDenseRank.alias("quantityDenseRank"),maxPurchaseQuantity.alias("maxPurchaseQuantity")).show(100,false)
spark.sql("""SELECT CustomerId, date, Quantity,rank(Quantity) OVER (PARTITION BY CustomerId, date ORDER BY Quantity DESC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as rank,
dense_rank(Quantity) OVER (PARTITION BY CustomerId, date ORDER BY Quantity DESC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as dRank,
max(Quantity) OVER (PARTITION BY CustomerId, date ORDER BY Quantity DESC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as maxPurchase
FROM dfWithDate WHERE CustomerId IS NOT NULL ORDER BY CustomerId """).show(100,false)
//Rollups - similar to multilabels with proc report to see different sums in the same aggregation table
val dfNoNull = dfWithDate.drop()
dfNoNull.createOrReplaceTempView("dfNoNull")
val rolledUpDF = dfNoNull.rollup("Date", "Country").agg(sum("Quantity")).selectExpr("Date", "Country", "`sum(Quantity)` as total_quantity").orderBy("Date")rolledUpDF.show()
// The NULL row here is the sum over all countries for a given date
rolledUpDF.where("Country IS NULL").show()
// the NULL row here shows the sum over all dates and countries
rolledUpDF.where("Date IS NULL").show()
// the cube rollsup over all dimensions similar to proc summary without the nway option to eliminate intermediate levels of summarization 
dfNoNull.cube("Date", "Country").agg(sum(col("Quantity"))).select("Date", "Country", "sum(Quantity)").orderBy("Date").show()
import org.apache.spark.sql.functions.{grouping_id, sum, expr}
// grouping metadata - the grouping id is 3 - highest-level aggregation, 2 - aggregations of individual stockCodes, 1 - on a per customer basis, 0 - total quantity by customerID and stockCode
dfNoNull.cube("customerId", "stockCode").agg(grouping_id(), sum("Quantity")).orderBy(expr("grouping_id()").desc).show()
// Pivots - transposing the data
val pivoted = dfWithDate.groupBy("date").pivot("Country").sum()
pivoted.printSchema
pivoted.where("date > '2011-12-05'").select("date" ,"`USA_sum(Quantity)`").show()
// create some datasets manually - for use in the Join examples
val person = Seq((0, "Bill Chambers", 0, Seq(100)),(1, "Matei Zaharia", 1, Seq(500, 250, 100)),(2, "Michael Armbrust", 1, Seq(250, 100))).toDF("id", "name", "graduate_program", "spark_status")
val graduateProgram = Seq((0, "Masters", "School of Information", "UC Berkeley"),(2, "Masters", "EECS", "UC Berkeley"),(1, "Ph.D.", "EECS", "UC Berkeley")).toDF("id", "degree", "department", "school")
val sparkStatus = Seq((500, "Vice President"),(250, "PMC Member"),(100, "Contributor")).toDF("id", "status")
// register these tables so that they can be used with Spark SQL
person.createOrReplaceTempView("person")
graduateProgram.createOrReplaceTempView("graduateProgram")
sparkStatus.createOrReplaceTempView("sparkStatus")
// view the datasets created manually
person.show(5)
graduateProgram.show(5)
sparkStatus.show(5)
// inner join - join the person dataframe to the graduateProgram dataframe by id
val joinExpression = person.col("graduate_program") === graduateProgram.col("id")
person.join(graduateProgram, joinExpression).show()  // the default is inner join 
var joinType = "inner"
person.join(graduateProgram, joinExpression, joinType).show()
spark.sql("""SELECT * FROM person JOIN graduateProgram ON person.graduate_program = graduateProgram.id""").show()
// outer join
joinType = "outer"
person.join(graduateProgram, joinExpression, joinType).show()
spark.sql("""SELECT * FROM person FULL OUTER JOIN graduateProgram ON graduate_program = graduateProgram.id""").show()
// left outer join
joinType = "left_outer"
graduateProgram.join(person, joinExpression, joinType).show()
spark.sql("""SELECT * FROM graduateProgram LEFT OUTER JOIN person ON person.graduate_program = graduateProgram.id""").show()
// right outer join
joinType = "right_outer"
person.join(graduateProgram, joinExpression, joinType).show()
spark.sql("""SELECT * FROM person RIGHT OUTER JOIN graduateProgram  ON person.graduate_program = graduateProgram.id""").show()
// left semi joins - if the right dataframees values exist in the left, these rows are kept in the result
joinType = "left_semi"
graduateProgram.join(person, joinExpression, joinType).show()
val gradProgram2 = graduateProgram.union(Seq((0, "Masters", "Duplicated Row", "Duplicated School")).toDF())
gradProgram2.createOrReplaceTempView("gradProgram2")
gradProgram2.join(person, joinExpression, joinType).show()
spark.sql("""SELECT * FROM gradProgram2 LEFT SEMI JOIN person ON gradProgram2.id = person.graduate_program""").show()
// left anti joins - keeps only the values that do not exist in the right dataframe
joinType = "left_anti"
graduateProgram.join(person, joinExpression, joinType).show()
spark.sql("""SELECT * FROM graduateProgram LEFT ANTI JOIN person ON graduateProgram.id = person.graduate_program""").show()
// cross joins aka cartesian products
joinType = "cross"
graduateProgram.join(person, joinExpression, joinType).show()
person.crossJoin(graduateProgram).show() 
spark.sql("""SELECT * FROM graduateProgram CROSS JOIN person ON graduateProgram.id = person.graduate_program""").show()
// Joins on complex types
import org.apache.spark.sql.functions.expr
person.withColumnRenamed("id", "personId").join(sparkStatus, expr("array_contains(spark_status, id)")).show()
spark.sql("""SELECT * FROM (select id as personId, name, graduate_program, spark_status FROM person) INNER JOIN sparkStatus ON array_contains(spark_status, id)""").show()
// handling duplicate column names
val gradProgramDupe = graduateProgram.withColumnRenamed("id", "graduate_program")
val joinExpr = gradProgramDupe.col("graduate_program") === person.col("graduate_program")
person.join(gradProgramDupe, joinExpr).show() // does not throw an error
// person.join(gradProgramDupe, joinExpr).select("graduate_program").show() //throws an error saying 'graduate_program' is ambiguous, could be: graduate_program, graduate_program
// approach 1 - use a string as the join expression, this automatically removes one of the columns for you 
person.join(gradProgramDupe,"graduate_program").select("graduate_program").show()
// approach 2 - drop one of the offending columns after the join
person.join(gradProgramDupe, joinExpr).drop(person.col("graduate_program")).select("graduate_program").show()
val joinExpr = person.col("graduate_program") === graduateProgram.col("id")
person.join(graduateProgram, joinExpr).drop(graduateProgram.col("id")).show()
// approach 3 - rename the column before the join
val gradProgram3 = graduateProgram.withColumnRenamed("id", "grad_id")
val joinExpr = person.col("graduate_program") === gradProgram3.col("grad_id")
person.join(gradProgram3, joinExpr).show()
// Spark has set this up as a broadcast join from looking at the explain plain
val joinExpr = person.col("graduate_program") === graduateProgram.col("id")
person.join(graduateProgram, joinExpr).explain()
// give the optimizer a hint that we want to use a broadcast join
import org.apache.spark.sql.functions.broadcast
val joinExpr = person.col("graduate_program") === graduateProgram.col("id")
person.join(broadcast(graduateProgram), joinExpr).explain()
// read in the data without defining a schema
spark.read.format("csv").option("header", "true").option("mode", "FAILFAST").option("inferSchema", "true").load("/user/sjf/data/flight_data/csv/2010-summary.csv").show(2)
// import a csv file and manually define a schema
import org.apache.spark.sql.types.{StructField, StructType, StringType, LongType}
val myManualSchema = new StructType(Array(new StructField("DEST_COUNTRY_NAME", StringType, true),new StructField("ORIGIN_COUNTRY_NAME", StringType, true),new StructField("count", LongType, false)))
spark.read.format("csv").option("header", "true").option("mode", "FAILFAST").schema(myManualSchema).load("/user/sjf/data/flight_data/csv/2010-summary.csv").show(5)
// read in a comma separated values file and write a tab separated values file
val csvFile = spark.read.format("csv").option("header", "true").option("mode", "FAILFAST").schema(myManualSchema).load("/user/sjf/data/flight_data/csv/2010-summary.csv")
csvFile.write.format("csv").mode("overwrite").option("sep", "\t").save("/user/sjf/data/flight_data/tsv/2010-summary.tsv")
// read a json file and write it to a csv file - the output is a directory that contains a randomly named file that holds the csv data
val jsonFile = spark.read.format("json").option("mode", "FAILFAST").schema(myManualSchema).load("/user/sjf/data/flight_data/json/2010-summary.json")
jsonFile.write.format("csv").option("mode", "OVERWRITE").option("dateFormat", "yyyy-MM-dd").option("path", "/user/sjf/data/flight_data/csv/2010-summary.csv").save()
// read a parquet file and then write it to a parquet file that is once again stored in a folder in the directory of the parquet name
spark.read.format("parquet").load("/user/sjf/data/flight_data/parquet/2010-summary.parquet").show(5)
csvFile.write.format("parquet").mode("overwrite").save("/user/sjf/data/flight_data/parquet/2010-summary.parquet")
// DF => SQL
spark.read.json("/user/sjf/data/flight_data/json/2010-summary.json").createOrReplaceTempView("2010_summary") 
// SQL => DF
spark.sql("""SELECT DEST_COUNTRY_NAME, sum(count) FROM 2010_summary GROUP BY DEST_COUNTRY_NAME """).where("DEST_COUNTRY_NAME like 'S%'").where("`sum(count)` > 10").count() 
// read in a json file and create a dataframe and then a view
val flights = spark.read.format("json").load("/user/sjf/data/flight_data/json/2015-summary.json")
val just_usa_df = flights.where("dest_country_name = 'United States'")
just_usa_df.createOrReplaceTempView("just_usa_view")
// show the explain plan using the dataFrame API
just_usa_df.selectExpr("*").explain
// show the explain plan using Spark SQL programming
spark.sql("""Explain select * from just_usa_view""").show(1,false)
//show the databases avaliable
spark.sql("""SHOW DATABASES""").show(20,false)
// cache the view
spark.sql("""CACHE TABLE JUST_USA_VIEW""").show(1,false)
// unchache the view
spark.sql("""UNCACHE TABLE JUST_USA_VIEW""").show(1,false)
// describe table versus printSchema
spark.sql("""DESCRIBE TABLE 2010_summary""").show()
// show tables in ppadev like sjf
spark.sql("""SHOW TABLES IN PPADEV like 'SJF*'""").show(100,false)
// view system functions
spark.sql("""SHOW SYSTEM FUNCTIONS""").show(100,false)
// create a user defined function for Spark SQL
def power3(number:Double):Double = number * number * number
spark.udf.register("power3", power3(_:Double):Double)
spark.sql("""SELECT count, power3(count) as count_cubed FROM 2010_summary""").show()
