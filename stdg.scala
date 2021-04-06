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
df.select(lit(5), lit("five"), lit(5.0)).show(10000)
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
df.select(lit(5), lit("five"), lit(5.0)).show(10000)
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
// make an all upper case columna nd an all lower case column
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
// the | signifies `OR` in regular expression syntax - here we replace the colors with the word COLOR
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
// Dates and Times - onnly InvoiceDate is a timestamp in this dataFrame
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
// findiing the days between two dates and the months between two dates
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
