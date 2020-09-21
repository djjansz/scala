//http://localhost:4040/jobs/
//https://github.com/databricks/Spark-The-Definitive-Guide/tree/master/data/flight-data
// create a dataframe with 1000 rows with values 0 to 999
val myRange = spark.range(1000).toDF("number")
// transformation to find all even numbers in our dataframe
val divisBy2 = myRange.where("number % 2 =0")
//read in a csv file
val flightData2015=spark.read.option("inferSchema","true").option("header","true").csv("/home/djjansz/book/data/flight-data/csv/2015-summary.csv")
//view the first five rows after importing the data
flightData2015.take(5)
//view how Spark will execute this query
flightData2015.sort("count").explain()
//transform the data by sorting it and then view the first five rows
flightData2015.sort("count").take(5)
//set the number of partitions to five to reduce the number of output partitions from the shuffle
 spark.conf.set("spark.sql.shuffle.partitions","5")
 flightData2015.take(5)
 
