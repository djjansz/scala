package org.economical.demo
import org.apache.spark.sql.functions._
import org.economical.SparkCommon

object sstdg_program1 extends SparkCommon {
  def main(argv: Array[String]): Unit = {
    val srcpath = "/user/sjf/stdg/"
    val ss = sparkSession

    val success = scala.util.Try(ss.read.
      format("csv").
      option("header", "true").
      load(srcpath + "2015-summary.csv")).isSuccess
    if (success) {
      val df = ss.read.
        format("csv").
        option("header", "true").
        load(srcpath + "2015-summary.csv")
      val colnames = df.columns
      println(colnames)

      val flightData2015 = df.select(
        col("DEST_COUNTRY_NAME").alias("DEST_COUNTRY_NAME1"),
        col("ORIGIN_COUNTRY_NAME").alias("ORIGIN_COUNTRY_NAME1"),
        col("Count").alias("count")
      )
      //show the flight data
      flightData2015.show(5)
      //sort the flight data
      flightData2015.sort("count").explain()
      // transformation to find all even count numbers in our dataframe
      val divisBy2 = flightData2015.where("count % 2 =0")
      divisBy2.show(10)
    }
  }
}
