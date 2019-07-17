package com.xxx

import breeze.linalg.split
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, Tokenizer}
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

import scala.collection.mutable.WrappedArray

object ldaDemo {
  def main(arg: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("TopicExtraction")
      .getOrCreate()
    import spark.implicits._
    val sourceData= spark.createDataFrame(Seq(
          (0,"soyo spark like spark hadoop spark and spark like spark"),
          (1,"i wish i can like java i"),
          (2,"but i dont know how to soyo"),
          (3,"spark is good spark tool")
        )).toDF("label","sentence")
    val tokenizer=new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val txtDfSplit=tokenizer.transform(sourceData)
    val cvModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .fit(txtDfSplit)
    val txtDfTrain = cvModel.transform(txtDfSplit)
    txtDfTrain.show(false) //show the DataFrame content
    val lda = new LDA().setK(2).setMaxIter(10).fit(txtDfTrain)
    // vocabulary created by CountVectorizer
    val vocab = spark.sparkContext.broadcast(cvModel.vocabulary)
    // describeTopics output:
    lda.describeTopics(4).show
    val toWords = udf( (x : WrappedArray[Int]) => { x.map(i => vocab.value(i)) })
    val tempDF = lda.describeTopics(4)
    val topics = tempDF.withColumn("topicWords", toWords(tempDF("termIndices")))
    topics.select("topicWords").show(false)
    println("topics====================" + topics)
    
    val wordsWithWeights = udf( (x : WrappedArray[Int],
                                 y : WrappedArray[Double]) =>
    { x.map(i => vocab.value(i)).zip(y)}
    )

    val topics2 = tempDF
      .withColumn("topicWords",
        wordsWithWeights(tempDF("termIndices"), tempDF("termWeights"))
      )
    println("topics2开始")
    topics2.show(false)
    println("topics2====================" + topics2)

    val topics2exploded = topics2
      .select("topic", "topicWords")
      .withColumn("topicWords", explode(topics2("topicWords")))
    topics2exploded.show()
    println("topics2exploded====================" + topics2exploded)

    val finalTopic = topics2exploded
      .select(
        topics2exploded("topic"),
        topics2exploded("topicWords").getField("_1").as("word"),
        topics2exploded("topicWords").getField("_2").as("weight")
      )
    finalTopic.show()
    println("finalTopic====================" + finalTopic)

  }
}



