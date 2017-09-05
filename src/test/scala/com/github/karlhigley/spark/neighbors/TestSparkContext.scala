package com.github.karlhigley.spark.neighbors

import org.scalatest.{ BeforeAndAfterAll, Suite }

import org.apache.spark.{ SparkConf, SparkContext }

trait TestSparkContext extends BeforeAndAfterAll { self: Suite =>
  @transient var sc: SparkContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName("LshUnitTest")
    sc = new SparkContext(conf)
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }

}