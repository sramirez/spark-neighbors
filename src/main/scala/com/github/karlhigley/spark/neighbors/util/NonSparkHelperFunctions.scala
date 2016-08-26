package com.github.karlhigley.spark.neighbors.util

/**
 * @author Thomas Moerman
 */
object NonSparkHelperFunctions {

  /**
   * Non-Spark counterpart of PairedRDD cogroup.
   */
  def cogroup[K, V](it1: Iterable[(K, V)], it2: Iterable[(K, V)]): Map[K, (Iterable[V], Iterable[V])] = {
    val init = Map[K, (List[V], List[V])]().withDefaultValue((Nil, Nil))

    val temp =
      it1
        .foldLeft(init) {
          case (acc, (product, entry1)) => {
            val (l, r) = acc(product)
            acc + (product -> (entry1 :: l, r))
          }
        }

    val result =
      it2
        .foldLeft(temp) {
          case (acc, (product, entry2)) => {
            val (l, r) = acc(product)
            acc + (product -> (l, entry2 :: r))
          }
        }

    result
  }

  /**
   * Non-Spark counterpart of org.apache.spark.mllib.rdd.MLPairRDDFunctions#topByKey.
   */
  def topByKey[K, V](num: Int, it: Iterable[(K, V)])(implicit ord: Ordering[V]): Map[K, List[V]] = {
    val acc = Map[K, BoundedPriorityQueue[V]]().withDefaultValue(new BoundedPriorityQueue[V](num)(ord))

    it.foreach { case (k, v) => acc(k) += v }

    acc.mapValues(_.toList.sorted(ord.reverse))
  }

}