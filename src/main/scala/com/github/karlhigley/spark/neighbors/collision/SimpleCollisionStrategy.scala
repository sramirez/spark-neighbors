package com.github.karlhigley.spark.neighbors.collision

import com.github.karlhigley.spark.neighbors.lsh.HashTableEntry
import org.apache.spark.rdd.RDD

import scala.util.hashing.MurmurHash3
import com.github.karlhigley.spark.neighbors.ANNModel.Point

/**
 * A very simple collision strategy for candidate identification
 * based on an OR-construction between hash functions/tables
 *
 * (See Mining Massive Datasets, Ch. 3)
 */
object SimpleCollisionStrategy extends CollisionStrategy with Serializable {

  /**
   * Convert hash tables into an RDD that is "collidable" using groupByKey.
   * The new keys contain the hash table id, and a hashed version of the signature.
   */
  def apply(hashTables: RDD[_ <: HashTableEntry[_]]): RDD[(Product, Point)] =
    hashTables
      .map(entry => {
        // Arrays are mutable and can't be used in RDD keys. Use a hash value (i.e. an int) as a substitute
        val key = (entry.table, MurmurHash3.arrayHash(entry.sigElements)).asInstanceOf[Product]

        (key, (entry.id, entry.point))
      })

}