package com.github.karlhigley.spark.neighbors.lsh

import org.apache.spark.mllib.linalg.{ Vector => MLLibVector, SparseVector }

/**
 * Abstract base class for locality-sensitive hash functions.
 */
abstract class LSHFunction[+T <: Signature[_]] {

  /**
   * Compute the hash signature of the supplied vector
   */
  def signature(v: MLLibVector): T

  /**
   * Build a hash table entry for the supplied vector
   */
  def hashTableEntry(id: Long, table: Int, v: MLLibVector): HashTableEntry[T]

}