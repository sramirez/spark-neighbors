package com.github.karlhigley.spark.neighbors.lsh

import scala.collection.immutable.BitSet
import org.apache.spark.mllib.linalg.{ Vector => MLLibVector }
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors

/**
 * This wrapper class allows ANNModel to ignore the
 * type of the hash signatures in its hash tables.
 */
sealed trait Signature[+T] extends Any {
  val elements: T
}

/**
 * Signature type for sign-random-projection LSH
 */
final case class BitSignature(elements: BitSet) extends AnyVal with Signature[BitSet]

/**
 * Signature type for scalar-random-projection LSH
 */
final case class IntSignature(elements: Array[Int]) extends AnyVal with Signature[Array[Int]]

/**
 * A hash table entry containing an id, a signature, and
 * a table number, so that all hash tables can be stored
 * in a single RDD.
 */
sealed abstract class HashTableEntry[+S <: Signature[_]] {

  val id: Long
  val table: Int
  val signature: S
  val point: LabeledPoint
  var membership: Array[Byte]

  def sigElements: Array[Int]
  lazy val norm: Float = Vectors.norm(point.features, 1).toFloat

}

final case class BitHashTableEntry(id: Long,
                                   table: Int,
                                   signature: BitSignature,
                                   point: LabeledPoint) extends HashTableEntry[BitSignature] {

  def sigElements: Array[Int] = signature.elements.toArray
  lazy val size: Int = signature.elements.size

}

final case class IntHashTableEntry(id: Long,
                                   table: Int,
                                   signature: IntSignature,
                                   point: LabeledPoint) extends HashTableEntry[IntSignature] {

  def sigElements: Array[Int] = signature.elements

}