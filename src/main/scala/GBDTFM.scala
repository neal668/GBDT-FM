// $example on$
import org.apache.spark.mllib.regression._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, FeatureType, Strategy}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, Node}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
// $example off$


/**
 * Created by zrf on 4/18/15.
  * Refined by TZH on 14/02,2017 for GBDT+FM
 */


object GBDTFM extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)


  override def main(args: Array[String]): Unit = {

    //get decision tree leaf's nodes
    def getLeafNodes(node: Node): Array[Int] = {
      var treeLeafNodes = new Array[Int](0)
      if (node.isLeaf) {
        treeLeafNodes = treeLeafNodes.:+(node.id)
      } else {
        treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.leftNode.get)
        treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.rightNode.get)
      }
      treeLeafNodes
    }

    // predict decision tree leaf's node value
    def predictModify(node: Node, features: DenseVector): Int = {
      val split = node.split
      if (node.isLeaf) {
        node.id
      } else {
        if (split.get.featureType == FeatureType.Continuous) {
          if (features(split.get.feature) <= split.get.threshold) {
            //          println("Continuous left node")
            predictModify(node.leftNode.get, features)
          } else {
            //          println("Continuous right node")
            predictModify(node.rightNode.get, features)
          }
        } else {
          if (split.get.categories.contains(features(split.get.feature))) {
            //          println("Categorical left node")
            predictModify(node.leftNode.get, features)
          } else {
            //          println("Categorical right node")
            predictModify(node.rightNode.get, features)
          }
        }
      }
    }


    //val conf = new SparkConf().setAppName("GBDTFM").setMaster("local[*]").set("spark.kryoserializer.buffer.max", "256m").set("spark.executor.memory", "1g").set("spark.driver.memory", "1g")
    val conf = new SparkConf().setAppName("GBDTFM").set("spark.kryoserializer.buffer.max","256m").set("spark.executor.memory", "2g").set("spark.driver.memory", "2g") // HDFS Version
    val sc = new SparkContext(conf)
    // $example on$

    /* Load and parse the data file.*/
    //val path = "data/train.d_label"   // labelled data
    val path = args(0)
    //val data = MLUtils.loadLibSVMFile(sc, path)    // LIBSVM format

    /** Transforming text data, split by space to Labelled Point */
    //val path = "data/20W.d"
    // text data
    val dataSet = sc.textFile(path).map(x => x.trim.split(","))
    // text format
    val data = dataSet.map { x =>
      val y = x.slice(1, x.length).map(_.toDouble)
      LabeledPoint(x(0).toDouble, Vectors.dense(y))
    }

    val time = args(1).toInt
    //println(data.collect().map(x=>x.features))
    val precise = new Array[Double](time)
    // prediction accuracy
    val NE_sum = new Array[Double](time) // Normalized Entropy

    for (i <- 0 until time) {
      val splits = data.randomSplit(Array(0.7, 0.3))
      val (trainingData, testData) = (splits(0), splits(1))
      println("the number of training data is " + trainingData.count())
      println("the number of testing data is " + testData.count())

      /** Train a GradientBoostedTrees model.
        * The defaultParams for Classification use LogLoss by default.
        */

      //val numTrees = args(1).toInt
      val numTrees = args(2).toInt
      val boostingStrategy = BoostingStrategy.defaultParams("Classification")
      boostingStrategy.setNumIterations(numTrees)
      val treeStratery = Strategy.defaultStrategy("Classification")
      treeStratery.setMaxDepth(5) // depth of tree
      treeStratery.setNumClasses(2) // binary classification
      boostingStrategy.setTreeStrategy(treeStratery)
      val gbdtModel = GradientBoostedTrees.train(trainingData, boostingStrategy)
      //val gbdtModelDir = "model"
      //gbdtModel.save(sc, gbdtModelDir)
      //val gbdtModel = GradientBoostedTreesModel.load(sc,gbdtModelDir)

      val labelAndPreds = testData.map { point =>
        val prediction = gbdtModel.predict(point.features)
        (point.label, prediction)
      }
      val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
      println("Test Error = " + testErr)
      //println("Learned classification GBT model:\n" + gbdtModel.toDebugString) // information of Model, how to split

      /** make 0,1 high dimension feature data */
      val treeLeafArray = new Array[Array[Int]](numTrees)
      for (i <- 0.until(numTrees)) {
        treeLeafArray(i) = getLeafNodes(gbdtModel.trees(i).topNode) // the leaf node of each point
        //println(treeLeafArray(i).foreach(x => print(x + " ")))
      }

      //val transdata = MLUtils.loadLibSVMFile(sc, path)   // LibSVM Format
      val transdata = sc.textFile(path).map(x => x.trim.split(",")).map {
        x =>
        val y = x.slice(1, x.length).map(_.toDouble)
        LabeledPoint(x(0).toDouble, Vectors.dense(y))
      }  // text Format

      transdata.map { x =>
        var newFeature = new Array[Double](0)
        // for each tree, the data is trained again to get the transformed data
        for (i <- 0.until(numTrees)) {
          val treePredict = predictModify(gbdtModel.trees(i).topNode, x.features.toDense)
          //gbdt tree is binary tree
          val treeArray = new Array[Double]((gbdtModel.trees(i).numNodes + 1) / 2)
          //println(gbdtModel.trees(i).numNodes)
          treeArray(treeLeafArray(i).indexOf(treePredict)) = 1
          newFeature = newFeature ++ treeArray
          //println(newFeature.foreach(x => print(x + " ")))
        }
        (x.label, newFeature)
      }
      //transdata.saveAsTextFile("")


      val splits2 = transdata.randomSplit(Array(0.7, 0.3))
      val train2 = splits2(0).cache()
      val test2 = splits2(1)
      println("the number of training data is " + train2.count())
      println("the number of testing data is " + test2.count())


      //    val task = args(1).toInt
      //    val numIterations = args(2).toInt
      //    val stepSize = args(3).toDouble
      //    val miniBatchFraction = args(4).toDouble

      /** task = 0 for regression; task = 1 for binary classification, Factorization Machine **/
      // train a model
      val fm1 = FMWithSGD.train(trainingData, task = 0, numIterations = args(3).toInt, stepSize = args(4).toDouble, miniBatchFraction = 1.0, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)
      println("Factorization Machine Training Completed")

      val ValueAndLabels = testData.map {
        y =>
          val x = fm1.predict(y.features)
          (x, y.label)
      }

      val PredictionAndLabel = testData.map{
        y =>
          val temp = fm1.predict(y.features)
          var flag: Int = 0
          if(temp > 0.5)
            flag = 1
          else
            flag = 0
          (temp, y.label)
      }

      val metrics = new MulticlassMetrics(PredictionAndLabel)
      val precision = metrics.precision
      precise(i) = precision
      //println("Precision = " + precision)
      val NE = -ValueAndLabels.map { case (p, v) => (1 + v) / 2 * math.log(p + 0.00000001) + (1 - v) / 2 * math.log(1 - p + 0.00000001) }.mean()
      //print("training Normalized Cross Entropy = " + NE)
      NE_sum(i) = NE

    }
    precise.map(x => println(x))
    println("average prediction value " + precise.sum / precise.length)
    NE_sum.map(x => println(x))
    println("normalized entropy value " + NE_sum.sum / NE_sum.length)

    /** Using LBFGS for FM training **/
    //val fm2 = FMWithLBFGS.train(training, task = 1, numIterations = 20, numCorrections = 5, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)

  }
}

