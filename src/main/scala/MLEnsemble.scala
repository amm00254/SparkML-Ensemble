import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, classification, linalg}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, GBTClassifier, LinearSVC, LogisticRegression, LogisticRegressionModel, MultilayerPerceptronClassifier, NaiveBayes, OneVsRest, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{HashingTF, IndexToString, MinMaxScaler, Normalizer, StringIndexer, Tokenizer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor, IsotonicRegression, RandomForestRegressor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import scala.collection.mutable.ArrayBuffer

object MLEnsemble {

  //-----------------------------------------------------------------------
  //LOADING OF FILE AND MODIFICATION OF NOMINAL VALUES BY NUMERICAL

  /**
   * DATASET LOAD FUNCTION AND NECESSARY TRANSFORMATIONS
   * @param spark Object to get Spark functionality
   * @param archivoEntrada Data set to load and transform for later use
   * @param nParticion Number of internal Spark partitions
   * @return Transformed dataset
   */
  def cargaDatosyModificacion(spark: SparkSession, archivoEntrada: String, classRegr: String, nParticion: Int): DataFrame = {

    var DF = spark.read.format("csv")
      .option("sep", ",")
      .option("header", "false")
      .option("inferSchema","true")
      .load(archivoEntrada)

    //Spark internal partitions
    DF = DF.repartition(nParticion)

    println("___________________________________________________________________")
    println("Partitions: " + DF.rdd.partitions.size)
    println("___________________________________________________________________")

    val columnas = DF.dtypes
    val columnaClase = columnas(columnas.length-1)._1.toString
    var claseNom = false

    //Column handling with dataType String
    (0 to columnas.length - 1).map { i =>

      val dataType = DF.schema(columnas(i)._1).dataType

      if (dataType == StringType) {

        val indexer = new StringIndexer()
          .setInputCol(columnas(i)._1.toString)
          .setOutputCol(columnas(i)._1.toString + "toN")

        val nombreOriginal = columnas(i)._1.toString

        if (i != columnas.length - 1) {
          val DFIndexed = indexer.fit(DF).transform(DF).drop(columnas(i)._1.toString).withColumnRenamed(columnas(i)._1.toString + "toN", nombreOriginal)
          DF = DFIndexed
        } else {
          claseNom = true
          val DFIndexed = indexer.fit(DF).transform(DF).drop(columnas(i)._1.toString).withColumnRenamed(columnas(i)._1.toString + "toN", "label")
          DF = DFIndexed
        }

      }

    }

    //In the event that the class column is not nominal we have to move it to leave it in last position and cast its dataType
    if(!claseNom) {
      DF = DF.withColumn("label", col(columnaClase).cast(DoubleType)).drop(columnaClase)
    }

    //In the case of having a data set where its class column has negative values, we must also treat it to leave them positive
    if(classRegr == "clasificacion" && DF.select("label").filter(col("label") < 0).count() > 0) {

      DF = DF.withColumn("label", col("label").cast(StringType))

      val converter = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("labelIndices")

      DF = converter.fit(DF).transform(DF).drop("label").withColumnRenamed("labelIndices", "label")

    }

    //Add the numerical values ​​to a dynamic array, and then pass them to a static array
    var colNames = ArrayBuffer[String]()

    (0 to columnas.length - 2).map { i =>
      colNames += columnas(i)._1
    }

    val colNamesStatic = colNames.toArray

    //Casting numeric values ​​to Double type
    (0 to columnas.length - 2).map { i =>
      DF = DF.withColumn(columnas(i)._1, col(columnas(i)._1).cast(DoubleType))
    }

    //Put all numeric attributes together in an array
    val assembler = new VectorAssembler()
      .setInputCols(colNamesStatic)
      .setOutputCol("features")

    val output = assembler.transform(DF)

    //Normalize the values ​​of the characteristics of the data set in a range
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("normFeatures")

    val scalerModel = scaler.fit(output)

    var scaledData = scalerModel.transform(output)
    scaledData = scaledData.drop("features").withColumnRenamed("normFeatures", "features")

    //Clear columns with isolated numeric attributes, leaving only the column containing the numeric attribute vectors
    (0 to columnas.length - 2).map { i =>
      scaledData = scaledData.drop(columnas(i)._1)
    }

    val reorderColumns = Array("features", "label")
    var DFResult = scaledData.select(reorderColumns.head, reorderColumns.tail: _*)

    DFResult = DFResult.dropDuplicates("features")

    DFResult

  }

  //-----------------------------------------------------------------------
  //CLASSIFICATION AND REGRESSION FUNCTIONS

  /**
   * SVM: CLASSIFICATION METHOD FOR DATA SETS WITH BINARY CLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @return Dataframe of predictions
   */
  def SVM(trainingData: Array[DataFrame], testData: DataFrame, inicio: Int, fin: Int): DataFrame = {

    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)

    // Fit the model
    val lsvcModel = lsvc.fit(trainingData(inicio))

    var modelo = lsvcModel.transform(testData).drop("rawPrediction")

    resultadosClasificador(s"Clasificador SVM: ", "label", "prediction", modelo, inicio)

    modelo = modelo.withColumnRenamed("prediction", "prediction" + inicio)

    for (i <- inicio+1 to fin){

      val model = lsvc.fit(trainingData(i))
      var modeloLoop = model.transform(testData)

      resultadosClasificador(s"Clasificador SVM: ", "label", "prediction", modeloLoop, i)

      modeloLoop = modeloLoop.drop("rawPrediction", "label").withColumnRenamed("prediction", "prediction" + i)

      modelo = modelo.join(modeloLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modelo, modelo.columns.takeRight(modelo.columns.length - 1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/SVMPredicciones", ensembleRdd.collect().toList)

    modelo

  }

  /**
   * MPC: CLASSIFICATION METHOD FOR DATA SETS WITH BINARY AND MULTI-CLASS CLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @param nClasses Variable that will set the different number of classes that the dataset contains
   * @param nFeatures Variable that will set the different number of features that the dataset contains
   * @return Dataframe of predictions
   */
  def MPC(trainingData: Array[DataFrame], testData: DataFrame, inicio: Int, fin: Int, nClasses: Int, nFeatures: Int): DataFrame = {

    val layers = Array[Int](nFeatures, 5, 4, nClasses)

    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    val MPCmodel = trainer.fit(trainingData(inicio))

    var modelo = MPCmodel.transform(testData).drop("rawPrediction", "probability")

    resultadosClasificador(s"Clasificador Multilayer Perceptron: ", "label", "prediction", modelo, inicio)

    modelo = modelo.withColumnRenamed("prediction", "prediction" + inicio)

    for (i <- inicio+1 to fin){

      val model = trainer.fit(trainingData(i))
      var modeloLoop = model.transform(testData)

      resultadosClasificador(s"Clasificador Multilayer Perceptron: ", "label", "prediction", modeloLoop, i)

      modeloLoop = modeloLoop.drop("rawPrediction", "probability", "label").withColumnRenamed("prediction", "prediction" + i)

      modelo = modelo.join(modeloLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modelo, modelo.columns.takeRight(modelo.columns.length - 1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/MPCPredicciones", ensembleRdd.collect().toList)

    modelo

  }

  /**
   * LOGISTIC REGRESSION: CLASSIFICATION METHOD FOR DATA SETS WITH BINARY CLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param path Path to load numerical parameters from Logistic Regression
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @param spark Object to get Spark functionality
   * @return Dataframe of predictions
   */
  def LogisticRegression(trainingData: Array[DataFrame], testData: DataFrame, path: String, inicio: Int, fin: Int, spark: SparkSession): DataFrame = {

    val arrayDatos = spark.sparkContext.textFile(path).collect()

    val lr = new LogisticRegression()
      .setMaxIter(arrayDatos(0).toInt)
      .setRegParam(arrayDatos(1).toDouble)
      .setElasticNetParam(arrayDatos(2).toDouble)

    // Fit the model
    val lrModel = lr.fit(trainingData(inicio))

    // Select example rows to display.
    var modelo = lrModel.transform(testData).drop("rawPrediction", "probability")

    resultadosClasificador(s"Clasificador Logistic Regression: ", "label", "prediction", modelo, inicio)

    modelo = modelo.withColumnRenamed("prediction", "prediction" + inicio)

    for (i <- inicio+1 to fin){

      val model = lr.fit(trainingData(i))
      var modeloLoop = model.transform(testData)

      resultadosClasificador(s"Clasificador Logistic Regression: ", "label", "prediction", modeloLoop, i)

      modeloLoop = modeloLoop.drop("rawPrediction", "probability", "label").withColumnRenamed("prediction", "prediction" + i)

      modelo = modelo.join(modeloLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modelo, modelo.columns.takeRight(modelo.columns.length - 1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/LRPredicciones", ensembleRdd.collect().toList)

    modelo

  }

  /**
   * DECISION TREE: CLASSIFICATION METHOD FOR DATA SETS WITH BINARY AND MULTI-CLASS CLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param DFOriginal Original data set needed to build a Pipeline
   * @param path Path to load numerical parameters from Decision Tree
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @param spark Object to get Spark functionality
   * @return Dataframe of predictions
   */
  def DecisionTree(trainingData: Array[DataFrame], testData: DataFrame, DFOriginal: DataFrame, path: String, inicio: Int, fin: Int, spark: SparkSession): DataFrame = {

    val arrayDatos = spark.sparkContext.textFile(path).collect()

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(DFOriginal)

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(arrayDatos(0).toInt)
      .fit(DFOriginal)

    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData(inicio))

    // Make predictions.
    var modeloDTree = model.transform(testData).withColumn("predictedLabel", col("predictedLabel").cast(DoubleType))

    resultadosClasificador(s"Clasificador Decision Tree: ", "indexedLabel", "prediction", modeloDTree, inicio)

    modeloDTree = modeloDTree.drop("indexedLabel", "indexedFeatures", "rawPrediction", "probability", "prediction").withColumnRenamed("predictedLabel", "prediction" + inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      var modeloDTreeLoop = modelLoop.transform(testData).withColumn("predictedLabel", col("predictedLabel").cast(DoubleType))

      resultadosClasificador(s"Clasificador Decision Tree: ", "indexedLabel", "prediction", modeloDTreeLoop, i)

      modeloDTreeLoop = modeloDTreeLoop.drop("label", "indexedLabel", "indexedFeatures", "rawPrediction", "probability", "prediction").withColumnRenamed("predictedLabel", "prediction" + i)

      modeloDTree = modeloDTree.join(modeloDTreeLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modeloDTree, modeloDTree.columns.takeRight(modeloDTree.columns.length - 1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/DecisionTreePredicciones", ensembleRdd.collect().toList)

    modeloDTree

  }

  /**
   * DECISION TREE REGRESSION: REGRESSION METHOD FOR DATA SETS WITH REAL CLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param DFOriginal Original data set needed to build a Pipeline
   * @param path Path to load numerical parameters from Decision Tree
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @param spark Object to get Spark functionality
   * @return Dataframe of predictions
   */
  def DecisionTreeRegr(trainingData: Array[DataFrame], testData: DataFrame, DFOriginal: DataFrame, path: String, inicio: Int, fin: Int, spark: SparkSession): DataFrame = {

    val arrayDatos = spark.sparkContext.textFile(path).collect()

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(arrayDatos(0).toInt)
      .fit(DFOriginal)

    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData(inicio))

    // Make predictions.
    var modeloDTreeR = model.transform(testData)

    resultadosRegresor(s"Regresor Decision Tree Regression: ", "label", "prediction", modeloDTreeR, inicio)

    modeloDTreeR = modeloDTreeR.drop( "indexedFeatures").withColumnRenamed("prediction", "prediction" + inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      var modeloDTreeRLoop = modelLoop.transform(testData)

      resultadosRegresor(s"Regresor Decision Tree Regression: ", "label", "prediction", modeloDTreeRLoop, i)

      modeloDTreeRLoop = modeloDTreeRLoop.drop( "indexedFeatures", "label").withColumnRenamed("prediction", "prediction" + i)

      modeloDTreeR = modeloDTreeR.join(modeloDTreeRLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modeloDTreeR, modeloDTreeR.columns.takeRight(modeloDTreeR.columns.length - 1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/DecisionTreeRegressionPredicciones", ensembleRdd.collect().toList)

    modeloDTreeR

  }

  /**
   * RANDOM FOREST: CLASSIFICATION METHOD FOR DATA SET WITH BINARY CLASSES AND MULTICLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param DFOriginal Original data set needed to build a Pipeline
   * @param path Path to load numerical parameters from Random Forest
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @param spark Object to get Spark functionality
   * @return Dataframe of predictions
   */
  def RandomForest(trainingData: Array[DataFrame], testData: DataFrame, DFOriginal: DataFrame, path: String, inicio: Int, fin: Int, spark: SparkSession): DataFrame = {

    val arrayDatos = spark.sparkContext.textFile(path).collect()

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(DFOriginal)

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(arrayDatos(0).toInt)
      .fit(DFOriginal)

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(arrayDatos(1).toInt)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData(inicio))

    // Make predictions.
    var modeloRF = model.transform(testData).withColumn("predictedLabel", col("predictedLabel").cast(DoubleType))

    resultadosClasificador(s"Clasificador Random Forest: ", "indexedLabel", "prediction", modeloRF, inicio)

    modeloRF = modeloRF.drop("indexedLabel", "indexedFeatures", "rawPrediction", "probability", "prediction").withColumnRenamed("predictedLabel", "prediction" + inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      var modeloRFLoop = modelLoop.transform(testData).withColumn("predictedLabel", col("predictedLabel").cast(DoubleType))

      resultadosClasificador(s"Clasificador Random Forest: ", "indexedLabel", "prediction", modeloRFLoop, i)

      modeloRFLoop = modeloRFLoop.drop("label", "indexedLabel", "indexedFeatures", "rawPrediction", "probability", "prediction").withColumnRenamed("predictedLabel", "prediction" + i)

      modeloRF = modeloRF.join(modeloRFLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modeloRF, modeloRF.columns.takeRight(modeloRF.columns.length - 1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/RandomForestPredicciones", ensembleRdd.collect().toList)

    modeloRF

  }

  /**
   * RANDOM FOREST REGRESSION: REGRESSION METHOD FOR DATA SETS WITH REAL CLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param DFOriginal Original data set needed to build a Pipeline
   * @param path Path to load numerical parameters from Random Forest
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @param spark Object to get Spark functionality
   * @return Dataframe of predictions
   */
  def RandomForestRegr(trainingData: Array[DataFrame], testData: DataFrame, DFOriginal: DataFrame, path: String, inicio: Int, fin: Int, spark: SparkSession): DataFrame = {

    val arrayDatos = spark.sparkContext.textFile(path).collect()

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(arrayDatos(0).toInt)
      .fit(DFOriginal)

    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, rf))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData(inicio))

    // Make predictions.
    var modeloRFR = model.transform(testData)

    resultadosRegresor(s"Regresor Random Forest Regression: ", "label", "prediction", modeloRFR, inicio)

    modeloRFR = modeloRFR.drop( "indexedFeatures").withColumnRenamed("prediction", "prediction" + inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      var modeloRFRLoop = modelLoop.transform(testData)

      resultadosRegresor(s"Regresor Random Forest Regression: ", "label", "prediction", modeloRFRLoop, i)

      modeloRFRLoop = modeloRFRLoop.drop( "indexedFeatures", "label").withColumnRenamed("prediction", "prediction" + i)

      modeloRFR = modeloRFR.join(modeloRFRLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modeloRFR, modeloRFR.columns.takeRight(modeloRFR.columns.length - 1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/RandomForestRegressionPredicciones", ensembleRdd.collect().toList)

    modeloRFR

  }

  /**
   * GRADIENT BOOSTED TREE REGRESSION: REGRESSION METHOD FOR DATA SETS WITH REAL CLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param DFOriginal Original data set needed to build a Pipeline
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @return Dataframe of predictions
   */
  def GradientBTRegr(trainingData: Array[DataFrame], testData: DataFrame, DFOriginal: DataFrame, inicio: Int, fin: Int): DataFrame = {

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(DFOriginal)

    // Train a RandomForest model.
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData(inicio))

    // Make predictions.
    var modeloGBT = model.transform(testData)

    resultadosRegresor(s"Regresor Gradient Boosted Tree: ", "label", "prediction", modeloGBT, inicio)

    modeloGBT = modeloGBT.drop("indexedFeatures").withColumnRenamed("prediction", "prediction" + inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      var modeloGBTLoop = modelLoop.transform(testData)

      resultadosRegresor(s"Regresor Gradient Boosted Tree: ", "label", "prediction", modeloGBTLoop, i)

      modeloGBTLoop = modeloGBTLoop.drop("indexedFeatures", "label").withColumnRenamed("prediction", "prediction" + i)

      modeloGBT = modeloGBT.join(modeloGBTLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modeloGBT, modeloGBT.columns.takeRight(modeloGBT.columns.length - 1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/GBTRegressionPredicciones", ensembleRdd.collect().toList)

    modeloGBT

  }

  /**
   * MÉTODO ISOTONIC REGRESSION: REGRESSION METHOD FOR DATA SET WITH REAL CLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @return Dataframe of predictions
   */
  def IsotonicRegr(trainingData: Array[DataFrame], testData: DataFrame, inicio: Int, fin: Int): DataFrame = {

    val ir = new IsotonicRegression()

    val model = ir.fit(trainingData(inicio))

    // Make predictions.
    var modeloIso = model.transform(testData)

    resultadosRegresor(s"Regresor Isotonic: ", "label", "prediction", modeloIso, inicio)

    modeloIso = modeloIso.withColumnRenamed("prediction", "prediction" + inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = ir.fit(trainingData(i))
      var modeloIsoLoop = modelLoop.transform(testData)

      resultadosRegresor(s"Regresor Isotonic: ", "label", "prediction", modeloIsoLoop, i)

      modeloIsoLoop = modeloIsoLoop.drop("label").withColumnRenamed("prediction", "prediction" + i)

      modeloIso = modeloIso.join(modeloIsoLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modeloIso, modeloIso.columns.takeRight(modeloIso.columns.length - 1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/IsotonicRegressionPredicciones", ensembleRdd.collect().toList)

    modeloIso

  }

  /**
   * MÉTODO NAIVE BAYES: CLASSIFICATION METHOD FOR DATA SETS WITH BINARY AND MULTI-CLASS CLASSES
   * @param trainingData Training dataset array
   * @param testData Test dataset
   * @param inicio Variable to take a dataset from the dataset array
   * @param fin Variable to take a dataset from the dataset array
   * @return Dataframe of predictions
   */
  def NaiveBayes(trainingData: Array[DataFrame], testData: DataFrame, inicio: Int, fin: Int): DataFrame = {

    // Train a NaiveBayes model.
    val model = new NaiveBayes().fit(trainingData(inicio))

    // Select example rows to display.
    var modeloNB = model.transform(testData).drop("rawPrediction", "probability")

    resultadosClasificador(s"Clasificador Naive Bayes: ", "label", "prediction", modeloNB, inicio)

    modeloNB = modeloNB.withColumnRenamed("prediction", "prediction" + inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = new NaiveBayes().fit(trainingData(i))
      var modeloNBLoop = modelLoop.transform(testData)

      resultadosClasificador(s"Clasificador Naive Bayes: ", "label", "prediction", modeloNBLoop, i)

      modeloNBLoop = modeloNBLoop.drop("rawPrediction", "probability", "label").withColumnRenamed("prediction", "prediction" + i)

      modeloNB = modeloNB.join(modeloNBLoop, "features")

    }

    //val ensembleRdd = DFtoRDD(modeloNB, modeloNB.columns.takeRight(modeloNB.columns.length-1))
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/NaiveBayesPredicciones", ensembleRdd.collect().toList)

    modeloNB

  }

  //-----------------------------------------------------------------------
  //NECESSARY FUNCTIONS
  /**
   * FUNCTION TO CHANGE A VALUE TO INT
   * @param v Value of a certain row
   * @return Value changed to int
   */
  def sqlRowToInt(v: org.apache.spark.sql.Row): Int = {
    v.get(0).toString.toDouble.toInt
  }

  /**
   * FUNCTION TO OBTAIN THE RESULTS BY USING CLASSIFICATION METHODS AND THE FINAL ASSEMBLY
   * @param informacion Name that we want to display of the classifier
   * @param columnaLabel Label column to be used for the dataframe
   * @param columnaPrediction Prediction column to be used for the dataframe
   * @param prediccion Classifier dataframe
   * @param nClasificador Integer to know what classifier number is being treated, to show information
   */
  def resultadosClasificador(informacion: String, columnaLabel: String, columnaPrediction: String, prediccion: DataFrame, nClasificador: Int): Unit = {

    val evaluatorAccuracy = new MulticlassClassificationEvaluator()
      .setLabelCol(columnaLabel)
      .setPredictionCol(columnaPrediction)
      .setMetricName("accuracy")

    val evaluatorWeightedPrecision = new MulticlassClassificationEvaluator()
      .setLabelCol(columnaLabel)
      .setPredictionCol(columnaPrediction)
      .setMetricName("weightedPrecision")

    val accuracy = evaluatorAccuracy.evaluate(prediccion)
    val weightedPrecision = evaluatorWeightedPrecision.evaluate(prediccion)

    println(s"-------------------------")
    println(informacion + nClasificador)
    println(s"Accuracy = ${(accuracy)}")
    println(s"Weighted Precision = ${(weightedPrecision)}")
    println(s"-------------------------")

  }

  /**
   * FUNCTION FOR OBTAINING RESULTS BY USING REGRESSION METHODS AND THE FINAL ASSEMBLY
   * @param informacion Name that we want to display of the classifier
   * @param columnaLabel Label column to be used for the dataframe
   * @param columnaPrediction Prediction column to be used for the dataframe
   * @param prediccion Regression dataframe
   * @param nClasificador Integer to know what classifier number is being treated, to show information
   */
  def resultadosRegresor(informacion: String, columnaLabel: String, columnaPrediction: String, prediccion: DataFrame, nClasificador: Int): Unit = {

    val evaluatorRMSE = new RegressionEvaluator()
      .setLabelCol(columnaLabel)
      .setPredictionCol(columnaPrediction)
      .setMetricName("rmse")

    //Mean Squared Error
    val evaluatorMSE = new RegressionEvaluator()
      .setLabelCol(columnaLabel)
      .setPredictionCol(columnaPrediction)
      .setMetricName("mse");

    //Regression through the origin
    val evaluatorR2 = new RegressionEvaluator()
      .setLabelCol(columnaLabel)
      .setPredictionCol(columnaPrediction)
      .setMetricName("r2");

    //Mean absolute error
    val evaluatorMAE = new RegressionEvaluator()
      .setLabelCol(columnaLabel)
      .setPredictionCol(columnaPrediction)
      .setMetricName("mae");

    val rmse = evaluatorRMSE.evaluate(prediccion)
    val mse = evaluatorMSE.evaluate(prediccion)
    val r2 = evaluatorR2.evaluate(prediccion)
    val mae = evaluatorMAE.evaluate(prediccion)

    println(s"-------------------------")
    println(informacion + nClasificador)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    println(s"Mean squared error (MSE) on test data =   $mse")
    println(s"Regression through the origin(R2) on test data =  + $r2")
    println(s"Mean absolute error (MAE) on test data =  + $mae")
    println(s"-------------------------")

  }

  /**
   * FUNCTION TO SHOW TIMES USED IN A CERTAIN ACTION
   * @param time Time that should have been previously generated
   * @param dato Information to recognize the taking of times in a specific place
   */
  def tiempo(time: Long, dato: String) {

    val duration = ((System.currentTimeMillis()/1000) - time)
    val seconds = duration % 60
    val minutes = (duration/60) % 60
    val hours = (duration/60/60)
    println(dato + " = " + "%02d:%02d:%02d".format(hours, minutes, seconds))

  }

  /**
   * FUNCTION TO CONVERT A DATAFRAME TO RDD [DENSEVECTOR]
   * @param DF Dataframe to convert
   * @param columns Columns of the dataframe to be converted
   * @return RDD[DenseVector]
   */
  def DFtoRDD(DF: DataFrame, columns: Array[String]): RDD[DenseVector] = {

    val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("AllPrediction")
    var ensemble = assembler.transform(DF).select("AllPrediction")
    val ensembleRdd = ensemble.rdd.map(row => row.getAs[org.apache.spark.ml.linalg.SparseVector]("AllPrediction").toDense)

    ensembleRdd

  }

  /**
   * FUNCTION TO SAVE A DENSEVECTOR LIST ON A PRESET ROUTE PER PARAMETER
   * @param path Path where the model will be saved
   * @param arrayDatos Model that will be a DenseVector list to be able to save it correctly
   */
  def savePredictionEnsemble(path: String, arrayDatos: List[DenseVector]): Unit = {

    val file = path
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))

    arrayDatos.map { i => writer.write(i + "\n") }

    writer.close()

  }

  /**
   * FUNCTION THAT ALLOWS US TO UNITE ALL THE PREDICTIONS GENERATED BY ALL THE CLASSIFICATION / REGRESSION METHODS USED
   * @param arrayDF Prediction dataframe array
   * @return Row Vector RDD of Prediction Array (A vector contains the elements of one of all predictions)
   */
  def unionDF(arrayDF: ArrayBuffer[DataFrame]): RDD[DenseVector] = {

    var DFPredicciones = arrayDF(0)

    (1 to arrayDF.length - 1).map { i =>

      arrayDF(i) = arrayDF(i).drop("label")
      DFPredicciones = DFPredicciones.join(arrayDF(i), "features")

    }

    DFPredicciones = DFPredicciones.drop("features")

    val ensembleRdd = DFtoRDD(DFPredicciones, DFPredicciones.columns)
    //savePredictionEnsemble("C:/Users/Usuario/Desktop/modelData/Pruebas/joinPredictions", ensembleRdd.collect().toList)

    ensembleRdd

  }

  /**
   * FUNCTION TO VOTE CLASSIFICATION OF THE PREDICTIONS OF THE CLASSIFICATION METHODS USED
   * @param predictionRDD RDD vector predictions
   * @param nClasses Number of classes in the dataset
   * @return RDD of integers, where each element is the winner by vote
   */
  def clasificacionRow(predictionRDD: RDD[DenseVector], nClasses: Int): RDD[(Double, Double)] = {

    val votos = predictionRDD.map { v => var arrayClasses = Array.ofDim[Int](nClasses); val elementosFila = v.toArray; elementosFila.takeRight(elementosFila.length - 1).map { e => arrayClasses(e.toInt) += 1};
      val indiceMax = arrayClasses.indexOf(arrayClasses.max); (elementosFila(0), indiceMax.toDouble) }

    votos

  }

  /**
   * FUNCTION TO OBTAIN THE AVERAGE OF THE PREDICTIONS OF THE EMPLOYED REGRESSION METHODS
   * @param predictionRDD RDD vector predictions
   * @return RDD of doubles, where each element is the mean of all the elements that made up its row of predictions
   */
  def regressionRow(predictionRDD: RDD[linalg.DenseVector]): RDD[(Double, Double)] = {

    val media = predictionRDD.map { v => var media = 0.0; val elementosFila = v.toArray; elementosFila.takeRight(elementosFila.length - 1).map { e =>  media += e }; media = media/(elementosFila.length - 1); (elementosFila(0), media) }

    media

  }

  /**
   * FUNCTION TO OBTAIN A SUB-SET OF TRAINING DATA
   * @param trainingData Original training data set
   * @param n Number of subsets of data that will be generated for us
   * @param porcentaje Percentage of original training dataset that will contain subsets of data
   * @return Array of dataframes, which will be the subsets of training data
   */
  def bootstrap(trainingData: DataFrame, n: Int, porcentaje: Double): Array[DataFrame] = {

    var subDF = ArrayBuffer[DataFrame]()

    val rows = (porcentaje*trainingData.count()).toInt

    (0 to n - 1).map { i =>
      subDF += trainingData.orderBy(rand()).limit(rows).orderBy(asc("label"))
    }

    val subDFReturn = subDF.toArray

    subDFReturn

  }

  //-----------------------------------------------------------------------
  //ENSEMBLE
  /**
   * MAIN FUNCTION PERFORMED BY THE ENSEMBLE BAGGING
   * @param n Number of classification / regression methods used
   * @param m Number of subsets of data that will be generated for us
   * @param porcentaje Percentage of the original training dataset that will contain the data subsets
   * @param modelosClass Array of classification / regression methods
   * @param paramArchivos Array with the necessary file parameters
   * @param trainingData Original training data set
   * @param testData Test dataset
   * @param nClasses Number of classes in the dataset
   * @param nFeatures Number of features in the dataset
   * @param DF Original dataset
   * @param nBagging Bagging run number being performed
   * @param classRegr String that checks if a classification or a regression is being used
   * @param spark Object to get Spark functionality
   */
  def bagging(n: Int, m: Int, porcentaje: Double, modelosClass: Array[String], paramArchivos: Array[String], trainingData: DataFrame,
              testData: DataFrame, nClasses: Int, nFeatures: Int, DF: DataFrame, nBagging: Int, classRegr: String, spark: SparkSession): Unit = {

    val timeDivision = System.currentTimeMillis()/1000

    val subDF = bootstrap(trainingData, m, porcentaje)

    tiempo(timeDivision, "Tiempo división subDataframes")

    val roundUp = m/n
    var iter = 0 to roundUp-1
    var inicio = 0
    var fin = roundUp-1

    var arrayDF = ArrayBuffer[DataFrame]()

    val timeClasificadores = System.currentTimeMillis()/1000

    (0 to n - 1).map { x =>

      if (x == n - 1) {

        iter = (roundUp * x) to (subDF.size - 1)
        inicio = roundUp * x
        fin = subDF.size - 1

        if (modelosClass(x) == "NaiveBayes") {
          arrayDF += NaiveBayes(subDF, testData, inicio, fin)
        }

        if (modelosClass(x) == "MPC") {
          arrayDF += MPC(subDF, testData, inicio, fin, nClasses, nFeatures)
        }

        if (modelosClass(x) == "SVM") {
          arrayDF += SVM(subDF, testData, inicio, fin)
        }

        if (modelosClass(x) == "LogisticRegression") {
          arrayDF += LogisticRegression(subDF, testData, paramArchivos(0), inicio, fin, spark)
        }

        if (modelosClass(x) == "GBT") {
          arrayDF += GradientBTRegr(subDF, testData, DF, inicio, fin)
        }

        if (modelosClass(x) == "Isotonic") {
          arrayDF += IsotonicRegr(subDF, testData, inicio, fin)
        }

        if (modelosClass(x) == "RandomForest") {
          if (classRegr == "clasificacion") {
            arrayDF += RandomForest(subDF, testData, DF, paramArchivos(2), inicio, fin, spark)
          } else {
            arrayDF += RandomForestRegr(subDF, testData, DF, paramArchivos(2), inicio, fin, spark)
          }

        }

        if (modelosClass(x) == "DecisionTree") {
          if (classRegr == "clasificacion") {
            arrayDF += DecisionTree(subDF, testData, DF, paramArchivos(1), inicio, fin, spark)
          } else {
            arrayDF += DecisionTreeRegr(subDF, testData, DF, paramArchivos(1), inicio, fin, spark)
          }
        }

      } else {

        if (modelosClass(x) == "NaiveBayes") {
          arrayDF += NaiveBayes(subDF, testData, inicio, fin)
        }

        if (modelosClass(x) == "MPC") {
          arrayDF += MPC(subDF, testData, inicio, fin, nClasses, nFeatures)
        }

        if (modelosClass(x) == "SVM") {
          arrayDF += SVM(subDF, testData, inicio, fin)
        }

        if (modelosClass(x) == "LogisticRegression") {
          arrayDF += LogisticRegression(subDF, testData, paramArchivos(0), inicio, fin, spark)
        }

        if (modelosClass(x) == "GBT") {
          arrayDF += GradientBTRegr(subDF, testData, DF, inicio, fin)
        }

        if (modelosClass(x) == "Isotonic") {
          arrayDF += IsotonicRegr(subDF, testData, inicio, fin)
        }

        if (modelosClass(x) == "RandomForest") {
          if (classRegr == "clasificacion") {
            arrayDF += RandomForest(subDF, testData, DF, paramArchivos(2), inicio, fin, spark)
          } else {
            arrayDF += RandomForestRegr(subDF, testData, DF, paramArchivos(2), inicio, fin, spark)
          }
        }

        if (modelosClass(x) == "DecisionTree") {
          if (classRegr == "clasificacion") {
            arrayDF += DecisionTree(subDF, testData, DF, paramArchivos(1), inicio, fin, spark)
          } else {
            arrayDF += DecisionTreeRegr(subDF, testData, DF, paramArchivos(1), inicio, fin, spark)
          }
        }

        iter = (roundUp * (x + 1)) to (roundUp * (x + 2)) - 1
        inicio = (roundUp * (x + 1))
        fin = (roundUp * (x + 2)) - 1

      }

    }

    tiempo(timeClasificadores, "Tiempo realización clasificadores")

    val timeTransfPredic = System.currentTimeMillis()/1000
    val RDDPredicciones = unionDF(arrayDF)
    tiempo(timeTransfPredic, "Tiempo unión y transformación de las predicciones en vectores por filas")

    import spark.implicits._

    if(classRegr == "clasificacion") {

      val timeClasificacion = System.currentTimeMillis()/1000
      val clasificacion = clasificacionRow(RDDPredicciones, nClasses)
      tiempo(timeClasificacion, "Tiempo realización clasificación por voto y guardado del resultado en fichero")

      val DFBaggingFinal = clasificacion.toDF("label", "baggingPrediction")

      //val ensembleRdd = DFtoRDD(DFBaggingFinal, DFBaggingFinal.columns)
      //savePredictionEnsemble(paramArchivos(3) + "baggingModel-" + nBagging, ensembleRdd.collect().toList)

      resultadosClasificador(s"RESULTADOS FINALES DEL ENSEMBLE CON CLASIFICACIÓN: ", "label", "baggingPrediction", DFBaggingFinal, nBagging)

    } else if(classRegr == "regresion"){

      val timeRegresion = System.currentTimeMillis()/1000
      val regresion = regressionRow(RDDPredicciones)
      tiempo(timeRegresion, "Tiempo realización regresión por media y guardado del resultado en fichero")

      val DFBaggingFinal = regresion.toDF("label", "baggingPrediction")

      //val ensembleRdd = DFtoRDD(DFBaggingFinal, DFBaggingFinal.columns)
      //savePredictionEnsemble(paramArchivos(3) + "baggingModel-" + nBagging, ensembleRdd.collect().toList)

      resultadosRegresor(s"RESULTADOS FINALES DEL ENSEMBLE CON REGRESIÓN: ","label","baggingPrediction",DFBaggingFinal,nBagging)

    }

  }

  //-----------------------------------------------------------------------
  //PRINCIPAL

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val timePrograma = System.currentTimeMillis()/1000

    val spark = SparkSession
      .builder()
      .appName("Spark SQL data")
      //.config("spark.master", "local") //Uncomment this line if you want to run locally
      .getOrCreate()

    //These routes should be modified according to the user who runs the program and their location of the project and files
    val paramNumericos = spark.sparkContext.textFile("/home/simidat/amm00254/SparkML-Ensemble/out/artifacts/data/param/parametrosNumericos.txt").collect()

    val paramModelosClass = spark.sparkContext.textFile("/home/simidat/amm00254/SparkML-Ensemble/out/artifacts/data/param/parametrosModelosClass.txt").collect()

    val paramArchivos = spark.sparkContext.textFile("/home/simidat/amm00254/SparkML-Ensemble/out/artifacts/data/param/parametrosArchivosES.txt").collect()

    val paramCR= spark.sparkContext.textFile("/home/simidat/amm00254/SparkML-Ensemble/out/artifacts/data/param/parametrosCR.txt").collect()

    //In the case of treating a data set with train and test together
    if(paramArchivos(4) == "false") {

      println("___________________________________________________________________")
      println("ENSEMBLE BAGGING CON TRAIN Y TEST JUNTOS")
      println("___________________________________________________________________")

      val timeCargaDatos = System.currentTimeMillis()/1000

      val DF = cargaDatosyModificacion(spark, paramArchivos(5), paramCR(0), paramNumericos(4).toInt)

      tiempo(timeCargaDatos, "Tiempo carga y transformación de datos")

      val Array(trainingData, testData) = DF.randomSplit(Array(paramNumericos(0).toDouble, paramNumericos(1).toDouble))

      val columnFeatures = DF.select("features").head
      val nFeatures = columnFeatures(0).asInstanceOf[DenseVector].size

      val nClasses = sqlRowToInt(DF.describe("label").filter("summary = 'max'").select("label").head) + 1

      val timeEnsembleBagging = System.currentTimeMillis()/1000

      bagging(paramModelosClass.length, paramNumericos(2).toInt, paramNumericos(3).toDouble, paramModelosClass, paramArchivos, trainingData, testData, nClasses, nFeatures, DF, 999, paramCR(0), spark)

      tiempo(timeEnsembleBagging, "Tiempo realización Ensemble Bagging")

      //In the case of dealing with already partitioned data sets
    } else {

      var inicio = 4
      val fin = paramArchivos.length - 1
      val result = fin - inicio
      val cont = 1

      (1 to result/2).map { i =>

        println("___________________________________________________________________")
        println("ENSEMBLE BAGGING " + i)
        println("___________________________________________________________________")

        val timeCargaDatos = System.currentTimeMillis()/1000

        val DFTrain = cargaDatosyModificacion(spark, paramArchivos(inicio + cont), paramCR(0), paramNumericos(4).toInt)
        inicio += 1
        val DFTest = cargaDatosyModificacion(spark, paramArchivos(inicio + cont), paramCR(0), paramNumericos(4).toInt)
        inicio += 1

        tiempo(timeCargaDatos, "Tiempo carga y transformación de datos")

        val columnFeatures = DFTrain.select("features").head
        val nFeatures = columnFeatures(0).asInstanceOf[DenseVector].size

        val nClasses = sqlRowToInt(DFTest.describe("label").filter("summary = 'max'").select("label").head) + 1
        val timeEnsembleBagging = System.currentTimeMillis()/1000

        bagging(paramModelosClass.length, paramNumericos(2).toInt, paramNumericos(3).toDouble, paramModelosClass, paramArchivos, DFTrain, DFTest, nClasses, nFeatures, DFTrain, i, paramCR(0), spark)

        tiempo(timeEnsembleBagging, "Tiempo realización Ensemble Bagging")

      }

    }

    tiempo(timePrograma, "Tiempo programa completo")

  }

}