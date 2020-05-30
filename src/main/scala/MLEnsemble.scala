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
    val DFResult = scaledData.select(reorderColumns.head, reorderColumns.tail: _*)

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

    val modelo = lsvcModel.transform(testData)

    var predictions = modelo.select("prediction").withColumnRenamed("prediction", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosClasificador(s"Clasificador SVM: ", "label", "prediction", modelo, inicio)

    for (i <- inicio+1 to fin){

      val model = lsvc.fit(trainingData(i))
      val modeloLoop = model.transform(testData)
      var colPrediction = modeloLoop.select("prediction").withColumnRenamed("prediction", "prediction" + i)

      resultadosClasificador(s"Clasificador SVM: ", "label", "prediction", modeloLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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

    val modelo = MPCmodel.transform(testData)

    var predictions = modelo.select("prediction").withColumnRenamed("prediction", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosClasificador(s"Clasificador Multilayer Perceptron: ", "label", "prediction", modelo, inicio)

    for (i <- inicio+1 to fin){

      val model = trainer.fit(trainingData(i))
      val modeloLoop = model.transform(testData)
      var colPrediction = modeloLoop.select("prediction").withColumnRenamed("prediction", "prediction" + i)

      resultadosClasificador(s"Clasificador Multilayer Perceptron: ", "label", "prediction", modeloLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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
    val modelo = lrModel.transform(testData)

    var predictions = modelo.select("prediction").withColumnRenamed("prediction", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosClasificador(s"Clasificador Logistic Regression: ", "label", "prediction", modelo, inicio)

    for (i <- inicio+1 to fin){

      val model = lr.fit(trainingData(i))
      val modeloLoop = model.transform(testData)
      var colPrediction = modeloLoop.select("prediction").withColumnRenamed("prediction", "prediction" + i)

      resultadosClasificador(s"Clasificador Logistic Regression: ", "label", "prediction", modeloLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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
    val modeloDTree = model.transform(testData).withColumn("predictedLabel", col("predictedLabel").cast(DoubleType))

    var predictions = modeloDTree.select("predictedLabel").withColumnRenamed("predictedLabel", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosClasificador(s"Clasificador Decision Tree: ", "indexedLabel", "prediction", modeloDTree, inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      val modeloDTreeLoop = modelLoop.transform(testData).withColumn("predictedLabel", col("predictedLabel").cast(DoubleType))
      var colPrediction = modeloDTreeLoop.select("predictedLabel").withColumnRenamed("predictedLabel", "prediction" + i)

      resultadosClasificador(s"Clasificador Decision Tree: ", "indexedLabel", "prediction", modeloDTreeLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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
    val modeloDTreeR = model.transform(testData)

    var predictions = modeloDTreeR.select("prediction").withColumnRenamed("prediction", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosRegresor(s"Regresor Decision Tree Regression: ", "label", "prediction", modeloDTreeR, inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      val modeloDTreeRLoop = modelLoop.transform(testData)
      var colPrediction = modeloDTreeRLoop.select("prediction").withColumnRenamed("prediction", "prediction" + i)

      resultadosRegresor(s"Regresor Decision Tree Regression: ", "label", "prediction", modeloDTreeRLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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
    val modeloRF = model.transform(testData).withColumn("predictedLabel", col("predictedLabel").cast(DoubleType))

    var predictions = modeloRF.select("predictedLabel").withColumnRenamed("predictedLabel", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosClasificador(s"Clasificador Random Forest: ", "indexedLabel", "prediction", modeloRF, inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      val modeloRFLoop = modelLoop.transform(testData).withColumn("predictedLabel", col("predictedLabel").cast(DoubleType))
      var colPrediction = modeloRFLoop.select("predictedLabel").withColumnRenamed("predictedLabel", "prediction" + i)

      resultadosClasificador(s"Clasificador Random Forest: ", "indexedLabel", "prediction", modeloRFLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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
    val modeloRFR = model.transform(testData)

    var predictions = modeloRFR.select("prediction").withColumnRenamed("prediction", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosRegresor(s"Regresor Random Forest Regression: ", "label", "prediction", modeloRFR, inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      val modeloRFRLoop = modelLoop.transform(testData)
      var colPrediction = modeloRFRLoop.select("prediction").withColumnRenamed("prediction", "prediction" + i)

      resultadosRegresor(s"Regresor Random Forest Regression: ", "label", "prediction", modeloRFRLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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
    val modeloGBT = model.transform(testData)

    var predictions = modeloGBT.select("prediction").withColumnRenamed("prediction", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosRegresor(s"Regresor Gradient Boosted Tree: ", "label", "prediction", modeloGBT, inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = pipeline.fit(trainingData(i))
      val modeloGBTLoop = modelLoop.transform(testData)
      var colPrediction = modeloGBTLoop.select("prediction").withColumnRenamed("prediction", "prediction" + i)

      resultadosRegresor(s"Regresor Gradient Boosted Tree: ", "label", "prediction", modeloGBTLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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
    val modeloIso = model.transform(testData)

    var predictions = modeloIso.select("prediction").withColumnRenamed("prediction", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosRegresor(s"Regresor Isotonic: ", "label", "prediction", modeloIso, inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = ir.fit(trainingData(i))
      val modeloIsoLoop = modelLoop.transform(testData)
      var colPrediction = modeloIsoLoop.select("prediction").withColumnRenamed("prediction", "prediction" + i)

      resultadosRegresor(s"Regresor Isotonic: ", "label", "prediction", modeloIsoLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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
    val modeloNB = model.transform(testData)

    var predictions = modeloNB.select("prediction").withColumnRenamed("prediction", "prediction" + inicio)
    predictions = predictions.withColumn("ID", monotonically_increasing_id())

    resultadosClasificador(s"Clasificador Naive Bayes: ", "label", "prediction", modeloNB, inicio)

    for (i <- inicio+1 to fin) {

      val modelLoop = new NaiveBayes().fit(trainingData(i))
      val modeloNBLoop = modelLoop.transform(testData)
      var colPrediction = modeloNBLoop.select("prediction").withColumnRenamed("prediction", "prediction" + i)

      resultadosClasificador(s"Clasificador Naive Bayes: ", "label", "prediction", modeloNBLoop, i)

      colPrediction = colPrediction.withColumn("ID", monotonically_increasing_id())
      predictions = predictions.join(colPrediction, predictions("ID") === colPrediction("ID"), "inner").drop("ID")
      predictions = predictions.withColumn("ID", monotonically_increasing_id())

    }

    predictions = predictions.drop("ID")

    predictions

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
   * FUNCTION TO SAVE OUR MODEL OF CLASSIFICATION ASSEMBLY ON A PREVIOUSLY ESTABLISHED ROUTE
   * @param path Path where the model will be saved
   * @param arrayDatos Model that will be a list of integers to be able to save it correctly
   */
  def savePredictionEnsembleClass(path: String, arrayDatos: List[Int]): Unit = {

    val file = path
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))

    arrayDatos.map { i => writer.write(i + "\n") }

    writer.close()

  }

  /**
   * FUNCTION TO SAVE OUR MODEL OF THE REGRESSION ASSEMBLY ON A PREVIOUSLY ESTABLISHED ROUTE
   * @param path Path where the model will be saved
   * @param arrayDatos Model that will be a list of integers to be able to save it correctly
   */
  def savePredictionEnsembleRegr(path: String, arrayDatos: List[Double]): Unit = {

    val file = path
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))

    arrayDatos.map { i => writer.write(i + "\n") }

    writer.close()

  }

  /**
   * FUNCTION TO LOAD OUR MODEL OF THE ASSEMBLY CLASSIFICATION / RETURN FROM THE ROUTE WHERE WE PREVIOUSLY SAVED IT
   * @param spark Object to get Spark functionality
   * @param archivoEntrada Path where the model is located
   * @return Ensemble model dataframe
   */
  def cargaDatosModeloEnsemble(spark: SparkSession, archivoEntrada: String): DataFrame = {

    var DF = spark.read.format("csv")
      .option("sep", ",")
      .option("header", "false")
      .load(archivoEntrada)

    val columnas = DF.dtypes

    //Change numeric values ​​to type Double
    (0 to columnas.length - 1).map { i =>
      DF = DF.withColumn(columnas(i)._1, col(columnas(i)._1).cast(DoubleType))
    }

    DF = DF.withColumnRenamed(columnas(0)._1.toString,"baggingPrediction")

    DF

  }

  /**
   * FUNCTION THAT ALLOWS US TO UNITE ALL THE PREDICTIONS GENERATED BY ALL THE CLASSIFICATION / REGRESSION METHODS USED
   * @param arrayDF Prediction dataframe array
   * @param testData Test dataset
   * @param spark Object to get Spark functionality
   * @return Row Vector RDD of Prediction Array (A vector contains the elements of one of all predictions)
   */
  def unionDF(arrayDF: ArrayBuffer[DataFrame], testData: DataFrame, spark: SparkSession): RDD[DenseVector]  = {

    var DFPredicciones = spark.emptyDataFrame
    DFPredicciones = testData.select("label")
    DFPredicciones = DFPredicciones.withColumn("ID", monotonically_increasing_id()).drop("label")

    (0 to arrayDF.length - 1).map { i =>

      arrayDF(i) = arrayDF(i).withColumn("IDArray", monotonically_increasing_id())
      DFPredicciones = DFPredicciones.join(arrayDF(i), DFPredicciones("ID") === arrayDF(i)("IDArray"),"inner").drop("IDArray").orderBy("ID")

    }

    DFPredicciones = DFPredicciones.drop("ID")

    val assembler = new VectorAssembler().setInputCols(DFPredicciones.columns).setOutputCol("AllPrediction")
    var ensemble = assembler.transform(DFPredicciones).select("AllPrediction")

    val ensembleRdd = ensemble.rdd.map(row => row.getAs[org.apache.spark.ml.linalg.SparseVector]("AllPrediction").toDense)

    ensembleRdd

  }

  /**
   * FUNCTION TO VOTE CLASSIFICATION OF THE PREDICTIONS OF THE CLASSIFICATION METHODS USED
   * @param predictionRDD RDD vector predictions
   * @param nClasses Number of classes in the dataset
   * @return RDD of integers, where each element is the winner by vote
   */
  def clasificacionRow(predictionRDD: RDD[linalg.DenseVector], nClasses: Int): RDD[Int] = {

    val votos = predictionRDD.map { v => var arrayClasses = Array.ofDim[Int](nClasses); v.toArray.map { e => arrayClasses(e.toInt) += 1};
      val indiceMax = arrayClasses.indexOf(arrayClasses.max); indiceMax }

    votos

  }

  /**
   * FUNCTION TO OBTAIN THE AVERAGE OF THE PREDICTIONS OF THE EMPLOYED REGRESSION METHODS
   * @param predictionRDD RDD vector predictions
   * @return RDD of doubles, where each element is the mean of all the elements that made up its row of predictions
   */
  def regressionRow(predictionRDD: RDD[linalg.DenseVector]): RDD[Double] = {

    val media = predictionRDD.map { v => var media = 0.0; v.toArray.map { e =>  media += e }; media = media/v.toArray.length; media }

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

    val RDDPredicciones = unionDF(arrayDF, testData, spark)

    tiempo(timeTransfPredic, "Tiempo unión y transformación de las predicciones en vectores por filas")

    if(classRegr == "clasificacion") {

      val timeClasificacion = System.currentTimeMillis()/1000
      val clasificacion = clasificacionRow(RDDPredicciones, nClasses)
      savePredictionEnsembleClass(paramArchivos(3)  + "baggingModel" + nBagging + ".data", clasificacion.collect().toList)
      tiempo(timeClasificacion, "Tiempo realización clasificación por voto y guardado del resultado en fichero")

    } else if(classRegr == "regresion"){

      val timeRegresion = System.currentTimeMillis()/1000
      val regresion = regressionRow(RDDPredicciones)
      savePredictionEnsembleRegr(paramArchivos(3)  + "baggingModel" + nBagging + ".data", regresion.collect().toList)
      tiempo(timeRegresion, "Tiempo realización regresión por media y guardado del resultado en fichero")

    }

    var baggingModelData = cargaDatosModeloEnsemble(spark, paramArchivos(3) + "baggingModel" + nBagging + ".data")
    baggingModelData = baggingModelData.withColumn("rowID2", monotonically_increasing_id())

    var DFLabelOriginal = spark.emptyDataFrame
    DFLabelOriginal = testData.select("label")
    DFLabelOriginal = DFLabelOriginal.withColumn("rowID1", monotonically_increasing_id())

    val DFBaggingFinal = DFLabelOriginal.as("df1").join(baggingModelData.as("df2"), DFLabelOriginal("rowId1") === baggingModelData("rowId2"), "inner").select("df1.label", "df2.baggingPrediction")

    if(classRegr == "clasificacion") {

      resultadosClasificador(s"RESULTADOS FINALES DEL ENSEMBLE CON CLASIFICACIÓN: ","label","baggingPrediction",DFBaggingFinal,nBagging)

    } else if (classRegr == "regresion") {

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