# Authors

- María Dolores Pérez Godoy: lperez@ujaen.es
- Antonio Jesús Rivera Rivas: arivera@ujaen.es
- Alberto Moreno Molina: amm00254@red.ujaen.es

# Ensembles Development

Treatment of different data sets to be able to work later comfortably and more efficiently, applying various methods from the ML library.
Implementation and application of the Ensemble Bagging.

Information about [ML](https://spark.apache.org/docs/latest/ml-guide.html). 

## Features

SparkML-Ensemble allows the realization of the Ensemble Bagging:

- **Loading and modifying the dataset**: There are many varieties of data sets, so we must have a universal idea to be able to treat everyone in one way or another so that they can be subsequently treated correctly and efficient as much for the various ML methods as for the Ensemble.
- **Partitions**: Allows the use of internal Spark partitioning, which would allow us to reduce the time spent executing with different data sets.
- **Use of various ML methods**: In the same execution of the software, you can use various methods of the ML library, and they are applied as many times as we want.
- **Classification by vote or average**: It has been developed to work with both classification and regression datasets. Once a certain number of classifiers or regressors are used, the program is able to gather all the predictions to carry out a classification by vote or a regression by average, which will be the one that the Ensemble will finally take to obtain a model and final results.

## Ensemble Bagging Outline

![Ensemble Bagging Structure](https://i.ibb.co/L01K0t7/EBagging.jpg)

## UML scheme of the developed software

![UML](https://i.ibb.co/sVxwkGC/uml.jpg)

## File .sbt

You have to configure it as follows to work correctly:

**SBT:**

```
name := "SparkML-Ensemble"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.5"
```

## Software parameter files

The parameter files are located in a folder named **data**, at the path **SparkML-Ensemble/out/artifacts/data**, next to the folder where the file **.jar** is located.

**parametrosArchivosES:**

-	Path of the file parameters method Logistic Regression.
-	Path of the file parameters method Decision Tree.
-	Path of the file parameters method Random Forest.
-	Path where the models obtained from the Ensemble Bagging will be stored.
-	Boolean variable that indicates whether partitions are used or not:
		- False: no partitions are used, and the next path to put is only the path of the dataset that contains Test and Train together.
		- True: partitions are used, and the next routes that are put are those of the partitions, interspersing a train route and a test route.

**parametrosModelosClass:**

-	The names of the methods you want to use are put by row: **SVM, MPC, LogisticRegression, NaiveBayes, RandomForest, DecisionTree, GBT, Isotonic**.
    When choosing methods, we must take into account how our dataset is, since some work for binary classes, and others for multiclasses. And in each line of the file a method is specified.

|              Methods             |             Class types             |
| ---------------------------------|-------------------------------------|
| Linear Support Vector Machine    | Binary                              |
| Multilayer Perceptron Classifier | Binary / Multiclass                 |
| Logistic Regression              | Binary                              |
| Naive Bayes                      | Binary / Multiclass / Real          |
| Decision Tree                    | Binary / Multiclass / Real          |
| Gradient Boosted Classifier      | Real                                |
| Isotonic                         | Real                                |
| Random Forest                    | Binary / Multiclass                 |

**parametrosNumericos:**

-	Parameter indicating how much percentage of the actual dataset will remain for Train. Taking into account that we use a dataset with Test and Train together.
-	Parameter indicating how much percentage of the actual dataset will remain for Test. Taking into account that we use a dataset with Test and Train together.
-	Number of classifiers that we want to run.
-	Percentage of the Train dataset that will have our subdatasets that will be sent to each classifier.
-	Number of internal Spark partitions.

**parametrosCR:**

-	Parameter that will indicate if we want a classification or a regression to be performed in our Ensemble. It should be put: classification or regression.
    In case of regression, the methods of the file parametrosModelosClass can only be: DecisionTree and RandomForest, since they are the ones implemented for this type of datasets.

**parametrosLogisticRegression:**

-	Maximum number of iterations
-	Regularization parameter
-	ElasticNet mixing parameter. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty. For alpha at (0,1), the penalty is a combination of L1 and L2. The default value is 0.0, which is an L2 penalty

**parametrosRandomForest:**

-	Number of maximum categories Vector Indexer
-	Number of trees

**parametrosDecisionTree:**

-	Number of maximum categories Vector Indexer

## Important changes in the parameter files

If we do not modify the parameter files of the methods, they will remain by default as they are, but the ones that are important to change are:

-	Dataset paths to execute.
-	Model methods that you want to execute.
-	Number of classifiers.
-	Percentage to be taken from the training dataset for the data subsets.
-	Number of partitions.

Within IntelliJ IDEA, we will select our project which contains our application, and we must generate a jar so that it can be launched from the server: **File -> Project Structure -> Artifacts -> + -> jar -> From modules with dependencies.. -> Select our class -> Apply/Ok**
And we must build the jar: **Build -> Build Artifacts -> Select jar -> Build**

### Software release on the server

Configuration of two files for correct execution:

**.sh:**

```
$SPARK_HOME/bin/spark-submit --master spark://bigdata:7077 path_jar
```

**.sbs:**

```
#!/bin/bash

$SBATCH --job-name=name
$SBATCH --partition=spark
$SBATCH --output=path/salidaResultado_%j.out
$SBATCH --error=path/salidaResultado_%j.err

path/archivo.sh
```

Very important when launching the execution on the server that the version we have of Spark Scala both in the Intellij project and on the server is identical.