# Desarrollo de Ensembles

Tratamiento de diversos conjuntos de datos para poder trabajar posteriormente de forma cómoda y más eficientemente, aplicándoles diversos métodos de la librería de ML.
Implementación y aplicación del Ensemble Bagging.

Información sobre [ML](https://spark.apache.org/docs/latest/ml-guide.html). 

## Características

SparkML-Ensemble permite la realización del Ensemble Bagging:

- **Carga y modificación del conjunto de datos**: Hay muchas variedades de conjuntos de datos, por lo que debemos tener una idea universal para poder tratar a todos de una forma u otra con el fin de que puedan ser tratados posteriormente de forma correcta y eficiente tantos por los diversos métodos de ML como por el Ensemble.
- **Particionamiento**: Permite el uso de particionamiento interno de Spark, lo que nos permitiría reducir los tiempos empleados al realizar ejecuciones con los distintos conjuntos de datos.
- ** Empleo de diversos métodos de ML**: En una misma ejecución del software, se pueden emplear diversos métodos de la biblioteca de ML, y que se apliquen tantas veces como deseemos.
- **Clasificación por voto o media**: Se ha desarrollado para que funcione tanto con conjuntos de datos de clasificación como de regresión. Una vez empleados un número determinado de clasificadores o regresores, el programa es capaz de juntar todas las predicciones para realizar una clasificación por voto o una regresión por media, que será la que tome el Ensemble finalmente para obtener un modelo y resultados finales.

## Esquema del Ensemble Bagging

![Estructura del Ensemble Bagging](https://i.ibb.co/L01K0t7/EBagging.jpg)

## Esquema UML del sofware desarrollado

![UML](https://i.ibb.co/sVxwkGC/uml.jpg)

## Archivo .sbt

Hay que configurarlo de la siguiente manera para que funcione correctamente:

**SBT:**

```
name := "SparkML-Ensemble"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.5"
```

## Archivos de parámetros del software

Los archivos de parámetros se encuentran dentro de una carpeta denominada **data**, en la ruta **SparkML-Ensemble/out/artifacts/data**, junto a la carpeta donde se encuentra el archivo **.jar**

**parametrosArchivosES:**

-	Ruta del fichero parámetros método Logistic Regression.
-	Ruta del fichero parámetros método Decision Tree.
-	Ruta del fichero parámetros método Random Forest.
-	Ruta donde se guardarán los modelos obtenidos del Ensemble Bagging.
-	Variable booleana que indica si se usan particiones o no :
		- False: no se usan particiones, y la siguiente ruta que se pone es únicamente la ruta del dataset que contiene interiormente Test y Train juntos.
		- True: se usan particiones, y las siguiente rutas que se ponen son las de las particiones, intercalando una ruta de train y una de test.

**parametrosModelosClass:**

-	Se ponen por fila los nombres de los métodos que se deseen usar: **SVM, MPC, LogisticRegression, NaiveBayes, RandomForest, DecisionTree, GBT, Isotonic**.
    A la hora de escoger métodos, debemos tener en cuenta como es nuestro dataset, ya que unos funcionan para clases binarias, y otros para multiclases. Y en cada línea del archivo se especifica un método.

|             Métodos              |            Tipos de clase           |
| ---------------------------------|-------------------------------------|
| Linear Support Vector Machine    | Binarios                            |
| Multilayer Perceptron Classifier | Binarios / Multiclase               |
| Logistic Regression              | Binarios                            |
| Naive Bayes                      | Binarios / Multiclase / Reales      |
| Decision Tree                    | Binarios / Multiclase / Reales      |
| Gradient Boosted Classifier      | Reales                              |
| Isotonic                         | Reales                              |
| Random Forest                    | Binarios / Multiclase               |

**parametrosNumericos:**

-	Parámetro que indica cuánto porcentaje del dataset real se quedará para Train. Teniendo en cuenta que usemos un dataset con Test y Train juntos.
-	Parámetro que indica cuánto porcentaje del dataset real se quedará para Test. Teniendo en cuenta que usemos un dataset con Test y Train juntos.
-	Número de clasificadores que deseamos que se ejecuten.
-	Porcentaje del dataset de Train que tendrán nuestros subdatasets que se enviarán a cada clasificador.
-	Número de particiones internas de Spark.

**parametrosCR:**

-	Parámetro que indicará si queremos que se realice una clasificación o una regresión en nuestro Ensemble. Se debe poner: clasificacion o regresion.
    En caso de regresion, los métodos del fichero parametrosModelosClass únicamente pueden ser: DecisionTree y RandomForest, ya que son los implementados para este tipo de datasets.	

**parametrosLogisticRegression:**

-	Número máximo de iteraciones
-	Parámetro de regularización
-	Parámetro de mezcla ElasticNet. Para alfa = 0, la penalización es una penalización L2. Para alfa = 1, es una penalización L1. Para alfa en (0,1), la penalización es una combinación de L1 y L2. El valor predeterminado es 0.0, que es una penalización L2	

**parametrosRandomForest:**

-	Número de categorías máximas Vector Indexer
-	Número de árboles	

**parametrosDecisionTree:**

•	Número de categorías máximas Vector Indexer

## Cambios importantes en los ficheros de parámetros

Si no modificamos los ficheros de parámetros de los métodos, se quedarán por defecto como están, pero los que si son importantes cambiar son:

-	Rutas de los datasets a ejecutar.
-	Métodos de modelos que se quieran ejecutar.
-	Número de clasificadores.
-	Porcentaje que se cogerá del conjunto de datos de entrenamiento para los subconjuntos de datos.
-	Número de particiones.

Dentro de IntelliJ IDEA, seleccionaremos nuestro proyecto el cual contiene nuestra aplicación, y deberemos generar un jar para que pueda ser lanzado desde el servidor: **File -> Project Structure -> Artifacts -> + -> jar -> From modules with dependencies.. -> Seleccionamos nuestra clase -> Apply/Ok**
Y debemos construir el jar: **Build -> Build Artifacts -> Seleccionamos jar -> Build**

### Lanzamiento del software en el servidor

Configuración de dos archivos para su correcta ejecución:

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

Muy importante a la hora de lanzar la ejecución en el servidor que la versión que tengamos de Spark Scala tanto en el proyecto Intellij como en el servidor sea idéntica.