"""Recursive feature elimination for binary classification in PySpark.

Notes:
    For regression: use pyspark.ml.feature.Bucketizer to split the target into buckets,
    then stratify and split.

"""

import argparse
import datetime
from collections import defaultdict

import pandas as pd
import pyspark.sql.functions as F
from pyspark import SparkConf
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from sklearn.utils import class_weight

parser = argparse.ArgumentParser(
    prog="Recursive feature elimination using Random Forest feature importances.",
    add_help=True,
    allow_abbrev=False,
)
parser.add_argument(
    "--input",
    action="store",
    help="A Hive table with epk_id, report_dt, target, and potential feature columns.",
    required=True,
)
parser.add_argument(
    "--output",
    action="store",
    help="A Hive table to write the selected features to.",
    required=True,
)
parser.add_argument(
    "--model-path",
    action="store",
    help="Folder on HDFS to save the last model to.",
    required=False,
)
parser.add_argument(
    "--num-features",
    action="store",
    help="Number of features to select.",
    required=False,
    type=int,
    default=1,
)
parser.add_argument(
    "--step",
    action="store",
    help="Percentage of feature to remove at each iteration.",
    required=False,
    type=float,
    default=0.2,
)
parser.add_argument(
    "--num-trees",
    action="store",
    help="Number of trees to train.",
    required=False,
    type=int,
    default=200,
)
parser.add_argument(
    "--max-depth",
    action="store",
    help="Maximum depth of a tree.",
    required=False,
    type=int,
    default=7,
)
parser.add_argument(
    "--max-bins",
    action="store",
    help="Maximum number of bins for discretizing continuous features.",
    required=False,
    type=int,
    default=128,
)
parser.add_argument(
    "--min-instances-per-node",
    action="store",
    help="Minumum number of instances each child must have after split.",
    required=False,
    type=int,
    default=5,
)
args = parser.parse_args()

conf = (
    SparkConf()
    .setAppName(r"//\(oo)/\\")
    .setMaster("yarn")
    .set("spark.executor.cores", 2)
    .set("spark.executor.memory", "6g")
    .set("spark.executor.memoryOverhead", "1g")
    # .set("spark.driver.memory", "6g")  # spark-submit --driver-memory ...
    .set("spark.driver.maxResultSize", "4g")
    .set("spark.shuffle.service.enabled", "true")
    .set("spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive", "true")
    .set("spark.dynamicAllocation.enabled", "true")
    .set("spark.dynamicAllocation.executorIdleTimeout", "10m")
    .set("spark.dynamicAllocation.initialExecutors", 3)
    .set("spark.dynamicAllocation.maxExecutors", 35)
    .set("spark.dynamicAllocation.cachedExecutorIdleTimeout", "60m")
    .set("spark.port.maxRetries", 150)
    .set("spark.ui.showConsoleProgress", "true")
)
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# The current time is used to generate unique names for artifacts
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%dT%H-%M-%S")

data = spark.read.table(args.input)

# Split the data randomly into 5 equal groups, stratifying by the target
window = Window.partitionBy("target").orderBy(F.rand(seed=now.microsecond))
groups = (
    data.select("epk_id", "report_dt", "target")
    .withColumn("__tile__", F.ntile(5).over(window))
    .cache()
)

# Adding the  __tile__ column directly to `data` and splitting into train and test by
# filtering on that column does not guarantee that there will be no intersection even
# with a fixed random seed. I am not entirely sure how this works, but it probably has
# something to do with how partitions are assigned to executors.

# Trigger the computation and materialization of the table with groups. If it's not
# materialized, the call to F.rand() in the window definition is done separately for
# train and test sets, making the split essentially random with replacement
groups.groupBy("target", "__tile__").count().orderBy("target", "__tile__").show()

# Use one of the groups (1/5 = 20%) as the hold-out test set
test = groups.filter(F.col("__tile__") == 1)
train = groups.filter(F.col("__tile__") != 1)
test = data.join(test, on=["epk_id", "report_dt"], how="left_semi").cache()
train = data.join(train, on=["epk_id", "report_dt"], how="left_semi").cache()

# Trigger the computation and materialization of the train and test subsets
intersection = train.join(test, on=["epk_id", "report_dt"], how="inner").count()
assert intersection == 0, "Non zero intersection between train and test!"

groups.unpersist()

# Assign a weight to each sample (How to do this in the multi-class case? In a loop?..)
y = train.select("target").toPandas().target
weights = class_weight.compute_class_weight("balanced", classes=[0, 1], y=y)
train = train.withColumn(
    "sample_weight",
    F.when(F.col("target") == 0, weights[0]).otherwise(weights[1]),
)

# Random Forest because it has feature importances built in
rfc = RandomForestClassifier(
    featuresCol="features",
    labelCol="target",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    weightCol="sample_weight",
    numTrees=args.num_trees,
    maxDepth=args.max_depth,
    maxBins=args.max_bins,
    minInstancesPerNode=args.min_instances_per_node,
)

# Metrics are spread out across two evaluator objects: binary (AUC ROC, AUC PR) and
# multiclass (accuracy, precision, recall, f1). Calculate multiclass ones for class 1
binary_evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="target",
)
multi_evaluator = MulticlassClassificationEvaluator(
    predictionCol="prediction",
    labelCol="target",
    probabilityCol="probability",
    metricLabel=1.0,
)

# Initial set of features is all columns in the provided table
current_feature_set = data.drop("epk_id", "report_dt", "target").columns

importance_history = []  # List of pd.Series of feature importances
metric_history = defaultdict(list)  # Dict of lists of scalars metric

while True:
    print(f"Testing {len(current_feature_set)} features")
    metric_history["n_features"].append(len(current_feature_set))

    types = train.select(*current_feature_set).dtypes  # [("column_name", "dtype"), ...]

    # Spark ML algorithms are on a strict no string diet
    string_cols = [x[0] for x in types if x[1] == "string"]
    indexed_string_cols = [f"{x}_indexed" for x in string_cols]

    indexer = StringIndexer(
        inputCols=string_cols,
        outputCols=indexed_string_cols,
        stringOrderType="frequencyDesc",
        handleInvalid="keep",
    )

    # Vector assemblers are on a strict no null values diet
    numerical_cols = [x[0] for x in types if x[1] != "string"]
    imputed_numerical_cols = [f"{x}_imputed" for x in numerical_cols]

    imputer = Imputer(
        inputCols=numerical_cols,
        outputCols=imputed_numerical_cols,
        strategy="median",
    )

    features = current_feature_set.copy()
    features = [f"{c}_indexed" if c in string_cols else c for c in features]
    features = [f"{c}_imputed" if c in numerical_cols else c for c in features]

    assembler = VectorAssembler(
        inputCols=features,
        outputCol="features",
    )

    pipeline = Pipeline(
        stages=[
            indexer,
            imputer,
            assembler,
            rfc,
        ]
    )

    model = pipeline.fit(train)

    train_prediction = model.transform(train).cache()
    test_prediction = model.transform(test).cache()

    for metric in ("areaUnderROC", "areaUnderPR"):
        binary_evaluator.setMetricName(metric)
        train_value = binary_evaluator.evaluate(train_prediction)
        test_value = binary_evaluator.evaluate(test_prediction)
        metric_history[f"train_{metric}"].append(train_value)
        metric_history[f"test_{metric}"].append(test_value)

    for metric in ("accuracy", "precisionByLabel", "recallByLabel", "fMeasureByLabel"):
        multi_evaluator.setMetricName(metric)
        train_value = multi_evaluator.evaluate(train_prediction)
        test_value = multi_evaluator.evaluate(test_prediction)
        metric_history[f"train_{metric}"].append(train_value)
        metric_history[f"test_{metric}"].append(test_value)

    train_prediction.unpersist()
    test_prediction.unpersist()

    feature_importance = pd.Series(
        data=model.stages[-1].featureImportances.toArray(),
        index=current_feature_set,
    )
    importance_history.append(feature_importance)

    if len(current_feature_set) == args.num_features:
        break

    if len(current_feature_set) * (1 - args.step) > args.num_features:
        threshold = feature_importance.quantile(args.step)
        worst = feature_importance[feature_importance <= threshold].index
        current_feature_set = [f for f in current_feature_set if f not in worst]
    else:
        current_feature_set = feature_importance.sort_values()[-args.num_features :]
        current_feature_set = current_feature_set.index.tolist()

full_importance_history = pd.concat(importance_history, axis="columns")
full_importance_history.to_csv(f"reports/rfe-{timestamp}-importances.csv")

full_history = pd.DataFrame(metric_history).melt(id_vars="n_features")
full_history[["set", "metric"]] = full_history["variable"].str.split("_", expand=True)
full_history = full_history[["n_features", "set", "metric", "value"]]
full_history.to_csv(f"reports/rfe-{timestamp}-metrics.csv", index=False)

if args.model_path is not None:
    model.save(f"{args.model_path}/model-rfe-{timestamp}")

out = data.select("epk_id", "report_dt", "target", *current_feature_set)
out.write.saveAsTable(args.output, mode="overwrite")
