"""Sequential feature selection paralellized through PySpark."""

import argparse
import datetime

import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

parser = argparse.ArgumentParser(
    prog="Recursive feature elimination using Random Forest feature importances.",
    add_help=True,
    allow_abbrev=False,
)
parser.add_argument(
    "--input",
    action="store",
    help="A local parquet with epk_id, report_dt, target, and feature columns.",
    required=True,
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
    "--subsample",
    action="store",
    help="Subsample size to use to speed up.",
    required=True,
    type=int,
)
args = parser.parse_args()

conf = (
    SparkConf()
    .setAppName(r"//\(oo)/\\")
    .setMaster("yarn")
    .set("spark.executor.cores", 2)
    .set("spark.executor.memory", "4g")
    .set("spark.executor.memoryOverhead", "2g")
    .set("spark.memory.storageFraction", 0)  # No caching in this app
    .set("spark.shuffle.service.enabled", "true")
    .set("spark.dynamicAllocation.enabled", "true")
    .set("spark.dynamicAllocation.executorIdleTimeout", "10m")
    .set("spark.dynamicAllocation.initialExecutors", 3)
    .set("spark.dynamicAllocation.maxExecutors", 55)
    .set("spark.port.maxRetries", 150)
    .set("spark.ui.showConsoleProgress", "true")
)
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# The current time is used to generate unique names for artifacts
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%dT%H-%M-%S")

data = pd.read_parquet(args.input)
data.set_index(["epk_id", "report_dt"], inplace=True)

potential_feature_set = set(data.columns) - {"epk_id", "report_dt", "target"}
selected_features = []
score_history = []

assert len(potential_feature_set) > args.num_features

# The preprocessor has to cover all possible column combinations
preprocessor = make_column_transformer(
    (
        "passthrough",
        make_column_selector(pattern="_nflag"),
    ),
    (
        SimpleImputer(strategy="median"),
        make_column_selector(pattern="_amt|_nv|_rate"),
    ),
    (
        SimpleImputer(strategy="constant", fill_value=0),
        make_column_selector(pattern="_qty"),
    ),
    (
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        ),
        make_column_selector(dtype_include=object),
    ),
    verbose_feature_names_out=False,
    remainder="drop",
)

clf = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=5,
    class_weight="balanced",
)

pipeline = make_pipeline(preprocessor, clf)

distributed_data = spark.sparkContext.broadcast(data.sample(args.subsample))
distributed_pipeline = spark.sparkContext.broadcast(pipeline)


def get_feature_score(feature):
    """Run cross-validation to get the score of the feature."""

    data = distributed_data.value
    pipeline = distributed_pipeline.value

    scores = cross_val_score(
        estimator=pipeline,
        X=data[selected_features + [feature]],
        y=data["target"],
        scoring="f1",
        cv=5,
    )

    return feature, scores.mean()


for i in range(args.num_features):
    n = len(potential_feature_set)
    tasks = spark.sparkContext.parallelize(potential_feature_set, numSlices=n)
    results = tasks.map(get_feature_score).collect()
    df = pd.DataFrame(results, columns=["feature", "score"])

    argmax = df["score"].argmax()
    best_feature = df.loc[argmax, "feature"]
    best_score = df.loc[argmax, "score"]

    selected_features.append(best_feature)
    potential_feature_set.remove(best_feature)
    score_history.append(best_score)

    print(f"Selected feature: {best_feature:<40} [f1: {best_score:.4f}]")

    # Useless feature can be removed as well, but:
    #  1) They don't slow the calculation too much
    #  2) They might be activated in a combination with some other features
    #
    # useless_features = set(df.loc[df.score == 0, "feature"])
    # potential_feature_set.difference_update(useless_features)
    #

out = pd.Series(score_history, index=selected_features, name="f1")
out.to_csv(f"reports/sfs-{timestamp}-selected-features.csv", index=True)
