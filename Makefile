python = /path/to/python/interpreter
spark-submit = /path/to/spark-submit

input = input_hive_table
output = output_hive_table
model_path = /hdfs/path/to/model


default:
	@echo "Profits are stolen wages"


rfe:
	$(spark-submit) \
	 --driver-memory 6g \
	 --conf "spark.pyspark.python=$(python)" \
	 --conf "spark.pyspark.driver.python=$(python)" \
	recursive_feature_elimination.py \
	 --input $(input) \
	 --output $(output) \
	 --model-path $(model_path) \
	 --num-features 50 \
	 --step 0.1 \
	 --num-trees 300 \
	 --max-depth 7 \
	 --max-bins 128 \
	 --min-instances-per-node 5
