import accelerate
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TestTrainShardIter").getOrCreate()
accelerate = accelerate.Accelerator()

num_output_partitions = [1, 2, 8, 3]


# generate_sequence_dataset(
#     spark,
#     num_records=10_000,
#     num_output_partitions=8,
#     num_events_range=(0, 100),
#     target_column=None,
#     tabular_features=False,
# )
