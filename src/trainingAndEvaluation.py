from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import DenseVector
from pyspark.sql import Row
import pandas as pd

# Load dataset
df = pd.read_pickle("/content/drive/MyDrive/audiozzi/samplesCompleti.pkl")

# Convert to Spark DataFrame
spark_df = spark.createDataFrame([
    Row(filename=row["filename"],
        features=DenseVector(row["features"]),
        label=int(row["label"]))
    for _, row in df.iterrows()
])

# Split data
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# Evaluation function
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

def train_and_evaluate_model(model, train_df, test_df, model_name="Model"):
    model_trained = model.fit(train_df)
    predictions = model_trained.transform(test_df)
    accuracy = evaluator.evaluate(predictions)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    return model_trained, accuracy

# Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
rf_model, rf_acc = train_and_evaluate_model(rf, train_df, test_df, "Random Forest")

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model, lr_acc = train_and_evaluate_model(lr, train_df, test_df, "Logistic Regression")

# Gradient-Boosted Trees
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50)
gbt_model, gbt_acc = train_and_evaluate_model(gbt, train_df, test_df, "Gradient-Boosted Trees")

