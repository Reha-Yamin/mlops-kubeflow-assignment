from kfp import dsl
from kfp import compiler
import kfp.components as comp


# ---------------------------
# Load component YAML files
# ---------------------------
data_extraction_op = comp.load_component_from_file(
    "components/data_extraction_component.yaml"
)

data_preprocessing_op = comp.load_component_from_file(
    "components/data_preprocessing_component.yaml"
)

model_training_op = comp.load_component_from_file(
    "components/model_training_component.yaml"
)

model_evaluation_op = comp.load_component_from_file(
    "components/model_evaluation_component.yaml"
)


# ---------------------------
# Define the Pipeline
# ---------------------------
@dsl.pipeline(
    name="mlops-housing-pipeline",
    description="End-to-end ML pipeline for Boston Housing dataset",
)
def housing_pipeline(
    repo_url: str = "https://github.com/YOUR-REPO/GOES-HERE.git",
    dvc_data_path: str = "data/raw_data.csv",
):

    # 1️⃣ DATA EXTRACTION
    extract_step = data_extraction_op(
        repo_url=repo_url,
        dvc_data_path=dvc_data_path,
        output_csv_path="data/raw_data_kfp.csv",
    )

    # 2️⃣ DATA PREPROCESSING
    preprocess_step = data_preprocessing_op(
        raw_csv_path=extract_step.outputs["Output"],
        test_size=0.2,
        random_state=42,
    )

    # 3️⃣ MODEL TRAINING
    #    ⚠ use POSITIONAL args for X_train_path, y_train_path to avoid kw-name issues
    train_step = model_training_op(
        preprocess_step.outputs["X_train_path"],   # X_train_path (positional)
        preprocess_step.outputs["y_train_path"],   # y_train_path (positional)
        n_estimators=100,
        max_depth=5,
        model_path="models/random_forest.joblib",
    )

    # 4️⃣ MODEL EVALUATION
    #    ⚠ again, use POSITIONAL for X_test_path, y_test_path
    evaluate_step = model_evaluation_op(
        train_step.outputs["Output"],                  # model_path
        preprocess_step.outputs["X_test_path"],        # X_test_path
        preprocess_step.outputs["y_test_path"],        # y_test_path
        metrics_path="metrics/metrics.json",
    )


# ---------------------------
# Compile pipeline to YAML
# ---------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=housing_pipeline,
        package_path="pipeline.yaml",
    )
    print("✅ pipeline.yaml generated successfully!")
