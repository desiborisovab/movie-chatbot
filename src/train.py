import json
import numpy as np
import mlflow
import mlflow.tensorflow
from mlflow.models import infer_signature

from src.model import build_model, load_config
from src.data_prep import prepare_datasets

def train(spark):
    config = load_config()

    train_ds_rep, val_ds_rep, vocab = prepare_datasets(spark, config)
    model = build_model(vocab_size=len(vocab), config=config)

    history = model.fit(
        train_ds_rep,
        validation_data=val_ds_rep,
        steps_per_epoch=config["STEPS_PER_EPOCH"],
        validation_steps=config["VAL_STEPS"],
        epochs=config["EPOCHS"],
    )

    mlflow.set_experiment(config["MLFLOW_EXPERIMENT_PATH"])

    input_example = np.zeros((1, config["SEQ_LEN"]), dtype=np.int32)
    pred_example = model.predict(input_example, verbose=0)
    signature = infer_signature(input_example, pred_example)

    with mlflow.start_run(run_name=config.get("RUN_NAME", "movie_lm")) as run:
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        mlflow.log_text(json.dumps(vocab), "vocab.json")

        for k, v in config.items():
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(k, v)
            else:
                mlflow.log_text(json.dumps(v), f"config_{k}.json")

        run_id = run.info.run_id

    return model, history, run_id
