from __future__ import annotations

from bank_predictor.config import DATA_PATH, METADATA_PATH, MODEL_PATH
from bank_predictor.ml import load_dataset, save_artifacts, train_model


def main() -> None:
    df = load_dataset(DATA_PATH)
    pipeline, metadata = train_model(df)
    save_artifacts(pipeline, metadata)

    print(f"Dataset shape: {df.shape}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(
        "Metrics: "
        f"accuracy={metadata['metrics']['accuracy']:.4f}, "
        f"precision={metadata['metrics']['precision']:.4f}, "
        f"recall={metadata['metrics']['recall']:.4f}, "
        f"f1={metadata['metrics']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()

