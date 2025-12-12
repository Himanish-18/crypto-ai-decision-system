import argparse
import logging
from pathlib import Path

import onnx
import tensorflow as tf
import tf2onnx
import torch

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onnx_converter")


def convert_keras(model_path: Path, output_path: Path):
    logger.info(f"🔄 Converting Keras model {model_path} to ONNX...")
    try:
        model = tf.keras.models.load_model(model_path)

        # Define input signature (Batch size, 30, 5) for TinyCNN example
        spec = (tf.TensorSpec((None, 30, 5), tf.float32, name="input"),)

        # Convert
        model_proto, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=13
        )

        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())

        logger.info(f"✅ Saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to convert Keras: {e}")


def convert_torch(model_path: Path, output_path: Path):
    logger.info(f"🔄 Converting PyTorch model {model_path} to ONNX...")
    try:
        # Load Model Class (Requires import, mocking for simplicity)
        # In real usage, we need the class def.
        # Assuming TorchScript or JIT for now, or just placeholder.
        logger.warning(
            "PyTorch conversion requires model class definition. Skipping for generic script."
        )
    except Exception as e:
        logger.error(f"Failed to convert PyTorch: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to input model")
    parser.add_argument("--type", type=str, choices=["keras", "torch"], default="keras")
    args = parser.parse_args()

    input_p = Path(args.model)
    output_p = input_p.with_suffix(".onnx")

    if args.type == "keras":
        convert_keras(input_p, output_p)
    else:
        convert_torch(input_p, output_p)
