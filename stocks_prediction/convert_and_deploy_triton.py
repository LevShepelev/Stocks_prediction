# convert_and_deploy_triton.py
# Script to convert an ONNX StockLSTM model to TensorRT engine and prepare Triton model repository

import os
import argparse
import tensorrt as trt
from importlib.metadata import version
pkg_version = version('tensorrt')
print(pkg_version)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path: str, trt_path: str, max_batch: int, seq_len: int, feature_dim: int):
    """
    Convert ONNX model to TensorRT engine and save.

    :param onnx_path: Path to ONNX file
    :param trt_path: Output path for TensorRT engine (.plan)
    :param max_batch: Maximum batch size
    :param seq_len: Sequence length dimension
    :param feature_dim: Number of features per timestep
    """
    # Create builder, network and parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        (1, seq_len, feature_dim),
        (max_batch, seq_len, feature_dim),
        (max_batch, seq_len, feature_dim),
    )
    config.add_optimization_profile(profile)

    # Build serialized network (TensorRT >=8)
    print("Building TensorRT engine (serialized network)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT serialized network build failed")

    # Deserialize to engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    # Serialize engine to disk
    with open(trt_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to: {trt_path}")


def create_triton_model_repo(model_name: str, trt_path: str, seq_len: int, feature_dim: int, max_batch: int, repo_dir: str):
    """
    Create Triton model repository structure with config.pbtxt and engine file.
    """
    model_dir = os.path.join(repo_dir, model_name)
    version_dir = os.path.join(model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)

    # Move .plan file
    dest_plan = os.path.join(version_dir, "model.plan")
    os.replace(trt_path, dest_plan)

    # Write Triton config
    config_pbtxt = f"""
name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch}
input [
  {{
    name: "X"
    data_type: TYPE_FP32
    dims: [ {seq_len}, {feature_dim} ]
  }}
]
output [
  {{
    name: "Y"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [ {{ kind: KIND_GPU }} ]
"""
    with open(os.path.join(model_dir, "config.pbtxt"), 'w') as f:
        f.write(config_pbtxt)
    print(f"Triton config written to: {os.path.join(model_dir, 'config.pbtxt')}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX to TensorRT and prepare Triton model repository"
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX model file")
    parser.add_argument("--max-batch", type=int, default=32, help="Max batch size for Triton")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length (model input dim)")
    parser.add_argument("--feature-dim", type=int, default=5, help="Number of features per timestep")
    parser.add_argument("--trt-path", default="stocklstm.plan", help="Output path for TensorRT engine (.plan)")
    parser.add_argument("--model-name", default="stocklstm", help="Triton model name")
    parser.add_argument("--repo-dir", default="model_repository", help="Triton model repository root")
    args = parser.parse_args()

    build_engine(
        onnx_path=args.onnx,
        trt_path=args.trt_path,
        max_batch=args.max_batch,
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
    )

    create_triton_model_repo(
        model_name=args.model_name,
        trt_path=args.trt_path,
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
        max_batch=args.max_batch,
        repo_dir=args.repo_dir,
    )

if __name__ == "__main__":
    main()