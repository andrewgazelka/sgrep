"""Convert Jina ColBERT v2 to CoreML format for ANE acceleration."""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig


class JinaBertEncoder(torch.nn.Module):
    """Wrapper that outputs last_hidden_state (per-token embeddings)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # [batch, seq_len, hidden_dim]


def convert_to_onnx(
    model_name: str,
    output_path: Path,
    max_seq_length: int = 512,
) -> Path:
    """Convert a Jina ColBERT model to ONNX format.

    Args:
        model_name: HuggingFace model name (e.g., "jinaai/jina-colbert-v2")
        output_path: Base path (will add .onnx extension)
        max_seq_length: Maximum sequence length for the model

    Returns:
        Path to the generated ONNX file
    """
    print(f"Loading model: {model_name}")

    # Load config and disable flash attention (not available without CUDA)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "use_flash_attn"):
        config.use_flash_attn = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
    model.eval()

    encoder = JinaBertEncoder(model)
    encoder.eval()

    # Create dummy inputs for export
    dummy_text = "This is a sample input for tracing the model."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    onnx_path = output_path.with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (input_ids, attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["token_embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "token_embeddings": {0: "batch", 1: "seq_len"},
            },
            opset_version=18,
            do_constant_folding=True,
        )

    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def convert_onnx_to_coreml(
    onnx_path: Path,
    output_path: Path,
    max_seq_length: int = 512,
) -> None:
    """Convert an ONNX model to CoreML format.

    Args:
        onnx_path: Path to the ONNX model
        output_path: Path to save the .mlpackage
        max_seq_length: Maximum sequence length for the model
    """
    import onnx

    print(f"Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(str(onnx_path))

    print("Converting ONNX to CoreML...")
    mlmodel = ct.convert(
        onnx_model,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, max_seq_length),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, max_seq_length),
                dtype=np.int32,
            ),
        ],
        outputs=[
            ct.TensorType(name="token_embeddings"),
        ],
        compute_units=ct.ComputeUnit.ALL,  # Use ANE when available
        convert_to="mlprogram",
    )

    # Add metadata
    mlmodel.author = "sgrep"
    mlmodel.short_description = "Jina BERT encoder for token embeddings"
    mlmodel.version = "1.0"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving CoreML model to: {output_path}")
    mlmodel.save(str(output_path))
    print("Done!")


def convert_to_coreml(
    model_name: str,
    output_path: Path,
    max_seq_length: int = 512,
) -> None:
    """Convert a Jina ColBERT model to CoreML format via ONNX.

    Args:
        model_name: HuggingFace model name (e.g., "jinaai/jina-colbert-v2")
        output_path: Path to save the .mlpackage
        max_seq_length: Maximum sequence length for the model
    """
    # Step 1: Export to ONNX
    onnx_path = convert_to_onnx(model_name, output_path, max_seq_length)

    # Step 2: Convert ONNX to CoreML
    convert_onnx_to_coreml(onnx_path, output_path, max_seq_length)

    # Optionally remove ONNX intermediate file
    # onnx_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Jina BERT models to CoreML format"
    )
    parser.add_argument(
        "--model",
        default="jinaai/jina-colbert-v2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/jina-colbert-v2.mlpackage"),
        help="Output path for the CoreML model",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Only export to ONNX (skip CoreML conversion)",
    )

    args = parser.parse_args()

    if args.onnx_only:
        convert_to_onnx(args.model, args.output, args.max_seq_length)
    else:
        convert_to_coreml(args.model, args.output, args.max_seq_length)


if __name__ == "__main__":
    main()
