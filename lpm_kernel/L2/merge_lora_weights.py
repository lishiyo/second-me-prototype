"""Utility for merging LoRA weights into base language models.

This module provides functions to merge trained LoRA adapter weights with a base model,
producing a standalone model that incorporates the adaptations without needing the
LoRA architecture during inference.
"""

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

def merge_lora_weights(base_model_path, lora_adapter_path, output_model_path):
    """Merge LoRA weights into a base model and save the result.
    
    This function loads a base model and a LoRA adapter, merges them together,
    and saves the resulting model to the specified output path.
    
    Args:
        base_model_path: Path to the base model directory.
        lora_adapter_path: Path to the LoRA adapter directory.
        output_model_path: Path where the merged model will be saved.
    """
    # Load the base model
    logging.info(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load the LoRA adapter and apply it to the base model
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Merge LoRA weights into the base model
    merged_model = lora_model.merge_and_unload()

    # Save the merged model and tokenizer
    merged_model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)


def merge_model_weights(
    base_model_path="resources/L2/base_models",
    lora_adapter_path="resources/model/output/personal_model",
    output_model_path="resources/model/output/merged_model",
):
    """Merge LoRA weights into base model with default paths.
    
    This is a convenience function that calls merge_lora_weights with default 
    paths that match the expected directory structure of the project.

    Args:
        base_model_path: Path to the base model. Defaults to "resources/L2/base_models".
        lora_adapter_path: Path to the LoRA adapter. Defaults to "resources/model/output/personal_model".
        output_model_path: Path to save the merged model. Defaults to "resources/model/output/merged_model".
    """
    merge_lora_weights(base_model_path, lora_adapter_path, output_model_path)


def parse_arguments():
    """Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into a base model."
    )
    parser.add_argument(
        "--base_model_path", type=str, required=True, help="Path to the base model."
    )
    parser.add_argument(
        "--lora_adapter_path", type=str, required=True, help="Path to the LoRA adapter."
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        required=True,
        help="Path to save the merged model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    merge_lora_weights(
        args.base_model_path, args.lora_adapter_path, args.output_model_path
    )
