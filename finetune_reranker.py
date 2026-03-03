"""
Fine-tune the cross-encoder reranker on custom data.

This script demonstrates how to fine-tune a cross-encoder model for
query-document relevance scoring. Replace the example data with your
own training dataset (typically from CSV, JSONL, or HuggingFace Datasets).

Usage:
    python finetune_reranker.py --output-dir ./my-model --epochs 3
"""

import argparse
import logging
import math
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_example_data() -> list:
    """Return example training data.

    In production, replace this with loading from your data source.
    Each InputExample has texts=[query, document] and a float label
    (1.0 for relevant, 0.0 for irrelevant/hard negative).
    """
    return [
        InputExample(
            texts=[
                "What is the speed of light?",
                "The speed of light in vacuum is approximately 299,792 kilometers per second.",
            ],
            label=1.0,
        ),
        InputExample(
            texts=[
                "What is the speed of light?",
                "Albert Einstein was a German-born theoretical physicist.",
            ],
            label=0.0,
        ),
        InputExample(
            texts=[
                "Who wrote Hamlet?",
                "The Tragedy of Hamlet, Prince of Denmark, often shortened to Hamlet, "
                "is a tragedy written by William Shakespeare.",
            ],
            label=1.0,
        ),
        InputExample(
            texts=[
                "Who wrote Hamlet?",
                "A hamlet is a small human settlement.",
            ],
            label=0.0,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a cross-encoder reranker model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Base model to fine-tune (HuggingFace model ID)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="custom-cross-encoder-model",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Ratio of training steps for learning rate warmup",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_path.absolute())

    # Load model
    logger.info("Loading base model: %s", args.model)
    try:
        model = CrossEncoder(args.model, num_labels=1)
    except Exception as e:
        logger.error("Failed to load model %s: %s", args.model, e)
        sys.exit(1)

    # Load training data
    train_examples = get_example_data()
    logger.info("Loaded %d training examples", len(train_examples))

    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size
    )

    # Calculate warmup steps
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * args.warmup_ratio)
    logger.info(
        "Training config: epochs=%d, batch_size=%d, warmup_steps=%d",
        args.epochs,
        args.batch_size,
        warmup_steps,
    )

    # Run training
    logger.info("Starting fine-tuning...")
    try:
        model.fit(
            train_dataloader=train_dataloader,
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            output_path=str(output_path),
            use_amp=True,  # Mixed precision training
        )
        logger.info("Fine-tuning complete! Model saved to: %s", output_path.absolute())
    except Exception as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
