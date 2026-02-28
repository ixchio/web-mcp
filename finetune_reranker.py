"""
Example script to fine-tune the cross-encoder model on your own Custom Data.
"""
import math
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

def main():
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # num_labels=1 is used for regression (relevance scoring)
    model = CrossEncoder(model_name, num_labels=1)

    # Example: Prepare your custom dataset
    # You would typically load this from a CSV, JSONL, or HuggingFace Dataset.
    # An InputExample consists of texts=[query, document] and a float label 
    # (e.g., 1.0 for relevant, 0.0 for irrelevant/hard negative).
    train_examples = [
        InputExample(texts=["What is the speed of light?", "The speed of light in vacuum is approximately 299,792 kilometers per second."], label=1.0),
        InputExample(texts=["What is the speed of light?", "Albert Einstein was a German-born theoretical physicist."], label=0.0),
        InputExample(texts=["Who wrote Hamlet?", "The Tragedy of Hamlet, Prince of Denmark, often shortened to Hamlet, is a tragedy written by William Shakespeare."], label=1.0),
        InputExample(texts=["Who wrote Hamlet?", "A hamlet is a small human settlement."], label=0.0),
    ]

    # Create a DataLoader for training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

    num_epochs = 1
    # Typically 10% of train data for warm-up
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) 

    print(f"Starting fine-tuning on {model_name}...")
    
    # Run the training loop
    model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path="custom-cross-encoder-model",
        use_amp=True # Mixed precision training
    )
    
    print("Fine-tuning complete. Model saved to 'custom-cross-encoder-model'")

if __name__ == "__main__":
    main()
