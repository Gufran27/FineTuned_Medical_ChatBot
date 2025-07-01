# FineTuned_Medical_ChatBot

Medical Chatbot Fine-Tuning with LLaMA-2
This project fine-tunes the NousResearch/Llama-2-7b-chat-hf model using LoRA (Low-Rank Adaptation) on a medical dataset to create a specialized medical chatbot. The script processes a JSONL dataset, applies quantization for efficiency, and saves the fine-tuned model and tokenizer. It also includes an interactive interface for generating responses to medical queries.
Features

Fine-Tuning: Uses LoRA to fine-tune the LLaMA-2 model on a medical dataset.
Quantization: Implements 4-bit quantization with bitsandbytes for efficient training.
Prompt Engineering: Injects a medical domain-specific system prompt to ensure relevant responses.
Interactive Interface: Allows users to input medical questions and receive generated responses.
Dataset Processing: Handles JSONL datasets with tokenization and splitting for training and evaluation.
Model Saving: Saves the fine-tuned model, tokenizer, and metadata for reuse.

Prerequisites

Python 3.8+
CUDA-compatible GPU (recommended for faster training)
Kaggle environment or equivalent with sufficient storage (up to 20GB for outputs)
Required libraries: transformers, datasets, peft, bitsandbytes, torch, accelerate, numpy, pandas, tqdm, pickle

Installation

Clone the Repository:
git clone https://github.com/your-username/medical-chatbot-llama2.git
cd medical-chatbot-llama2


Install Dependencies:Ensure you have pip installed, then run:
pip install -q transformers datasets peft bitsandbytes torch accelerate numpy pandas tqdm


Set Up Environment:

The script is designed for a Kaggle environment (e.g., /kaggle/input/ for datasets).
Ensure your dataset (e.g., cleaned_medical.jsonl) is placed in the appropriate input directory.
Set environment variables for CUDA optimization:export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True





Usage

Prepare the Dataset:

Place your cleaned_medical.jsonl file in the input directory (e.g., /kaggle/input/medical/).
The dataset should contain a text column with medical prompts or data.


Run the Script:Execute the script to fine-tune the model and test the chatbot:
python medical_chatbot.py


The script loads the dataset, tokenizes it, fine-tunes the model, and saves outputs to /kaggle/working/results.
After training, it enters an interactive mode where you can input medical questions (type exit to quit).


Output:

Model and Tokenizer: Saved to /kaggle/working/medical-llama2.
Metadata: Saved as /kaggle/working/medical_llama2_meta.pkl.
Checkpoints: Training checkpoints are saved to /kaggle/working/results.


Example Interaction:
Please enter your prompt (type 'exit' to quit):
Enter your Question: What are the symptoms of diabetes?
Question: What are the symptoms of diabetes?
Answer: Symptoms of diabetes may include increased thirst, frequent urination, extreme fatigue, blurred vision, slow-healing wounds, and unexplained weight loss. Type 1 diabetes may develop quickly, while Type 2 symptoms may appear gradually. Consult a healthcare provider for a proper diagnosis.
--------------------------------------------------
Enter your Question: exit
Bye!



Code Structure

medical_chatbot.py: Main script for dataset loading, model fine-tuning, and interactive response generation.
Input Directory: /kaggle/input/medical/ for the dataset (e.g., cleaned_medical.jsonl).
Output Directory: /kaggle/working/ for model checkpoints, saved model, tokenizer, and metadata.
Key Functions:
format_prompt_with_system_instruction(): Adds a medical-specific system prompt.
tokenize_function(): Tokenizes the dataset with padding and truncation.
generate_response(): Generates responses for user prompts using the fine-tuned model.
CustomTrainer: Custom trainer class to handle loss computation for training.



Requirements
The script uses the following Python packages:

transformers, datasets, peft, bitsandbytes: For model loading, fine-tuning, and quantization.
torch, accelerate: For GPU acceleration and training optimization.
numpy, pandas: For data processing.
tqdm, pickle: For progress bars and metadata saving.

Notes

Environment: The script is optimized for Kaggle’s environment but can be adapted for other platforms with minor changes to file paths.
Quantization: Uses 4-bit quantization to reduce memory usage, making it suitable for GPUs with limited VRAM.
Tokenization Warning: The script may show a huggingface/tokenizers parallelism warning. To suppress it, set:export TOKENIZERS_PARALLELISM=false


Training Parameters: Configured for 500 steps with a small batch size (16) and FP16 precision for efficiency.
Model Output: The fine-tuned model is saved with LoRA adapters, keeping the file size manageable.

