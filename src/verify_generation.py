from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sys
import os


def test_device(device_name, model_name="gpt2-medium"):
    print(f"\nTesting {model_name} on {device_name}...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device_name)

        prompts = [
            "What is the capital of France?",
            "The capital of France is",
            "2 + 2 =",
        ]

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                device_name
            )
            # greedy decoding
            outputs = model.generate(
                input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id
            )
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  Q: {prompt}")
            print(f"  A: {res}")

    except Exception as e:
        print(f"  Error on {device_name}: {e}")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        test_device("mps")
    else:
        print("MPS not available.")
