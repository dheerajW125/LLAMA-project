from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")


def generate_text(prompt, max_length=50):
    model_name = "meta-llama/Llama-3.2"  

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Use GPU if available
    )

    # Prepare input and generate text
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, max_length=max_length, temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    prompt = "Once upon a time in a distant galaxy, there was a starship captain named"
    print(generate_text(prompt))
