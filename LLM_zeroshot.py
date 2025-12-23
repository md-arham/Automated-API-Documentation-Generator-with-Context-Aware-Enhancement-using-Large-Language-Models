import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

MODEL_PATH = "./t5_finetuned_api"
TEST_FILE = "./api_dataset/test.json"
OUTPUT_FILE = "./predictions_zero_shot.json"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()

def generate(prompt, max_length=80):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

predictions = []

with open(TEST_FILE, encoding="utf-8", errors="replace") as f:
    for line in tqdm(f):
        item = json.loads(line)

        # ZERO-SHOT PROMPT
        prompt = (
            "document:\n"
            "You are an API documentation expert.\n"
            "Generate a concise, human-readable description.\n\n"
            f"{item['input_text']}"
        )

        output = generate(prompt)

        predictions.append({
            "input": item["input_text"],
            "reference": item["target_text"],
            "prediction": output
        })

with open(OUTPUT_FILE, "w") as f:
    json.dump(predictions, f, indent=2)

print("✅ Inference complete. Saved to", OUTPUT_FILE)