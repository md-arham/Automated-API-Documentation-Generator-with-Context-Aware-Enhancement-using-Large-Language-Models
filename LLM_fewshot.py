# batch_infer_t5_fewshot.py

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ------------------------------
# 1️⃣ Paths & Config
# ------------------------------
MODEL_DIR = Path("./t5_finetuned_api")
TEST_FILE = Path("./api_dataset/test.json")
OUTPUT_FILE = Path("./predictions_few_shot.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 1289

print(f"Using device: {DEVICE}")

# ------------------------------
# 2️⃣ Load Model & Tokenizer
# ------------------------------
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# ------------------------------
# 3️⃣ Few-shot Examples (5)
# ------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "input": "Method: GET | Path: /users | Summary: Get users | Parameter: page (integer)",
        "output": "Specifies the page number of results to retrieve."
    },
    {
        "input": "Method: GET | Path: /users | Summary: Get users | Parameter: pageSize (integer)",
        "output": "Defines the number of user records returned per page."
    },
    {
        "input": "Method: POST | Path: /login | Summary: User login | Parameter: username (string)",
        "output": "The username used to authenticate the user."
    },
    {
        "input": "Method: POST | Path: /login | Summary: User login | Parameter: password (string)",
        "output": "The password associated with the user's account."
    },
    {
        "input": "Method: GET | Path: /articles | Summary: Search articles | Parameter: query (string)",
        "output": "The search keyword used to filter articles."
    },
]

def build_few_shot_prompt(input_text: str) -> str:
    prompt = (
        "You are an expert API documentation generator.\n"
        "Given API metadata, write a concise and accurate description.\n\n"
    )

    for ex in FEW_SHOT_EXAMPLES:
        prompt += f"Input:\n{ex['input']}\nOutput:\n{ex['output']}\n\n"

    prompt += f"Input:\n{input_text}\nOutput:\n"
    return prompt

# ------------------------------
# 4️⃣ Generation Function
# ------------------------------
def generate_description(prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_LENGTH,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------------
# 5️⃣ Run Few-shot Inference
# ------------------------------
predictions = []

with open(TEST_FILE, encoding="utf-8", errors="replace") as f:
    for line in tqdm(f, desc="Running few-shot inference"):
        item = json.loads(line)

        prompt = build_few_shot_prompt(item["input_text"])
        prediction = generate_description(prompt)

        predictions.append({
            "input_text": item["input_text"],
            "reference": item["target_text"],
            "prediction": prediction
        })

# ------------------------------
# 6️⃣ Save Outputs
# ------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2)

print(f"✅ Few-shot predictions saved to {OUTPUT_FILE}")