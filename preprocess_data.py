import os
import yaml
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Configuration
ROOT_DIR = "open_api_specs"  # Make sure this matches your actual folder name
TARGET_FOLDERS = ["broken", "business", "deployed", "public", "specs-3.0"]

def clean_text(text):
    """Clean text for training."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Replace newlines with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_dict(obj):
    """Safe check if object is dictionary."""
    return isinstance(obj, dict)

def safe_get(obj, key, default=None):
    """Safe dictionary get that handles non-dict objects."""
    if not is_dict(obj):
        return default
    return obj.get(key, default)

def parse_yaml_file(filepath, filename):
    """Extracts ONLY operation descriptions with full error handling."""
    dataset = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Skipping malformed file {filename}: {e}")
        return []

    if not is_dict(data):
        return []

    # --- PROCESS PATHS ONLY ---
    paths = safe_get(data, 'paths')
    if paths:
        for path, path_content in paths.items():
            if not is_dict(path_content):
                continue
                
            for method_key, method_content in path_content.items():
                # Skip non-method keys
                if method_key in ['summary', 'description', 'parameters', 'servers', '$ref']:
                    continue
                    
                # Only process if method_content is a dict
                if not is_dict(method_content):
                    continue
                
                # Extract operation data SAFELY
                description = safe_get(method_content, 'description')
                summary = safe_get(method_content, 'summary', '')
                tags = safe_get(method_content, 'tags', [])
                
                # Construct Input Context
                # We format tags nicely if they exist
                tags_str = ', '.join(tags) if isinstance(tags, list) else ''
                context_str = f"Method: {method_key.upper()} | Path: {path} | Summary: {summary} | Tags: {tags_str}"
                
                # Filter: Keep only substantial descriptions (> 5 words)
                # This removes "Returns 200 OK" type descriptions
                cleaned_desc = clean_text(description)
                if cleaned_desc and len(cleaned_desc.split()) > 5:
                    dataset.append({
                        "source_file": filename,
                        "type": "operation_description",
                        "input_text": context_str,
                        "target_text": cleaned_desc
                    })

    return dataset

def main():
    print("🚀 Starting Focused Data Extraction (Operations Only)...")
    all_data = []
    skipped_files = 0
    
    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(ROOT_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️  Folder not found: {folder_path}")
            continue
            
        print(f"📂 Processing folder: {folder}...")
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.yaml', '.yml', '.json')):
                    full_path = os.path.join(root, file)
                    try:
                        extracted = parse_yaml_file(full_path, file)
                        all_data.extend(extracted)
                    except Exception as e:
                        skipped_files += 1
                        print(f"⚠️  Error processing {file}: {str(e)[:50]}...")

    print(f"\n📊 RESULTS:")
    print(f"✅ Successfully processed files: {len(all_data) > 0}")
    print(f"⚠️  Skipped malformed files: {skipped_files}")
    
    if not all_data:
        print("❌ No valid data extracted! Check your ROOT_DIR path.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Remove duplicates
    df.drop_duplicates(subset=['input_text'], inplace=True)
    
    print(f"✅ Extracted {len(df)} unique operation descriptions!")
    
    # Save splits (80/10/10)
    print("✂️ Splitting data...")
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # Save to JSON
    train.to_json("train.json", orient="records", lines=True, force_ascii=False)
    val.to_json("val.json", orient="records", lines=True, force_ascii=False)
    test.to_json("test.json", orient="records", lines=True, force_ascii=False)
    
    print(f"\n🎉 SUCCESS! Dataset ready for Training:")
    print(f"   📄 train.json: {len(train)} examples")
    print(f"   📄 val.json: {len(val)} examples") 
    print(f"   📄 test.json: {len(test)} examples")

if __name__ == "__main__":
    main()
