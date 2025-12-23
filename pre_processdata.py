import os
import yaml
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Configuration
ROOT_DIR = "open_api_specs" 
TARGET_FOLDERS = ["broken", "business", "deployed", "public", "specs-3.0"]

def clean_text(text):
    """Clean text for training."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', str(text))
    text = text.replace('\n', ' ').replace('\r', ' ')
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
    """Extracts data with full error handling."""
    dataset = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Skipping malformed file {filename}: {e}")
        return []

    if not is_dict(data):
        return []

    # --- PART 1: PROCESS PATHS ---
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
                
                context_str = f"Method: {method_key.upper()} | Path: {path} | Summary: {summary} | Tags: {', '.join(tags) if isinstance(tags, list) else ''}"
                
                if description and len(clean_text(description).split()) > 5:
                    dataset.append({
                        "source_file": filename,
                        "type": "operation_description",
                        "input_text": context_str,
                        "target_text": clean_text(description)
                    })

    # --- PART 2: PROCESS COMPONENTS ---
    components = safe_get(data, 'components')
    if components:
        # Process examples
        examples = safe_get(components, 'examples')
        if examples:
            for name, example_content in examples.items():
                if not is_dict(example_content):
                    continue
                    
                summary = safe_get(example_content, 'summary', '')
                description = safe_get(example_content, 'description', '')
                
                if description and summary:
                    val_str = str(safe_get(example_content, 'value', ''))[:200]
                    context = f"Example: {name} | Summary: {summary} | Data: {val_str}"
                    dataset.append({
                        "source_file": filename,
                        "type": "example_description",
                        "input_text": context,
                        "target_text": clean_text(description)
                    })
                elif summary and not description:
                    val_str = str(safe_get(example_content, 'value', ''))[:200]
                    context = f"Example: {name} | Data: {val_str}"
                    dataset.append({
                        "source_file": filename,
                        "type": "example_summary",
                        "input_text": context,
                        "target_text": clean_text(summary)
                    })

        # Process schemas
        schemas = safe_get(components, 'schemas')
        if schemas:
            for name, schema_content in schemas.items():
                if not is_dict(schema_content):
                    continue
                    
                description = safe_get(schema_content, 'description')
                if description and len(clean_text(description).split()) > 3:
                    props = list(safe_get(schema_content, 'properties', {}).keys())
                    context = f"Schema: {name} | Fields: {', '.join(props)}"
                    dataset.append({
                        "source_file": filename,
                        "type": "schema_description",
                        "input_text": context,
                        "target_text": clean_text(description)
                    })

    return dataset

def main():
    print("🚀 Starting ROBUST Data Extraction (Error-Proof)...")
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
    
    # Create DataFrame and clean
    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset=['input_text'], inplace=True)
    
    print(f"✅ Extracted {len(df)} unique examples!")
    print("\n📈 Breakdown by type:")
    print(df['type'].value_counts())
    
    # Save splits (80/10/10)
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    train.to_json("train.json", orient="records", lines=True, force_ascii=False)
    val.to_json("val.json", orient="records", lines=True, force_ascii=False)
    test.to_json("test.json", orient="records", lines=True, force_ascii=False)
    
    # Save summary stats
    summary = {
        "total_examples": len(df),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "type_breakdown": df['type'].value_counts().to_dict()
    }
    with open("dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 SUCCESS! Dataset ready:")
    print(f"   📄 train.json: {len(train)} examples")
    print(f"   📄 val.json: {len(val)} examples") 
    print(f"   📄 test.json: {len(test)} examples")
    print(f"   📊 dataset_summary.json: Stats file")

if __name__ == "__main__":
    main()
