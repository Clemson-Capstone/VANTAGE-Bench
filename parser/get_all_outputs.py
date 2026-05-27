import os
import json

def load_json(file_path):
    """Helper to safely load JSON files."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def extract_benchmarks(root_directory):
    compiled_data = {}

    # Map filename keywords to the desired JSON key name
    benchmark_map = {
        "Astro2D_metrics.json": "astro",
        "VANTAGE_2DGrounding_metrics.json": "grounding",
        "VANTAGE_2DPointing_results.json": "pointing",
        "VANTAGE_DVC_metrics.json": "dvc",
        "VANTAGE_EventVerification_acc.json": "event_verification",
        "VANTAGE_Temporal_metrics.json": "temporal",
        "VANTAGE_VQA_results.json": "vqa",
        "model_config.json": "config"
    }

    for root, dirs, files in os.walk(root_directory):
        model_name = os.path.basename(os.path.dirname(root))
        
        if not model_name or model_name == "outputs":
            continue

        if model_name not in compiled_data:
            compiled_data[model_name] = {}

        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Check if the filename matches any of our known benchmarks
            for keyword, key_name in benchmark_map.items():
                if keyword in filename:
                    data = load_json(file_path)
                    if data:
                        # This creates the nested structure: model -> benchmark_type -> {metrics}
                        compiled_data[model_name][key_name] = data
                    break 

    return compiled_data

# --- Execution ---
root_path = "../outputs/" 
results = extract_benchmarks(root_path)

# Clean up empty entries
results = {k: v for k, v in results.items() if v}

output_file = 'vlmevalkit_outputs.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Processed {len(results)} models. Stats saved to {output_file}.")