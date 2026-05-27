import json
import os
from datetime import datetime

def update_leaderboard(leaderboard_path, new_results_path):
    # 1. Load existing leaderboard or create template
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, 'r') as f:
            leaderboard = json.load(f)
    
    if os.path.exists(new_results_path):
        with open(new_results_path, 'r') as f:
            new_results_data = json.load(f)

    else:
        # Minimal template if file doesn't exist
        leaderboard = {
            "schema_version": "1.0",
            "generated_at": "",
            "benchmark": {},
            "task_definitions": {},
            "results": []
        }

    task_defs = leaderboard.get("task_definitions", {})
    
    # Mapping for tasks that use a nested 'Overall' or 'overall' key in raw data
    nested_key_map = {
        "vqa": "Overall",
        "pointing": "Overall",
        "temporal": "overall",
        "dvc": "overall"
    }

    # 2. Process each model in the new data
    for model_name, tasks in new_results_data.items():
        task_results = {}

        # Iterate through tasks defined in the leaderboard schema
        for task_id, info in task_defs.items():
            required_metrics = info.get("metrics", [])
            raw_task_data = tasks.get(task_id, {})

            # Drill down into nested keys if necessary (e.g., vqa -> Overall)
            if task_id in nested_key_map:
                sub_key = nested_key_map[task_id]
                raw_task_data = raw_task_data.get(sub_key, {})

            # Extract only the metrics listed in task_definitions
            extracted_metrics = {}
            for metric in required_metrics:
                print(raw_task_data)
                val = raw_task_data.get(metric, 0)
                
                if isinstance(val, (int, float)):
                    # --- NORMALIZATION LOGIC ---
                    # If the value is small (0-1 range) and not a count, scale to 100
                    # We check 'total' or 'count' specifically to avoid scaling them
                    if val <= 1.0 and val > 0 and "count" not in metric and "total" not in metric:
                        val = val * 100
                    
                    extracted_metrics[metric] = round(float(val), 2)
                else:
                    extracted_metrics[metric] = val if val is not None else ""

            task_results[task_id] = extracted_metrics

        # 3. Calculate simple Overall Score (Average of primary metrics)
        primary_scores = []
        for t_id, t_res in task_results.items():
            primary_key = task_defs[t_id]["primary_metric"]
            score = t_res.get(primary_key, 0)
            if isinstance(score, (int, float)):
                primary_scores.append(score)
        
        print(primary_scores)
        overall_score = round(sum(primary_scores) / len(primary_scores), 1) if primary_scores else 0

        # 4. Create the model entry
        new_entry = {
            "model_name": model_name,
            "is_baseline": False,
            "params": tasks.get("config", ""), # Pull from config if available
            "submission_type": "open-weights",
            "overall_score": overall_score,
            "task_results": task_results,
            "domain_scores": {
                "public_safety": 0,
                "transportation": 0,
                "warehouse": 0
            },
            "metadata": {
                "evaluated_at": datetime.now().strftime("%Y-%m-%d"),
            }
        }

        # 5. Overwrite or Append
        existing_index = next((i for i, item in enumerate(leaderboard["results"]) 
                              if item["model_name"] == model_name), None)
        
        if existing_index is not None:
            # Preserve existing metadata like 'is_baseline' or 'params' if they weren't in new data
            leaderboard["results"][existing_index].update(new_entry)
            print(f"Updated: {model_name}")
        else:
            leaderboard["results"].append(new_entry)
            print(f"Added: {model_name}")

    # Update generation timestamp
    leaderboard["generated_at"] = datetime.now().isoformat()

    # 6. Save back to JSON
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)

# --- Usage ---
if __name__ == "__main__":
    update_leaderboard('../hf/leaderboard.json', "vlmevalkit_outputs.json")