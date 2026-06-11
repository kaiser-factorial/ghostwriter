import json
import random
from pathlib import Path

def main():
    personas = ["vangogh", "pepys", "mansfield", "maclane"]
    persona_mapping = {
        "vangogh": "van_gogh",
        "pepys": "pepys",
        "mansfield": "mansfield",
        "maclane": "maclane"
    }
    
    all_entries = []
    
    for p in personas:
        file_path = Path(f"data/clean/{p}.jsonl")
        if not file_path.exists():
            print(f"Skipping {p}, file not found.")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Get up to 15 valid entries for each
        valid_entries = []
        for line in lines:
            data = json.loads(line)
            # Only take entries that are reasonably long
            if len(data["text"]) > 150:
                # Update persona string to match frontend expectations
                data["persona"] = persona_mapping[p]
                valid_entries.append(data)
                
            if len(valid_entries) >= 15:
                break
                
        all_entries.extend(valid_entries)
        print(f"Added {len(valid_entries)} entries for {p}")

    # Shuffle them to create a blended timeline
    random.seed(42)
    random.shuffle(all_entries)

    output_path = Path("frontend/src/entries.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2)
        
    print(f"Successfully wrote {len(all_entries)} entries to {output_path}")

if __name__ == "__main__":
    main()
