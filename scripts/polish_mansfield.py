import json
import os
import sys
import difflib
import time
from pathlib import Path
from google import genai
from google.genai import types

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set it before running this script: export GEMINI_API_KEY='your-key'")
        sys.exit(1)

    client = genai.Client()
    
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "clean" / "mansfield.jsonl"
    output_file = base_dir / "data" / "clean" / "mansfield_polished.jsonl"
    
    if not input_file.exists():
        print(f"Error: Could not find {input_file}")
        sys.exit(1)
        
    print(f"Reading {input_file}...")
    
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))
            
    print(f"Found {len(entries)} entries. Starting polish pass...")
    
    system_instruction = """You are an expert OCR corrector and editor. 
Your task is to fix OCR errors, typos, and stray spacing in a diary entry by Katherine Mansfield.
DO NOT rewrite her prose.
DO NOT alter her punctuation style, fragmented sentences, or voice.
DO NOT add commentary or explanation. 
Output ONLY the corrected text of the entry."""

    changes_made = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, entry in enumerate(entries):
            original_text = entry.get('text', '')
            if not original_text.strip():
                out_f.write(json.dumps(entry) + '\n')
                continue
                
            try:
                # Sleep briefly to avoid hammering the API
                time.sleep(0.1)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=original_text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1
                    )
                )
                polished_text = response.text.strip()
                
                # Compare similarity to ensure no wild rewrites
                similarity = difflib.SequenceMatcher(None, original_text, polished_text).ratio()
                
                if similarity < 0.7:
                    # Too different, LLM might have rewritten it entirely
                    print(f"[{i+1}/{len(entries)}] Warning: High edit distance (similarity {similarity:.2f}). Keeping original.")
                    final_text = original_text
                elif original_text != polished_text:
                    changes_made += 1
                    print(f"[{i+1}/{len(entries)}] Polished entry (similarity {similarity:.2f})")
                    final_text = polished_text
                else:
                    final_text = original_text
                    
            except Exception as e:
                print(f"[{i+1}/{len(entries)}] Error processing entry: {e}. Keeping original.")
                final_text = original_text
                
            entry['text'] = final_text
            out_f.write(json.dumps(entry) + '\n')
            out_f.flush()
            
    print(f"Done! Polished {changes_made} out of {len(entries)} entries.")
    print(f"Polished corpus saved to {output_file}")
    print("You can use a diff tool to compare the original and polished files.")

if __name__ == "__main__":
    main()
