import json, os
from datetime import datetime

entries = []
files = ['maclane.jsonl', 'mansfield.jsonl', 'pepys.jsonl', 'van_gogh.jsonl']
for f in files:
    persona = f.replace('.jsonl', '')
    path = f'data/clean/{f}'
    if not os.path.exists(path): continue
    with open(path) as infile:
        for line in infile:
            try:
                data = json.loads(line)
                data['persona'] = persona
                entries.append(data)
            except:
                pass

def parse_date(date_str):
    if not date_str: return datetime.min
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except:
        return datetime.min

entries.sort(key=lambda x: parse_date(x.get('date', '')))
with open('frontend/src/entries.json', 'w') as out:
    json.dump(entries, out)
