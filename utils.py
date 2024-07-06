import json

# Load your JSON data
json_file = 'que-faire-a-paris.json'
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to prompt/completion pairs
formatted_data = []
for entry in data:
    prompt = f"Suggest an event in Paris with details:\nTitle: {entry['title']}\nDescription: {entry['description']}\nDate: {entry['date_description']}\n"
    completion = f"Event: {entry['title']}\nLocation: {entry['address_name']}, {entry['address_street']}, {entry['address_city']}\nDate: {entry['date_description']}\nDetails: {entry['description']}"
    formatted_data.append({"prompt": prompt, "completion": completion})

# Save as JSONL
jsonl_file = 'formatted_que-faire-a-paris.jsonl'
with open(jsonl_file, 'w', encoding='utf-8') as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + '\n')

print(f'Converted JSON data to {jsonl_file}.')
