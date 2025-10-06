import json

def create_hard_val_set():
    hard_val = []
    with open('data/curebench_valset_phase1.jsonl', 'r') as f:
        for line in f:
            record = json.loads(line)
            # Criteria closer to test set complexity
            is_long = len(record['question']) > 180
            is_multi_sentence = len(record['question'].split('.')) > 2
            has_patient = 'patient' in record['question'].lower()
            has_scenario = any(word in record['question'].lower() for word in ['diagnosed', 'treated', 'receiving'])
            
            # Include if it matches test-like complexity
            if is_long or is_multi_sentence or (has_patient and has_scenario):
                hard_val.append(record)
    
    return hard_val

print(f"Number of hard validation samples: {len(create_hard_val_set())}")

def create_jsonl(dict_list, filepath):
    with open(filepath, 'w') as f:
        for record in dict_list:
            f.write(json.dumps(record) + '\n')
create_jsonl(create_hard_val_set(), 'data/hard_valset.jsonl')
