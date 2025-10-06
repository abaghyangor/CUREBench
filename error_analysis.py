import pandas as pd
import json

def analyze_submission_with_patterns():
    # Load data
    val_data = pd.read_json('data/curebench_valset_phase1.jsonl', lines=True)
    answers_dict = dict(zip(val_data['id'], val_data['correct_answer']))
    question_types = dict(zip(val_data['id'], val_data['question_type']))
    question = dict(zip(val_data['id'], val_data['question']))
    
    df = pd.read_csv('competition_results/submission.csv')
    
    correct_answers = 0
    total_mc_questions = 0
    errors = []
    
    for _, row in df.iterrows():
        question_id = row['id']
        model_choice = row['choice']
        
        # Skip open-ended questions
        if model_choice == "NOTAVALUE":
            continue
            
        correct_answer = answers_dict[question_id]
        total_mc_questions += 1
        
        if model_choice == correct_answer:
            correct_answers += 1
        else:
            errors.append({
                'id': question_id,
                'question_type': question_types[question_id],
                'question': question[question_id],
                'predicted': model_choice,
                'correct': correct_answer,
                'full_response': row['prediction']
            })
    
    # Pattern Analysis
    print(f"=== PERFORMANCE SUMMARY ===")
    print(f"MC Questions: {total_mc_questions}")
    print(f"Correct: {correct_answers}")
    print(f"Accuracy: {correct_answers/total_mc_questions:.2%}")
    
    # Error patterns
    print(f"\n=== ERROR PATTERNS ===")
    
    # By predicted answer
    pred_errors = {}
    for error in errors:
        pred = error['predicted']
        pred_errors[pred] = pred_errors.get(pred, 0) + 1
    
    print("Model's wrong predictions:")
    for pred, count in sorted(pred_errors.items()):
        print(f"  {pred}: {count} times ({count/len(errors):.1%})")
    
    # By correct answer
    correct_errors = {}
    for error in errors:
        correct = error['correct']
        correct_errors[correct] = correct_errors.get(correct, 0) + 1
    
    print("Correct answers model missed:")
    for correct, count in sorted(correct_errors.items()):
        print(f"  {correct}: {count} times ({count/len(errors):.1%})")
    
    # Sample errors for manual review
    print(f"\n=== SAMPLE ERRORS (First 5) ===")
    for i, error in enumerate(errors):
        print(f"\nError {i+1} ({error['question_type']}):")
        print(f"  Predicted: {error['predicted']}, Correct: {error['correct']}")
        print(f"  Response: {error['full_response']}")
    
    # Save for detailed analysis
    json.dump(errors, open('detailed_errors.json', 'w'), indent=2)
    pd.DataFrame(errors).to_csv('detailed_errors.csv', index=False)
    print(f"\nSaved {len(errors)} errors to detailed_errors.json")
    
    return errors

if __name__ == "__main__":
    errors = analyze_submission_with_patterns()