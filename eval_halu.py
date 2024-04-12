import os
import json
import time
import torch
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift

class ModelEvaluation:
    def __init__(self, model_dirs, model_type, test_path, eval_limit=1000, max_new_tokens=256):
        self.model_dirs = model_dirs
        self.model_type = model_type
        self.test_path = test_path
        self.eval_limit = eval_limit
        self.max_new_tokens = max_new_tokens
        self.template_type = get_default_template_type(self.model_type)
        self.model, self.tokenizer = get_model_tokenizer(self.model_type, torch.bfloat16, {'device_map': 'auto'})
        self.json_errors = 0

    def convert_to_json_correctly(self, s):
        replacements = {
            '\"': "\'",
            "\'thought\': \'": '\"thought\": \"',
            "\'thought\': \"": '\"thought\": \"',
            "yes": "no"
        }
        for old, new in replacements.items():
            s = s.replace(old, new)
        return s

    def load_and_evaluate_model(self):
        for model_dir in self.model_dirs:
            model = Swift.from_pretrained(self.model, model_dir, inference_mode=True)
            template = get_template(self.template_type, self.tokenizer)
            model.generation_config.max_new_tokens = self.max_new_tokens

            results = self.evaluate_predictions(model, template)
            json_correctness_rate = (self.eval_limit - self.json_errors) / self.eval_limit if self.eval_limit else 0
            self.log_results(model_dir, results, json_correctness_rate)

    def evaluate_predictions(self, model, template):
        results = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'total': 0}
        with open(self.test_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                response, _ = inference(model, template, data['prompt'])
                response = self.convert_to_json_correctly(response)
                try:
                    response_json = json.loads(response)
                    results = self.update_stats(response_json, data, results)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    self.json_errors += 1
                results['total'] += 1
                if results['total'] >= self.eval_limit:
                    break
        return results

    def update_stats(self, response_json, data, stats):
        hallucination_response = response_json.get("hallucination")
        if hallucination_response:
            if hallucination_response == data['response']:
                if hallucination_response == 'yes':
                    stats['TP'] += 1
                else:
                    stats['TN'] += 1
            else:
                if hallucination_response == 'yes':
                    stats['FP'] += 1
                else:
                    stats['FN'] += 1
        return stats

    def log_results(self, model_dir, results, json_correctness_rate):
        accuracy = (results['TP'] + results['TN']) / sum(results.values()) if sum(results.values()) else 0
        precision = results['TP'] / (results['TP'] + results['FP']) if (results['TP'] + results['FP']) else 0
        recall = results['TP'] / (results['TP'] + results['FN']) if (results['TP'] + results['FN']) else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        with open(f'log{time.time()}.txt', 'a', encoding='utf-8') as log_file:
            log_file.write(f"Model Directory: {model_dir}\n")
            log_file.write(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}, JSON Correctness: {json_correctness_rate:.2f}\n")

# Set the CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Instantiate and use the model evaluation class
model_dirs = ["/path/to/your/model/dir"]
model_evaluator = ModelEvaluation(model_dirs, ModelType.llama2_7b_chat, 'path/to/your/test_data.jsonl')
model_evaluator.load_and_evaluate_model()
