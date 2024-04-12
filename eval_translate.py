import os
import json
import torch
import tensorflow as tf
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.utils import seed_everything
from bleurt import score
import numpy as np
import gc

# Constants for terminal color
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

class ModelEvaluator:
    def __init__(self, model_dirs, model_type, bleurt_model_path, seed=2024):
        self.model_dirs = model_dirs
        self.model_type = model_type
        self.template_type = get_default_template_type(model_type)
        self.bleurt_model_path = bleurt_model_path
        self.setup_cuda()
        seed_everything(seed)
        self.model, self.tokenizer = self.load_model()

    def setup_cuda(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        
    def free_gpu_memory(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def load_model(self):
        kwargs = {'use_flash_attn': True}
        model_t, tokenizer = get_model_tokenizer(self.model_type, torch.bfloat16, {'device_map': 'auto'}, **kwargs)
        return model_t, tokenizer

    def process_data(self):
        history = [
            ("your few shot example query", "your few shot example response"),
            ("your few shot example query", "your few shot example response"),
        ]
        len_history = len(self.tokenizer.tokenize(str(history)))
        
        for model_dir in self.model_dirs:
            responses, queries, true_responses = [], [], []
            with open('/path/to/your/test.jsonl', 'r') as f:
                for count, line in enumerate(f):
                    data = json.loads(line)
                    query = data["query"]
                    if self.is_query_too_long(data['response'], query, len_history):
                        continue
                    response, _ = inference(self.model, self.get_template(), query, history)
                    responses.append(response)
                    queries.append(query)
                    true_responses.append(data["response"])
                    self.print_response(response, data["response"], count)

            self.output_responses(model_dir, queries, responses, true_responses)
            self.free_gpu_memory()
            self.evaluate_with_bleurt(queries, responses, true_responses, model_dir)

    def is_query_too_long(self, response, query, len_history, max_len=4096):
        return len(self.tokenizer.tokenize(response)) + len(self.tokenizer.tokenize(query)) + len_history > max_len

    def get_template(self):
        return get_template(self.template_type, self.tokenizer)

    def print_response(self, response, true_response, count):
        print(f"{GREEN}Response: \n{response}{RESET}")
        print(f"{RED}True Response: \n{true_response}{RESET}")
        print("="*20 + str(count))

    def output_responses(self, model_dir, queries, responses, true_responses):
        with open(f'output_responses_{model_dir.split("/")[-2]}.jsonl', 'w') as out_f:
            for query, response, true_response in zip(queries, responses, true_responses):
                output_data = {
                    "query": query,
                    "response": response,
                    "true_response": true_response,
                }
                json.dump(output_data, out_f)
                out_f.write('\n')

    def evaluate_with_bleurt(self, queries, responses, true_responses, model_dir):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        tf.config.set_visible_devices([], 'GPU')
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[1], 'GPU')
        bleurt_scorer = score.BleurtScorer(self.bleurt_model_path)
        
        bleurt_scores = []
        with open(f'output_responses_{model_dir.split("/")[-2]}.jsonl', 'w') as out_f:
            for query, response, true_response in zip(queries, responses, true_responses):
                bleurt_score = bleurt_scorer.score(references=[true_response], candidates=[response])[0]
                output_data = {
                    "query": query,
                    "response": response,
                    "true_response": true_response,
                    "bleurt": bleurt_score
                }
                json.dump(output_data, out_f)
                out_f.write('\n')
                bleurt_scores.extend(bleurt_score)
        print(np.mean(bleurt_scores))
        tf.keras.backend.clear_session()
        gc.collect()

# Usage of ModelEvaluator
model_dirs = ["/path/to/your/model"]
evaluator = ModelEvaluator(model_dirs, ModelType.gemma_7b_instruct, "/root/bleurt/BLEURT-20")
evaluator.process_data()
