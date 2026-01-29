import json
import os
import argparse
from threading import Lock 
from datetime import datetime
from transformers import AutoTokenizer
from itertools import count

request_id_counter = count()

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

class RequestGenerator:
    def __init__(self, model_path, origin_dataset_file, processed_dataset_file):
        self._model_path = model_path
        self._origin_dataset_file = origin_dataset_file
        self._processed_dataset_file = processed_dataset_file
        self.block_size = 512
        self.hash_session_id_map = {}
        self.session_id_map_lock = Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.stable_tokens = self._build_stable_token_mapping()

    def _build_stable_token_mapping(self) -> dict:
        stable_mapping = {}
        for token_id in range(1000, min(50000, self.tokenizer.vocab_size)):
            if token_id in self.tokenizer.all_special_ids:
                continue
                
            text = self.tokenizer.decode([token_id])
            reencoded = self.tokenizer.encode(text, add_special_tokens=False)
            if len(reencoded) == 1 and reencoded[0] == token_id:
                stable_mapping[len(stable_mapping)] = token_id 
                if len(stable_mapping) >= 5000:
                    break
                    
        if not stable_mapping:
            raise ValueError("failed to build stable token mapping")
        
        return stable_mapping

    def _get_tokens_for_hash(self, hash_id: int) -> int:
        sample_token_ids = []
        for i in range(32):
            sample_token_ids.append((hash_id + i) % len(self.stable_tokens))
        return sample_token_ids

    def _get_token_for_hash(self, hash_id: int) -> int:
        return self.stable_tokens[hash_id % len(self.stable_tokens)]


    def generate_request(self, record: dict, more_tokens: bool) -> dict:
        token_ids = []
        remaining = record["input_length"]
        hash_ids = record["hash_ids"]
        request_id = next(request_id_counter)

        if more_tokens:
            for hid in record["hash_ids"][:-1]:
                block_size = min(self.block_size, remaining)
                sample_token_ids = self._get_tokens_for_hash(hid)
                for token_id in sample_token_ids:
                    token_ids.extend([token_id] * int(block_size/len(sample_token_ids)))
                
            if remaining > 0:
                token_id = self._get_tokens_for_hash(record["hash_ids"][-1])
                same_tokens_len = int(block_size/len(sample_token_ids))
                for token_id in sample_token_ids:
                    if remaining > same_tokens_len:
                        token_ids.extend([token_id] * same_tokens_len) 
                        remaining -= same_tokens_len
                    else:
                        token_ids.extend([token_id] * max(remaining,1))
                        break         
        else:
            for hid in record["hash_ids"][:-1]:
                block_size = min(self.block_size, remaining)
                token_id = self._get_token_for_hash(hid)
                token_ids.extend([token_id] * block_size)
                remaining -= block_size
            
            if remaining > 0:
                token_id = self._get_token_for_hash(record["hash_ids"][-1])
                token_ids.extend([token_id] * remaining)            

        prompts = self.tokenizer.decode(token_ids)
        input_ids = self.tokenizer.encode(prompts, add_special_tokens=False)
        
        if len(input_ids) != len(token_ids):
            input_ids = token_ids
        
        session_id = ""
        hash_session_id = ""
        g_session_id = ""
        if len(record["hash_ids"]) >= 2:
            session_id = str(record["hash_ids"][0]) + str(record["hash_ids"][1])
        else:
            session_id = str(record["hash_ids"][0])

        with self.session_id_map_lock:
            now_time = datetime.now().timestamp()
            if session_id in self.hash_session_id_map:
                hash_session_id = self.hash_session_id_map[session_id]
            else:
                hash_session_id = str(now_time) + "@" + str(session_id)
                self.hash_session_id_map[session_id] = hash_session_id
            g_session_id = hash_session_id + "@" + str(request_id)  

        output_len = record["output_length"]

        log_entry = {
            "timestamp": record["timestamp"],
            "request_id": request_id,
            "g_session_id": g_session_id,
            "hash_session_id": hash_session_id,
            "input_length": record["input_length"],  
            "num_input_ids": len(input_ids),  
            "output_len": output_len,
            "hash_ids":hash_ids,
            "prompts": prompts,
            "input_ids": input_ids 
        }
        print(f"{request_id},{hash_session_id}, {g_session_id},len(input_ids)={len(input_ids)},output_len={output_len}")
        output_dir = os.path.dirname(self._processed_dataset_file)
        os.makedirs(output_dir, exist_ok=True)
        with open(self._processed_dataset_file, "a") as f:  
            json.dump(log_entry, f)
            f.write("\n")  

    def valid_record(self, record):
        valid = True
        if record["timestamp"] < 0 or record["input_length"] <= 0 \
        or record["output_length"] < 0 or len(record["hash_ids"]) < 1:
            valid = False
        return valid

    def generate_from_file(self, more_tokens):
        with open(self._origin_dataset_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                if not self.valid_record(record):
                    continue
                self.generate_request(record, more_tokens)

def main():
    parser = argparse.ArgumentParser(description="Process dataset for DualMap experiments")
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--origin_dataset_file', type=str, required=True, help='Dataset type')
    parser.add_argument('--processed_dataset_file', type=str, required=True, help='Processed dataset file')
    args = parser.parse_args()

    model_path = args.model_path
    more_tokens = False
    origin_dataset_file = args.origin_dataset_file
    processed_dataset_file = args.processed_dataset_file
    request_generator1 = RequestGenerator(model_path, origin_dataset_file, processed_dataset_file)
    request_generator1.generate_from_file(more_tokens)

if __name__ == "__main__":
    main()