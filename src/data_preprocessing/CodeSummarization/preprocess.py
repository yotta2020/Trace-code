import json
import os
import gzip
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CodeSummarization Preprocessing')
    parser.add_argument('--lang', type=str, required=True, help='Programming language (e.g., python, java, php)')
    parser.add_argument('--input_base_dir', type=str, default='data/raw/CodeSummarization', help='Base directory for raw data')
    parser.add_argument('--output_base_dir', type=str, default='data/processed/CodeSummarization', help='Base directory for preprocessed data')
    return parser.parse_args()


def load_jsonl_files(jsonl_dir, split):
    """
    Load all .jsonl or .jsonl.gz files for a given split and build url->data dict.
    
    Args:
        jsonl_dir: Directory containing the jsonl files
        split: train/valid/test
    
    Returns:
        dict: {url: json_object}
    """
    data_dict = {}
    split_dir = os.path.join(jsonl_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Warning: Directory {split_dir} does not exist. Skipping.")
        return data_dict
    
    files = []
    for root, dirs, filenames in os.walk(split_dir):
        for filename in filenames:
            if '.jsonl' in filename:
                files.append(os.path.join(root, filename))
    
    print(f"Found {len(files)} jsonl files for {split}")
    
    for filepath in files:
        if filepath.endswith('.gz'):
            os.system(f"gzip -d {filepath}")
            filepath = filepath.replace('.gz', '')
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    js = json.loads(line)
                    data_dict[js['url']] = js
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(data_dict)} unique samples for {split}")
    return data_dict


def apply_language_specific_processing(data_dict, lang):
    """
    Apply language-specific post-processing.
    
    Args:
        data_dict: {url: json_object}
        lang: programming language
    """
    if lang == 'java':
        for url, obj in data_dict.items():
            if '(non-Javadoc)' in obj.get('docstring', ''):
                obj['docstring'] = obj['docstring'].replace('(non-Javadoc)', '').strip()
                obj['docstring_tokens'] = obj['docstring'].split()
    
    elif lang == 'php':
        for url, obj in data_dict.items():
            if not obj.get('code', '').startswith('<?php'):
                obj['code'] = '<?php\n' + obj['code'] + '\n?>'


def main():
    args = parse_args()
    
    jsonl_dir = os.path.join(args.input_base_dir, args.lang, 'final', 'jsonl')
    output_dir = os.path.join(args.output_base_dir, args.lang)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing language: {args.lang}")
    print(f"Input directory: {jsonl_dir}")
    print(f"Output directory: {output_dir}")
    print("")
    
    for split in ['train', 'valid', 'test']:
        print(f"Processing {split}...")
        
        data_dict = load_jsonl_files(jsonl_dir, split)
        
        if not data_dict:
            print(f"No data found for {split}. Skipping.")
            continue
        
        apply_language_specific_processing(data_dict, args.lang)
        
        txt_file = os.path.join(args.input_base_dir, args.lang, f'{split}.txt')
        output_file = os.path.join(output_dir, f'{split}.jsonl')
        
        if not os.path.exists(txt_file):
            print(f"Warning: {txt_file} not found. Skipping {split}.")
            continue
        
        written_count = 0
        missing_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f_out, open(txt_file, 'r', encoding='utf-8') as f_txt:
            for line in f_txt:
                url = line.strip()
                if not url:
                    continue
                
                if url in data_dict:
                    f_out.write(json.dumps(data_dict[url]) + '\n')
                    written_count += 1
                else:
                    missing_count += 1
        
        print(f"Written {written_count} samples to {output_file}")
        if missing_count > 0:
            print(f"Warning: {missing_count} urls from {split}.txt not found in jsonl files")
        print("")


if __name__ == '__main__':
    main()