import argparse
import json
import os
import sys
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer


def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line := line.strip():
                yield json.loads(line)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)


def load_laws(path: str) -> List[Dict[str, str]]:
    laws: List[Dict[str, str]] = []
    if not os.path.exists(path):
        print(f"Error: law file '{path}' not found.")
        return laws

    for item in read_jsonl(path):
        content = item.get('law_article_content') or item.get('content') or ''
        if not content:
            continue
        ident = item.get('law_article_id') or len(laws) + 1
        laws.append({'id': str(ident), 'content': content})
    return laws


def parse_args():
    p = argparse.ArgumentParser(description='Generate fact-law match JSONL via embeddings.')
    p.add_argument('--input', type=str, required=True, help='Your facts file.')
    p.add_argument('--law-file', type=str, required=True, help='../data/knowledge/law_articles.jsonl')
    p.add_argument('--output', type=str, required=True, help='Your output path.')
    p.add_argument('--model-name', type=str, default='Qwen3-Embedding-8B', help='Embedding model name.')
    p.add_argument('--top-k', type=int, default=20, help='Number of top law articles per fact.')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size for fact encoding.')
    p.add_argument('--limit', type=int, default=0, help='Optional max number of facts to process.')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run model.')
    return p.parse_args()


def get_fact_id(rec: Dict[str, Any]) -> str:
    return str(rec.get('fact_id') or rec.get('id'))


def main():
    args = parse_args()
    ensure_parent(args.output)

    print(f"Loading laws from {args.law_file}...")
    laws = load_laws(args.law_file)
    if not laws:
        print('No laws loaded. Abort.')
        sys.exit(1)
    law_texts = [l['content'] for l in laws]
    print(f"Loaded {len(laws)} law articles.")

    print(f"Loading embedding model '{args.model_name}' on {args.device}...")
    model = SentenceTransformer(args.model_name, device=args.device, trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16})

    print('Encoding law articles...')
    law_embeddings = model.encode(law_texts, convert_to_tensor=True, show_progress_bar=True)

    print(f"Reading facts from {args.input}...")
    records: List[Dict[str, Any]] = []
    for rec in read_jsonl(args.input):
        if not isinstance(rec, dict):
            continue
        fact_txt = rec.get('fact_text') or rec.get('fact')
        if not fact_txt:
            continue
        records.append(rec)
        if args.limit and len(records) >= args.limit:
            break
    print(f"Collected {len(records)} facts to process.")

    out_count = 0
    with open(args.output, 'w', encoding='utf-8') as wf:
        batch: List[Dict[str, Any]] = []

        def process_batch(batch_records: List[Dict[str, Any]]):
            nonlocal out_count
            if not batch_records:
                return
            fact_texts = [r.get('fact_text') or r.get('fact') for r in batch_records]
            fact_embeddings = model.encode(fact_texts, prompt_name='query', convert_to_tensor=True, show_progress_bar=False)

            sim_matrix = model.similarity(fact_embeddings, law_embeddings)
            top_scores, top_indices = torch.topk(sim_matrix, k=args.top_k)

            for i in range(len(fact_embeddings)):
                rec = batch_records[i]
                matches = []
                for rank in range(args.top_k):
                    law_idx = top_indices[i][rank].item()
                    score = top_scores[i][rank].item()
                    law = laws[law_idx]
                    matches.append({
                        'rank': rank + 1,
                        'score': round(score, 4),
                        'law_article_id': law['id'],
                        'law_article_content': law['content']
                    })

                out_obj = {
                    "fact_id": get_fact_id(rec),
                    "fact_text": rec.get('fact_text') or rec.get('fact'),
                    "matches": matches
                }
                wf.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
                out_count += 1


        for rec in records:
            batch.append(rec)
            if len(batch) >= args.batch_size:
                process_batch(batch)
                batch = []
                print(f"Processed {out_count} facts...", end='\r')

        if batch:
            process_batch(batch)

    print(f"\nDone. Wrote {out_count} lines to {args.output}")


if __name__ == '__main__':
    main()
