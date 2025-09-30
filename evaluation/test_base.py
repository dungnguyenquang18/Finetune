import argparse
import json
import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sacrebleu
from tqdm import tqdm


def load_dataset(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_target_from_decoded(decoded: str) -> str:
    # Expect prompt contains 'Target:' and we want the part after it
    if 'Target:' in decoded:
        return decoded.split('Target:')[-1].strip()
    return decoded.strip()



def evaluate(
    base_model: str,
    adapter_dir: str,
    data_file: str,
    max_new_tokens: int = 500,
    beam: int = 3,
    device: str = None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading data from {data_file}...")
    docs = load_dataset(data_file)
    sources = [d['en'] for d in docs]
    references = [d['vi'] for d in docs]

    print(f"Loading tokenizer and base model: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("Loading base model (may use device_map='auto')...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )

    
    model = base

    model.eval()

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device)

    hypotheses: List[str] = []
    outputs_detail: List[dict] = []

    print("Generating translations...")
    for src, ref in tqdm(list(zip(sources, references)), desc="Generating"):
        prompt = f"Translate English to Vietnamese.\nSource: {src}\nTarget:"
        inputs = tokenizer(prompt, return_tensors='pt')
        try:
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except Exception:
            pass

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=beam,
                do_sample=False,
                early_stopping=True,
            )

        decoded_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tgt = extract_target_from_decoded(decoded_full)
        hypotheses.append(tgt)

        outputs_detail.append({
            "src": src,
            "ref": ref,
            "decoded_full": decoded_full,
            "hypothesis": tgt,
        })

    print("Computing sacreBLEU and chrF...")
    bleu2 = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=2).corpus_score(hypotheses, [references])
    bleu4 = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=4).corpus_score(hypotheses, [references])
    chrf = sacrebleu.metrics.CHRF().corpus_score(hypotheses, [references])

    results = {
        'bleu_2': float(bleu2.score),
        'bleu_4': float(bleu4.score),
        'chrf': float(chrf.score),
        'n_samples': len(hypotheses),
    }

    print(f"BLEU-2: {results['bleu_2']:.2f}")
    print(f"BLEU-4: {results['bleu_4']:.2f}")
    print(f"chrF: {results['chrf']:.2f}")

    try:
        from bert_score import score as bert_score
        print("Computing BERTScore (may take time)...")
        P, R, F1 = bert_score(hypotheses, references, lang='vi', rescale_with_baseline=True)
        results['bertscore_f1_mean'] = float(F1.mean().cpu().item())
    except Exception:
        print("bert_score not available or failed; skipping BERTScore.")

    return outputs_detail, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen3-0.6B-Base')
    parser.add_argument('--adapter_dir', type=str, default='Finetune/finetune/qwen3_envi_lora')
    parser.add_argument('--data_file', type=str, default='Finetune/data/test.json')
    parser.add_argument('--max_new_tokens', type=int, default=666)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--out', type=str, default='test_base.json')
    args, _ = parser.parse_known_args()

    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()
    data_file = args.data_file
    if not os.path.isabs(data_file):
        data_file = os.path.normpath(os.path.join(base_dir, args.data_file))

    adapter_dir = args.adapter_dir
    if adapter_dir and not os.path.isabs(adapter_dir):
        adapter_dir = os.path.normpath(os.path.join(base_dir, args.adapter_dir))

    outputs_detail, results = evaluate(
        base_model=args.base_model,
        adapter_dir=adapter_dir,
        data_file=data_file,
        max_new_tokens=args.max_new_tokens,
        beam=args.beam,
    )

    print("Results:")
    print(json.dumps(results, indent=2))

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'outputs': outputs_detail}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
