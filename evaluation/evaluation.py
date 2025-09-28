import sacrebleu
import json
with open('/data/test.json','r', encoding = 'utf-8') as f:
    test = json.load(f)
refs = [doc['vi'] for doc in test]
# inference_one.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def evaluation_base_model():
    # ======================
    # Config
    # ======================
    BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
    # ADAPTER_DIR = "/kaggle/working/Finetune/train/qwen3_envi_lora"   # thư mục adapter LoRA bạn đã train
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_NEW_TOKENS = 328

    # ======================
    # Load model + tokenizer
    # ======================
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )
    # model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()


    # ======================
    # Hàm dịch (EN -> VI)
    # ======================
    def translate_en_vi(en_sentence: str, max_new_tokens=MAX_NEW_TOKENS):
        prompt = f"Translate English to Vietnamese.\nSource: {en_sentence}\nTarget:"

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            do_sample=False,
            early_stopping=True
        )

        # decode rồi cắt phần prompt, chỉ lấy phần sau "Target:"
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Target:" in decoded:
            return decoded.split("Target:")[-1].strip()
        return decoded


    # ======================
    # Test một mẫu
    # ======================
    hypotheses = []
    for doc in test:
        
        hypothese = translate_en_vi(doc['en'])
        hypotheses.append(hypothese)


    # reference: list các câu chuẩn (gold translations)
    # hypothesis: list các câu model dịch ra


    bleu1 = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=2)
    print("BLEU-2:", bleu1.corpus_score(hypotheses, [refs]).score)

    bleu2 = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=4)
    print("BLEU-4:", bleu2.corpus_score(hypotheses, [refs]).score)
    
    return bleu1.corpus_score(hypotheses, [refs]).score, bleu2.corpus_score(hypotheses, [refs]).score

import re
def rere(txt):
    # Dùng regex để lấy phần main response
    match = re.split(r"<\\cot>", txt, maxsplit=1)
    
    if len(match) > 1:
        main_response = match[1].strip()
    else:
        main_response = txt.strip()
    
    # print("Main response:", main_response)
    return main_response

def evaluation_finetune_base():
    # inference_one.py

    refs = [doc['vi'] for doc in test[]]
    # inference_one.py

    # ======================
    # Config
    # ======================
    BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
    ADAPTER_DIR = "/train/qwen3_envi_lora"   # thư mục adapter LoRA bạn đã train
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_NEW_TOKENS = 1600

    # ======================
    # Load model + tokenizer
    # ======================
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()


    # ======================
    # Hàm dịch (EN -> VI)
    # ======================
    def translate_en_vi(en_sentence: str, max_new_tokens=MAX_NEW_TOKENS):
        prompt = f"Translate English to Vietnamese.\nSource: {en_sentence}\nTarget:"

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            do_sample=False,
            early_stopping=True
        )

        # decode rồi cắt phần prompt, chỉ lấy phần sau "Target:"
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Target:" in decoded:
            return decoded.split("Target:")[-1].strip()
        return decoded


    # ======================
    # Test một mẫu
    # ======================
    hypotheses = []
    for doc in test:
        
        hypothese = translate_en_vi(doc['en'])
        hypotheses.append(hypothese)
    
    bleu1 = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=2)
    print("BLEU-2:", bleu1.corpus_score(hypotheses, [refs]).score)          
    bleu2 = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=4)
    print("BLEU-4:", bleu2.corpus_score(hypotheses, [refs]).score)  
    return bleu1.corpus_score(hypotheses, [refs]).score, bleu2.corpus_score(hypotheses, [refs]).score

def evaluation_finetune_CoT():
    # inference_one.py

    refs = [doc['vi'] for doc in test[]]
    # inference_one.py

    # ======================
    # Config
    # ======================
    BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
    ADAPTER_DIR = "/train/qwen3_envi_lora_cot"   # thư mục adapter LoRA bạn đã train
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_NEW_TOKENS = 3868

    # ======================
    # Load model + tokenizer
    # ======================
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()


    # ======================
    # Hàm dịch (EN -> VI)
    # ======================
    def translate_en_vi(en_sentence: str, max_new_tokens=MAX_NEW_TOKENS):
        prompt = f"Translate English to Vietnamese.\nSource: {en_sentence}\nTarget:"

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            do_sample=False,
            early_stopping=True
        )

        # decode rồi cắt phần prompt, chỉ lấy phần sau "Target:"
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Target:" in decoded:
            return decoded.split("Target:")[-1].strip()
        return decoded



    hypotheses = []
    for doc in test:
        
        hypothese = translate_en_vi(doc['en'])
        hypotheses.append(rere(hypothese))
    
    bleu1 = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=2)
    print("BLEU-2:", bleu1.corpus_score(hypotheses, [refs]).score)          
    bleu2 = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=4)
    print("BLEU-4:", bleu2.corpus_score(hypotheses, [refs]).score)  
    return bleu1.corpus_score(hypotheses, [refs]).score, bleu2.corpus_score(hypotheses, [refs]).score

if __name__ == "__main__":
    print("Evaluating base model...")
    bleu2_base, bleu4_base = evaluation_base_model()
    print(f"Base model BLEU-2: {bleu2_base}, BLEU-4: {bleu4_base}")

    print("\nEvaluating fine-tuned model without CoT...")
    bleu2_finetune, bleu4_finetune = evaluation_finetune_base()
    print(f"Fine-tuned model BLEU-2: {bleu2_finetune}, BLEU-4: {bleu4_finetune}")

    print("\nEvaluating fine-tuned model with CoT...")
    bleu2_finetune_cot, bleu4_finetune_cot = evaluation_finetune_CoT()
    print(f"Fine-tuned CoT model BLEU-2: {bleu2_finetune_cot}, BLEU-4: {bleu4_finetune_cot}")
    #load to json file
    with open('bleu_scores.json', 'w') as f:
        results = {
            "base_model": {"BLEU-2": bleu2_base, "BLEU-4": bleu4_base},
            "fine_tuned_model": {"BLEU-2": bleu2_finetune, "BLEU-4": bleu4_finetune},
            "fine_tuned_CoT_model": {"BLEU-2": bleu2_finetune_cot, "BLEU-4": bleu4_finetune_cot}
        }
        json.dump(results, f, indent=4)
    