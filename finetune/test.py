import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ======================
# Config
# ======================
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
ADAPTER_DIR = "finetune/finetune/qwen3_envi_lora"   # thư mục adapter LoRA bạn đã train
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 26

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
        num_beams=3,
        do_sample=False,
        early_stopping=True,
    )

    # decode rồi cắt phần prompt, chỉ lấy phần sau "Target:"
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Target:" in decoded:
        return decoded.split("Target:")[-1].strip()
    return decoded


# ======================
# Test một mẫu
# ======================
if __name__ == "__main__":
    en_sample = "The quick brown fox jumps over the lazy dog."
    vi_translation = translate_en_vi(en_sample)
    print("English:", en_sample)
    print("Vietnamese:", vi_translation)
