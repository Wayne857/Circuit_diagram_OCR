from PIL import Image
import torch
from modelscope import AutoModelForCausalLM, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHOSEN_TASK = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula'
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

model_path = "PaddlePaddle/PaddleOCR-VL"
image_path = rf"C:\Users\11\Desktop\pj\image_extract\runs\run20\detection_results\text\Power_detected_text_1_2_conf_0.91.jpg"
image = Image.open(image_path).convert("RGB")

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {"role": "user",         
     "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[CHOSEN_TASK]},
        ]
    }
]
inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 	
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)

outputs = model.generate(**inputs, max_new_tokens=1024)
outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(outputs)