import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- İŞTE O CANAVAR! ARTIK BUNU KALDIRACAK GÜCÜMÜZ VAR! ---
model_adi = "huihui-ai/Huihui-gemma-3n-E4B-it-abliterated"
bot_marka_adi = "Hürbilge"

print(f"'{model_adi}' modeli Oracle canavarı üzerinde yükleniyor...")

model = AutoModelForCausalLM.from_pretrained(model_adi, torch_dtype="auto", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_adi, trust_remote_code=True)

print("Model başarıyla yüklendi! Hürbilge hizmetinizde.")

def sohbet_fonksiyonu(mesaj, gecmis):
    sohbet_formati = []
    for kullanici_mesaji, bot_cevabi in gecmis:
        sohbet_formati.append({"role": "user", "content": kullanici_mesaji})
        sohbet_formati.append({"role": "assistant", "content": bot_cevabi})
    sohbet_formati.append({"role": "user", "content": mesaj})
    
    input_ids = tokenizer.apply_chat_template(sohbet_formati, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=512, do_sample=True, temperature=0.7)
    cevap = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return cevap

arayuz = gr.ChatInterface(
    fn=sohbet_fonksiyonu,
    title=bot_marka_adi,
    description=f"Ben {bot_marka_adi}. Sansürsüz, filtresiz ve özgür bir yapay zekayım. Oracle Cloud üzerinde hizmetinizdeyim."
)

# --- DİKKAT! SUNUCUNUN DIŞARIYA AÇILMASINI SAĞLAYAN SİHİRLİ AYAR ---
arayuz.launch(server_name="0.0.0.0", server_port=7860)
