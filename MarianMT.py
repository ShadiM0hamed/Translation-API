from transformers import MarianMTModel, MarianTokenizer
import torch

def load_model_and_tokenizer(model_name):
    tokenizer = MarianTokenizer.from_pretrained(f"./{model_name}-tokenizer", local_files_only=True)
    model = MarianMTModel.from_pretrained(f"./{model_name}-model", local_files_only=True)
    return model, tokenizer

# Load models and tokenizers
model_en_ar, tokenizer_en_ar = load_model_and_tokenizer("Helsinki-NLP/opus-mt-en-ar")
model_ar_en, tokenizer_ar_en = load_model_and_tokenizer("Helsinki-NLP/opus-mt-ar-en")

def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Function to translate English to Arabic
def translate_en_to_ar(text):
    return translate(text, model_en_ar, tokenizer_en_ar)

# Function to translate Arabic to English
def translate_ar_to_en(text):
    return translate(text, model_ar_en, tokenizer_ar_en)

def translater(text, tgt_lang):
    if tgt_lang == 'arb':
      return translate_en_to_ar(text)
    else:
      return translate_ar_to_en(text)
def translate_by_sentences(text, tgt_lang):
    max_length = 100
 #   print(text)
    sentences = text.split('.')  # Splitting the text by newline characters
#    print(sentences)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
#        print(sentence, "##################################")
        # Check if adding the next sentence would exceed the max length
        if len(current_chunk) + len(sentence) + 1 > max_length and current_chunk:
 
            current_chunk = translater(current_chunk, tgt_lang)

            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            # Add a newline character if the chunk isn't empty
            if current_chunk:
                current_chunk += "\n"
            current_chunk += sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        current_chunk = translater(current_chunk, tgt_lang)
        chunks.append(current_chunk)

    print(str(" ".join(chunks)))
    return str(" ".join(chunks))

