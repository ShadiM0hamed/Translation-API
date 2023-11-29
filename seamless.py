from transformers import AutoProcessor, SeamlessM4TModel

processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

def detect_language(input_text):
    try:
        language_code = detect(input_text)
        return language_code
    except Exception as e:
        # Handle exceptions (e.g., if language detection fails)
        return None
def translator(contents, src_lang="eng", tgt_lang="arb", chunk_size=1000):
    if src_lang is None:
        detected_language = detect_language(contents)
        if detected_language is not None:
            src_lang = detected_language
        else:
            raise ValueError("Failed to detect the input language.")
    total_length = len(contents)
    translated_text = ''
    start_idx = 0
    while start_idx < total_length:
        # Extract a chunk of text of the defined size or until the nearest period before the chunk size
        end_idx = min(start_idx + chunk_size, total_length)
        nearest_period_idx = contents.rfind('.', start_idx, end_idx)
        if nearest_period_idx != -1 and nearest_period_idx != start_idx:
            # If a period was found within the chunk size, adjust the end index to include the period
            end_idx = nearest_period_idx + 1
        # Extract the chunk based on the determined indices
        chunk = contents[start_idx:end_idx]
        text_inputs = processor(text=chunk, src_lang=src_lang, return_tensors="pt")
        # Translate the chunk of text
        output_tokens = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)
        # Decode the translated text for this chunk and append it to the result
        translated_chunk = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        translated_text += translated_chunk
        # Update the start index for the next chunk
        start_idx = end_idx
    return translated_text
