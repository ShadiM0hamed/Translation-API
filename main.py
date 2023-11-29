from seamless import translator
from MarianMT import translate_by_sentences




def translate_process(text, lang, model):

	if model == 'Seamless':
		out = translator(text, tgt_lang=lang)
		print(out)
		return out
	elif model == 'MarianMT':
		out = translate_by_sentences(text, lang)
		print(out)
		return out
		
	else:
		pass
