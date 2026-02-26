# at the moment these functions are very simple, but we can add more as we develop the rest of the program

import pymupdf.layout # allows use of pdf as text and json
import pymupdf4llm
import spacy
import anyascii

def pdf_to_text(path, as_text=False, as_json=False):
    if as_text and as_json:
        raise ValueError("either as_text=True OR as_json=True")
    
    if as_text:
        return pymupdf4llm.to_text(path)
    
    if as_json:
        return pymupdf4llm.to_json(path)
    
    return pymupdf4llm.to_markdown(path)

def split_into_sentence(text, minumium_character_length=2):
    nlp = spacy.load("en_core_web_sm") # TODO: needs to be preloaded somewhere else
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        sent = sent.text
        sent.strip()
        if (sent == "") or (len(sent) <= minumium_character_length):
            continue
        sentences.append(sent)
        
    return sentences

def clean_text(text, unicode_to_ascii=False):
    if unicode_to_ascii:
        text = anyascii.anyascii(text)
    
    text = " ".join(text.split())
    
    return text