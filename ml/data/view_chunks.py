from pathlib import Path
from text_helpers import pdf_to_text, chunk_legal_text_with_offsets

PDF_PATH = Path("ml/data/test.pdf")  # change this

def main():
    text = pdf_to_text(str(PDF_PATH), as_text=True)
    chunks = chunk_legal_text_with_offsets(text)

    print(f"\nTotal chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, start=1):
        print("=" * 80)
        print(f"Chunk {i}")
        print(f"[{chunk['start_char']} -> {chunk['end_char']}] (len={len(chunk['text'])})")
        print(chunk["text"])
        print()

if __name__ == "__main__":
    main()