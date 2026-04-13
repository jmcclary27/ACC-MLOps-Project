# ml/debug/inspect_chunking.py

from ml.data.text_helpers import chunk_legal_text_with_offsets, pdf_to_text

def inspect(path):
    text = pdf_to_text(path, as_text=True)
    chunks = chunk_legal_text_with_offsets(text)

    print(f"\nTotal chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:30]):
        print(f"\n--- Chunk {i} ---")
        print(f"Chars: {chunk['start_char']} → {chunk['end_char']}")
        print(f"Length: {len(chunk['text'])}")
        print(chunk["text"])
        
def verify_offsets(text, chunks):
    all_good = True

    for i, chunk in enumerate(chunks):
        start = int(chunk["start_char"])
        end = int(chunk["end_char"])
        original = text[start:end]
        chunk_text = str(chunk["text"]).strip()

        if original.strip() != chunk_text:
            print(f"\nOFFSET MISMATCH in chunk {i}")
            print(f"start={start}, end={end}")
            print("Original slice:")
            print(repr(original[:300]))
            print("Chunk text:")
            print(repr(chunk_text[:300]))
            all_good = False

    return all_good

def analyze_chunks(chunks):
    lengths = [len(str(c["text"])) for c in chunks]
    if not lengths:
        print("No chunks found.")
        return

    avg_len = sum(lengths) / len(lengths)
    small = sum(1 for x in lengths if x < 40)
    large = sum(1 for x in lengths if x > 400)

    print(f"Total chunks: {len(chunks)}")
    print(f"Average length: {avg_len:.2f}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Chunks < 40 chars: {small}")
    print(f"Chunks > 400 chars: {large}")
    
def run_validation_checks(text, chunks):
    print("\nValidation checks")
    print("-----------------")

    offsets_ok = verify_offsets(text, chunks)
    print(f"Offsets valid: {offsets_ok}")

    lengths = [len(str(c['text'])) for c in chunks]
    too_small = sum(1 for x in lengths if x < 40)
    too_large = sum(1 for x in lengths if x > 400)

    print(f"Too many tiny chunks? {'YES' if too_small > len(chunks) * 0.2 else 'NO'}")
    print(f"Too many huge chunks? {'YES' if too_large > len(chunks) * 0.2 else 'NO'}")

if __name__ == "__main__":
    inspect("ml/data/test.pdf")