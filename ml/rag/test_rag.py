from ml.rag.answer_question import run_rag


def test_rag_manual():
    question = "What is the territory?"

    result = run_rag(question)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(result.answer)

    print("\nCITATIONS:")
    for c in result.citations:
        print("-", c.text[:100])


if __name__ == "__main__":
    test_rag_manual()