import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.data.text_helpers import pdf_to_text, split_into_sentence, clean_text
# TEST RESULTS:
# For all three methods, the tests passed successfully.
# new line characters \n are added to the end of the text whenever there is a new line in the PDF document.
# clean text takes care of this
def test_pdf_to_text():
    
    # test with a sample pdf file
    text = pdf_to_text("tests/sample.pdf", as_text=True)
    #check for string type, non empty, correct content
    assert isinstance(text, str)
    assert len(text) > 0
    assert text == "Hello, this is a test PDF. This agreement shall terminate upon breach. The tenant must pay a rent of $500 on the first of each month. \n\n"
    #checking for the new line characters
    print(repr(text))
def test_split_into_sentence():
    text = pdf_to_text("tests/sample.pdf", as_text=True)
    sentences = split_into_sentence(text)
    result = ["Hello, this is a test PDF.", "This agreement shall terminate upon breach.", "The tenant must pay a rent of $500 on the first of each month. \n\n"]
    assert result == sentences
def test_clean_text():
    text = pdf_to_text("tests/sample.pdf", as_text=True)
    cleaned_text = clean_text(text)
    assert cleaned_text == "Hello, this is a test PDF. This agreement shall terminate upon breach. The tenant must pay a rent of $500 on the first of each month."

test_pdf_to_text()
test_split_into_sentence()
test_clean_text()