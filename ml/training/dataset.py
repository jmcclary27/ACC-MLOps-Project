import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
# extract data
# split and tokenize sentences with auto tokenizer in transformers
# convert labels to numeric IDs

class ContractDataset(Dataset):
    #init runs after creating an instance of the class, it initializes the dataset with the provided texts and labels, and sets up the tokenizer and label mapping.
    def __init__(self, texts, labels, model_name = "distilbert-base-uncased", model_max_length=128):
        self.texts = texts
        self.labels = labels
        self.model_max_length = model_max_length
        #set up pretrained tokenizer https://huggingface.co/docs/transformers/model_doc/distilbert
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        uniquelabels = sorted(set(labels))
        self.label2id = {}
        #label mapping
        for index, label in enumerate(uniquelabels):
            self.label2id[label] = index

    def __len__(self):
        #returns number of samples, or the length of text list, same as the length of labels list
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # Tokenize the text, padding for consistency
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.model_max_length, return_tensors='pt')
        label_id = self.label2id[label]
        item = {"input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(label_id, dtype=torch.long)}
        #return everything together as a dictionary, and squeeze to remove extra batch dimension

        return item

if __name__ == "__main__":
    text1 = [
        "This agreement shall terminate upon breach.",
        "The tenant must pay rent on the first of each month."
    ]

    label1 = ["termination", "payment"]

    dataset = ContractDataset(text1, label1)
    #get item
    sample = dataset[1]

    print("Sample keys:", sample.keys())
    #check the items in dictionary
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Attention mask shape:", sample["attention_mask"].shape)
    print("Label:", sample["labels"])
    
    #input ids: tokens
    #attention mask: which tokens are actual data and which are padding
    #labels: numeric ID for the labels attatched to each text sample