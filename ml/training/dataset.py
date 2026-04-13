# ml/training/dataset.py
import torch
from torch.utils.data import Dataset


class ContractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, model_max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.model_max_length = model_max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.model_max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label2id[label], dtype=torch.long),
        }

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