import torch 
from datasets import load_dataset
from torch.utils.data import Dataset
from datasets import load_dataset

class CelebADataset(Dataset): 
    def __init__(self, data_path, transform): 
        data = load_dataset(data_path) 
        self.data = data['train']
        self.transform = transform 
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index): 
        image = self.data[index]['image'].convert("RGB") 
        return self.transform(image), torch.tensor(index).long()


