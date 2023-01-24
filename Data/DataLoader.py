from torch.utils.data import DataLoader

class FinTextDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=10, shuffle=False, num_workers=0):
        super(FinTextDataLoader, self).__init__(dataset, batch_size, shuffle, num_workers)
    
    def add_transforms(self, transforms):
        self.transforms = transforms
        
    def __iter__(self):
        for data in self:
            yield data
