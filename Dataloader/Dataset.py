from torch.utils.data import Dataset

class FinTextDataset(Dataset):
    def __init__(self, df, bundle_size=15):
        super().__init__()
        self.df = df
        self.bundle_size = bundle_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = self.df.iloc[index, :-1]
        y = self.df.iloc[index, -1]
        return x, y