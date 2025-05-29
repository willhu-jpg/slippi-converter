from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class FrameDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = list(Path(root_dir).rglob("*.jpg"))
        self.transform = T.Compose([
            T.Resize((64,64)),
            T.ToTensor(),                   # [0,1]
            T.Normalize(0.5, 0.5),          # -> [–1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)