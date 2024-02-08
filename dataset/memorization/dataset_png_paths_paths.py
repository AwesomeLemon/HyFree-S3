from pathlib import Path

from typing import List

from PIL import Image
from torch.utils.data import Dataset


class DatasetPngPathsReturnPaths(Dataset):
    def __init__(self, paths: List[Path], transform, extension='.png'):
        self.images = []

        def _try_get_num_from_name(path: Path):
            try:
                return int(path.name.split(".")[0].split("_")[0])
            except:
                try:
                    return int(path.name.split(".")[0].split("_")[1])
                except:
                    return -1

        self.paths = [str(p.absolute()) for p in sorted(
            [p for p in paths if extension in p.name], key=_try_get_num_from_name)]
        for path in self.paths:
            im = Image.open(path).convert('RGB')
            self.images.append(im)

        self.transform = transform
        assert self.transform is not None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        path = self.paths[idx]

        image = self.transform(image)

        return image, path