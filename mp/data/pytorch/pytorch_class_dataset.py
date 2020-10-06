
class ImgClassificationDataset(PytorchDataset):
    """Classification dataset that loads images from file paths.
    """
    def __init__(self, dataset, ix_lst=None, resize=None, transform=[], norm=None):
        super().__init__(dataset, ix_lst, resize, transform, norm)
        self.x_paths = [ex.x for ex in self.instances]
        self.y = torch.LongTensor([float(ex.y) for ex in self.instances])

    def __getitem__(self, idx):
        image = Image.open(self.x_paths[idx])
        image = self.transform(image)
        return image, self.y[idx]
