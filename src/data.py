import os
# PIL: python image library
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset


class Stenosis_Dataset(Dataset):
    def __init__(self, mode="train"):
        data_root = f"./data/stenosis_data/{mode}"
        file_names = os.listdir(f"./data/stenosis_data/{mode}/images")
        # here we normalize imags to float tensors in [0,1]
        self.imgs = [pil_to_tensor(Image.open(f"{data_root}/images/{file_name}"))[0, :, :].unsqueeze(0) / 255
                     for file_name in file_names]
        # masks will then be converted to
        self.masks = [pil_to_tensor(Image.open(f"{data_root}/masks/{file_name}")).squeeze(0) / 255
                      for file_name in file_names]

    def __len__(self):
        # length of Stenosis_Dataset is the number of images in the dataset
        # how to access: m=Stenosis_Dataset(Dataset)    len(m)
        return len(self.imgs)

    def __getitem__(self, idx)->tuple:
        # how to access: m=Stenosis_Dataset(Dataset)    m[idx]
        """
        m[0][0].shape==torch.Size([1, 512, 512])
        m[0][0].dtype==torch.float32
        m[0][0] picks value in (0,1)

        m[0][1].shape==torch.Size([512, 512])
        m[0][1].dtype==torch.int64
        m[0][1] picks value in {0,1}
        """
        return self.imgs[idx].float(), self.masks[idx].long()

