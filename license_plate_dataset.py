"""
Here we implement a Pytorch Dataset for the Large-License-Plate-Detection-Dataset 
(see https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset/data)

Code adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
"""
import csv
import os

import torch
import torchvision

from torchvision import tv_tensors


def _read_label_file(file_path, image_shape):
    _, height, width = image_shape
    out = list()
    with open(file_path, newline="\n") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        for row in reader:
            x_center = int(float(row[1]) * width)
            y_center = int(float(row[2]) * height)
            label_half_width = int(float(row[3]) * width / 2)
            label_half_width = (
                label_half_width if label_half_width > 0 else label_half_width + 1
            )
            label_half_height = int(float(row[4]) * width / 2)
            label_half_height = (
                label_half_height if label_half_height > 0 else label_half_height + 1
            )
            x1 = x_center - label_half_width
            y1 = y_center - label_half_height
            x2 = x_center + label_half_width
            y2 = y_center + label_half_height
            out.append([x1, y1, x2, y2])

    return out


class LLPDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        # mode is one of 'train', 'val', 'test'
        self.mode = mode
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images", mode))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels", mode))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.mode, self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.mode, self.labels[idx])
        img = torchvision.io.read_image(img_path) / 255

        # get bounding box coordinates for each license plate
        label_list = _read_label_file(file_path=label_path, image_shape=img.shape)
        num_objs = len(label_list)
        boxes = torch.tensor(label_list, dtype=torch.float)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # masks
        masks = list()
        for box in boxes:
            masks.append(torch.zeros(img.shape[1:]))
            masks[-1][
                int(box[0].item()) : int(box[2].item()),
                int(box[1].item()) : int(box[3].item()),
            ] = 1
        masks = torch.stack(masks)
        masks = tv_tensors.Mask(masks)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": self.imgs[idx],
        }

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    import pytorch_utils.utils

    root = "../data/LargeLicensePlateDetectionDataset"

    dataset = LLPDataset(root=root, mode="train")
    print(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=pytorch_utils.utils.collate_fn
    )

    images, targets = next(iter(data_loader))

    print(images)
    print()
    print(targets)
