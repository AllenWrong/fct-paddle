from paddle.io import DataLoader
import os
from paddle.vision import DatasetFolder
from .data_transforms import get_data_transforms
import paddle
from .imagenet import ImageNet2012Dataset


def get_image_folder(
        name: str, data_root: str, num_workers: int,
        batch_size: int, sampler=None, num_classes=None
):
    """
    Args:
        name: name of dataset (currently cifar100, imagenet)
        data_root: path to a directory with training and validation subdirs of the dataset.
        num_workers: number of workers for data loader.
        batch_size: size of the batch per GPU.
        num_classes: mumber of classes to use for training. This should
            be smaller ot equal than the total number of classes in the
        dataset. not that for evaluation we use all classes.
    Returns:

    """
    if name == "cifar100":
        return Cifar100Folder(data_root, num_workers, batch_size, sampler, num_classes)
    elif name == 'imagenet':
        return ImageNetFolder(data_root, num_workers, batch_size, sampler, num_classes)


class ImageNetFolder:
    def __init__(self, data_root: str, num_workers: int, batch_size: int, sampler: str = None,
                 num_classes: int = None
    ) -> None:
        train_transforms, val_transforms = get_data_transforms("imagenet")
        self.train_dataset = ImageNet2012Dataset(data_root, mode="train", transform=train_transforms)
        self.val_dataset = ImageNet2012Dataset(data_root, mode="val", transform=val_transforms)

        if num_classes is not None:
            sub_image_path_list = []
            sub_label_list = []
            for img_path, label in zip(self.train_dataset.img_path_list, self.train_dataset.label_list):
                if label < num_classes:
                    sub_image_path_list.append(img_path)
                    sub_label_list.append(label)

            self.train_dataset.img_path_list = sub_image_path_list
            self.train_dataset.label_list = sub_label_list

        if sampler == "dist":
            self.train_sampler = paddle.io.DistributedBatchSampler(
                self.train_dataset, batch_size, shuffle=True, drop_last=False
            )
            self.val_sampler = paddle.io.DistributedBatchSampler(
                self.val_dataset, batch_size, shuffle=False, drop_last=False
            )

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=self.train_sampler,
                num_workers=num_workers
            )

            # Note: for evaluation we use all classes.
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_sampler=self.val_sampler,
                num_workers=num_workers
            )
        else:
            self.train_sampler = None
            self.val_sampler = None

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

            # Note: for evaluation we use all classes.
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )


class Cifar100Folder:
    def __init__(self, data_root: str, num_workers: int, batch_size: int, sampler: str = None,
                 num_classes: int = None
    ) -> None:
        # Data loading code
        traindir = os.path.join(data_root, "training")
        valdir = os.path.join(data_root, "validation")

        train_transforms, val_transforms = get_data_transforms("cifar100")

        self.train_dataset = DatasetFolder(traindir, transform=train_transforms)
        self.val_dataset = DatasetFolder(valdir, transform=val_transforms)

        # Filtering out some classes
        if num_classes is not None:
            self.train_dataset.samples = [
                (path, cls_num)
                for path, cls_num in self.train_dataset.samples
                if cls_num < num_classes
            ]

        self.train_dataset.samples = self.train_dataset.samples

        if sampler == "dist":
            self.train_sampler = paddle.io.DistributedBatchSampler(
                self.train_dataset, batch_size, shuffle=True, drop_last=False
            )
            self.val_sampler = paddle.io.DistributedBatchSampler(
                self.val_dataset, batch_size, shuffle=False, drop_last=False
            )

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=self.train_sampler,
                num_workers=num_workers
            )

            # Note: for evaluation we use all classes.
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_sampler=self.val_sampler,
                num_workers=num_workers
            )
        else:
            self.train_sampler = None
            self.val_sampler = None

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

            # Note: for evaluation we use all classes.
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
