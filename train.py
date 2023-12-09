"""
Code adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
"""
import torch

import pytorch_utils.engine
import license_plate_dataset
import license_plate_model
import pytorch_utils.utils


def train(
    dataset_root,
    device,
    number_of_epochs,
    checkpoint_freq,
    checkpoint_path,
    epoch_size=None,
    test_size=None,
):
    dataset = license_plate_dataset.LLPDataset(dataset_root, mode="train")
    dataset_test = license_plate_dataset.LLPDataset(dataset_root, mode="test")

    # reduce size of dataset
    if epoch_size is not None:
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:epoch_size])
    if test_size is not None:
        test_indices = torch.randperm(len(dataset_test)).tolist()
        dataset_test = torch.utils.data.Subset(dataset_test, test_indices[:test_size])

    # define training and test data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=pytorch_utils.utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=pytorch_utils.utils.collate_fn,
    )

    # get the model using our helper function
    model = license_plate_model.get_model_instance_segmentation()

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train
    for epoch in range(number_of_epochs):
        # train for one epoch, printing every 10 iterations
        pytorch_utils.engine.train_one_epoch(
            model,
            optimizer,
            data_loader,
            device,
            epoch,
            print_freq=10,
            checkpoint_freq=checkpoint_freq,
            checkpoint_path=checkpoint_path,
        )
        # update the learning rate
        lr_scheduler.step()

    return model


if __name__ == "__main__":
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train(
        dataset_root="../data/LargeLicensePlateDetectionDataset",
        device=device,
        number_of_epochs=1,
        checkpoint_freq=50,
        checkpoint_path=os.path.abspath("./license_plate_detection_model"),
        epoch_size=10,
        test_size=5,
    )

    print(model)
