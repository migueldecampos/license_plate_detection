"""
Code adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
"""
import torch

import engine
import license_plate_dataset
import license_plate_model
import utils


def train(dataset_root, device, number_of_epochs=1):
    dataset = license_plate_dataset.LLPDataset(
        "../data/LargeLicensePlateDetectionDataset", mode="train"
    )
    dataset_test = license_plate_dataset.LLPDataset(
        "../data/LargeLicensePlateDetectionDataset", mode="test"
    )

    # define training and test data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
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
        engine.train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10
        )
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        engine.evaluate(model, data_loader_test, device=device)

    return model


if __name__ == "__main__":
    import pickle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train(
        dataset_root="../data/LargeLicensePlateDetectionDataset", device=device
    )

    print(model)

    with open("license_plate_detection_model.pkl", "wb") as p:
        pickle.dump(model, p)
