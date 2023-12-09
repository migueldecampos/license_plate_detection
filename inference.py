import torch
import torchvision


SCORE_CUTOFF = 0.8


def open_image(image_path):
    image = torchvision.io.read_image(image_path)
    return image


def get_license_plate_boxes(model, image):
    """
    image: <torch.Tensor>, as out of torchvision.io.read_image(image_path)

    return: <torch.Tensor> of shape (N, 4), where N is the number of license plates found.
    """

    image_norm = image / 255
    model.eval()
    x = [image_norm]
    predictions = model(x)  # Returns predictions
    predictions = predictions[0]

    boxes = list()
    for box, label, score in zip(
        predictions["boxes"], predictions["labels"], predictions["scores"]
    ):
        if label == 1:
            if score >= SCORE_CUTOFF:
                boxes.append(box)
            else:
                break

    return torch.stack(boxes)
