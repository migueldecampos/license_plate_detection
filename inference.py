import torch
import torchvision


SCORE_CUTOFF = 0.7


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

    return torch.stack(boxes) if boxes else torch.zeros((0, 4))


if __name__ == "__main__":
    import pickle
    import sys

    if len(sys.argv) == 3:
        with open(sys.argv[1], "rb") as p:
            model = pickle.load(p)
        image_path = sys.argv[2]
        image = torchvision.io.read_image(image_path)
        boxes = get_license_plate_boxes(model, image)
        colors = [
            "blue",
            "yellow",
            "red",
            "orange",
            "pink",
            "black",
            "green",
            "brown",
            "gold",
            "gray",
            "lime",
            "magenta",
            "olive",
            "violet",
            "cyan",
            "navy",
        ]
        output = torchvision.utils.draw_bounding_boxes(
            image, boxes, colors=colors, width=3
        )
        output_path = image_path[:-4] + "_output.jpg"
        torchvision.io.write_jpeg(output, output_path)
    else:
        print("python inference.py <path to model pickle file> <path to image>")
