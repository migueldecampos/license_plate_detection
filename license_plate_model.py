"""
Code adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
"""
import torchvision


def get_model_instance_segmentation():
    num_classes = 2  # 1 class (license plate) + background

    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = (
        torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )
    )

    return model


if __name__ == "__main__":
    import pickle

    model = get_model_instance_segmentation()

    with open("license_plate_detection_model-test.pkl", "wb") as p:
        pickle.dump(model, p)
