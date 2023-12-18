"""
Copied from https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py.
"""
import math
import pickle
import sys
import time
import json

import torch
import torchvision.models.detection.mask_rcnn
import pytorch_utils.utils
from pytorch_utils.coco_eval import CocoEvaluator
from pytorch_utils.coco_utils import get_coco_api_from_dataset


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq,
    checkpoint_freq,
    checkpoint_path,
    scaler=None,
):
    model.train()
    metric_logger = pytorch_utils.utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", pytorch_utils.utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    step = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        image_ids = list()
        for target_dict in targets:
            image_ids.append(target_dict["image_id"])
        images = list(image.to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]
        error_occured = False
        try:
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        except Exception as e:
            error_occured = True
            print("ERROR!")
            print(e)
            print("image_ids:", image_ids)
            print()
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = pytorch_utils.utils.reduce_dict(loss_dict)
        loss_dict_reduced_python_types = {
            key: value.item() for key, value in loss_dict_reduced.items()
        }
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if not error_occured:
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

        if step % checkpoint_freq == 0:
            with open(checkpoint_path + "_{}_{}.pkl".format(epoch, step), "wb") as p:
                pickle.dump(model, p)
            with open(checkpoint_path + "_{}_{}.json".format(epoch, step), "w") as f:
                json.dump(loss_dict_reduced_python_types, f)
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        step += 1

    with open(checkpoint_path + "_{}_final.pkl".format(epoch), "wb") as p:
        pickle.dump(model, p)

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = pytorch_utils.utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
