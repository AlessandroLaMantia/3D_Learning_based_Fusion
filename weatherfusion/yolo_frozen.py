"""Frozen official YOLOv8n, differentiable wrt the fused image only."""
import os
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from .core import ROOT

os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))
config_path=ROOT/'.ultralytics'
config_path.mkdir(parents=True,exist_ok=True)

def load(domain=True):
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT,SETTINGS
    SETTINGS.update({'sync':False,'hub':False})
    model=YOLO(str(ROOT/'pretrained/yolov8n.pt'),task='detect').model.float().eval()
    args=dict(DEFAULT_CFG_DICT)
    args.update(model.args if isinstance(model.args,dict) else vars(model.args))
    model.args=SimpleNamespace(**args)
    prepared=ROOT/'runs/pioggia_llvip_yolo/domain_detector.pt'
    if domain and prepared.exists():
        model.load_state_dict(torch.load(prepared,map_location='cpu',weights_only=True),strict=True)
    model.requires_grad_(False)
    return model

def loss(model,image,boxes):
    h,w=image.shape[-2:]
    xywh=boxes.clone()
    xywh[:,:2]=(boxes[:,:2]+boxes[:,2:])/2
    xywh[:,2:]=boxes[:,2:]-boxes[:,:2]
    xywh/=torch.tensor([w,h,w,h],dtype=torch.float32)
    batch={'img':F.interpolate(image[None],size=(512,640),mode='bilinear',align_corners=False),'batch_idx':torch.zeros(len(boxes)),
           'cls':torch.zeros((len(boxes),1)),'bboxes':xywh}
    model.model[-1].training=True
    try:
        values=model.loss(batch)
        return values[0].sum()
    finally:model.model[-1].training=False

def predict(model,image):
    from ultralytics.utils.nms import non_max_suppression
    model.eval()
    with torch.no_grad():
        output=model(F.interpolate(image[None],size=(512,640),mode='bilinear',align_corners=False))
        prediction=non_max_suppression(output,conf_thres=.05,iou_thres=.5,classes=[0],max_det=100)[0]
    h,w=image.shape[-2:]
    prediction[:,:4]*=torch.tensor([w/640,h/512,w/640,h/512])
    prediction[:,[0,2]].clamp_(0,w);prediction[:,[1,3]].clamp_(0,h)
    return {'boxes':prediction[:,:4].cpu().tolist(),'scores':prediction[:,4].cpu().tolist()}
