"""CPU-friendly, pretrained AMG-guided fusion supervised by pedestrian boxes.

AMG's frozen feature extractor works at 40x32. Its pretrained RGB output
convolution is adapted, then its output determines a smooth modality weight
map. Fusion uses the original 320x256 measurements to retain object details.
This is an AMG-guided pipeline, not native full-resolution AMG output.
"""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn
import torch.nn.functional as F
from .core import ROOT, BASE, load_model, sha

DATA=ROOT/'data/llvip_subset/manifest.json'
RUN=ROOT/'runs/pioggia_llvip_yolo'
CACHE=RUN/'cache'
WIDTH=320
LOW=(32,40)

def rows():
    data=json.loads(DATA.read_text(encoding='utf-8'))
    ids=[r['id'] for r in data]
    if len(ids)!=len(set(ids)):raise ValueError('Overlapping sample IDs')
    groups={s:{r['scene'] for r in data if r['split']==s} for s in ('train','validation','test')}
    if groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test']:
        raise ValueError('Overlapping scene identifiers')
    return data

def rain(image,severity,seed):
    rng=np.random.default_rng(seed)
    h,w=image.shape[:2]
    layer=Image.new('L',(w,h))
    draw=ImageDraw.Draw(layer)
    for _ in range(int(h*w*severity/95)):
        x,y=rng.integers(-10,w),rng.integers(-20,h)
        length=int(rng.integers(6,17))
        draw.line((int(x),int(y),int(x+length*.25),int(y+length)),fill=int(rng.integers(90,210)),width=1)
    streak=np.asarray(layer,dtype=np.float32)[...,None]/255
    base=Image.fromarray((np.clip(image,0,1)*255).astype('uint8')).filter(ImageFilter.GaussianBlur(.2+.45*severity))
    blurred=np.asarray(base,dtype=np.float32)/255
    result=blurred*(1-.15*severity)+.035*severity+streak*.7
    return np.clip(result,0,1).astype('float32')

def measurements(row):
    vi_image=Image.open(ROOT/row['visible']).convert('RGB')
    ir_image=Image.open(ROOT/row['thermal']).convert('L')
    if vi_image.size!=(row['width'],row['height']) or ir_image.size!=vi_image.size:
        raise ValueError('Image/annotation alignment mismatch')
    size=(WIDTH,round(row['height']/row['width']*WIDTH))
    vi=np.asarray(vi_image.resize(size),dtype=np.float32)/255
    ir=np.asarray(ir_image.resize(size),dtype=np.float32)[...,None]/255
    seed=1700+int(row['id'])
    severity=float(np.random.default_rng(seed).uniform(.4,.9)) if row['split']=='train' else .7
    degraded=rain(vi,severity,seed)
    box=torch.tensor(row['boxes'],dtype=torch.float32).reshape(-1,4)
    box*=torch.tensor([size[0]/row['width'],size[1]/row['height']]*2)
    return {'visible':torch.from_numpy(degraded.transpose(2,0,1).copy()),
            'thermal':torch.from_numpy(ir.transpose(2,0,1).copy()),'boxes':box,
            'severity':severity,'id':row['id'],'split':row['split']}

class CaptureFeatures(Exception):pass

def extract_features(model,visible,thermal):
    result={}
    def hook(module,args):
        result['features']=args[0].detach().clone()
        raise CaptureFeatures()
    handle=model.output_f.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():model(visible,thermal)
    except CaptureFeatures:pass
    finally:handle.remove()
    if 'features' not in result:raise RuntimeError('Feature extraction failed')
    return result['features']

def build_cache(progress=print):
    CACHE.mkdir(parents=True,exist_ok=True)
    model=load_model()
    torch.set_num_threads(2)
    model.requires_grad_(False)
    provenance={'base_sha256':sha(BASE),'manifest_sha256':sha(DATA),'low_size':list(LOW),
                'width':WIDTH,'rain_version':1,'feature':'input to original output_f convolution'}
    meta=CACHE/'provenance.json'
    if meta.exists() and json.loads(meta.read_text())!=provenance:
        raise ValueError('Cache provenance changed; use a new run directory')
    meta.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    data=rows()
    for index,row in enumerate(data,1):
        destination=CACHE/f"{row['id']}.pt"
        if destination.exists():continue
        item=measurements(row)
        vl=F.interpolate(item['visible'][None],size=LOW,mode='area')
        tl=F.interpolate(item['thermal'][None],size=LOW,mode='area')
        item['features']=extract_features(model,vl,tl).half()
        item['visible_low']=vl.half();item['thermal_low']=tl.half()
        torch.save(item,destination)
        progress(f"Cache {index}/{len(data)}: {row['split']} {row['id']}")
    head=copy.deepcopy(model.output_f).float().requires_grad_(True)
    torch.save(head.state_dict(),RUN/'original_head.pt')
    return head

def cached(row):
    return torch.load(CACHE/f"{row['id']}.pt",map_location='cpu',weights_only=True)

def blend(head,item,return_alpha=False):
    features=item['features'].float()
    reconstructed=head(features)
    vl=item['visible_low'].float();tl=item['thermal_low'].float()
    # Smooth projection of AMG's reconstruction onto a two-modality mixture.
    distance_ir=(reconstructed-tl).square().mean(1,keepdim=True)
    distance_vi=(reconstructed-vl).square().mean(1,keepdim=True)
    alpha=torch.sigmoid(4*(distance_ir-distance_vi))
    alpha_hi=F.interpolate(alpha,size=item['visible'].shape[-2:],mode='bilinear',align_corners=False)[0]
    output=alpha_hi*item['visible']+(1-alpha_hi)*item['thermal']
    return (output,alpha) if return_alpha else output

def make_head(state=None):
    state=state or RUN/'original_head.pt'
    head=nn.Conv2d(96,3,3,padding=1,bias=False)
    head.load_state_dict(torch.load(state,map_location='cpu',weights_only=True),strict=True)
    return head

def detector():
    from .yolo_frozen import load
    return load()

def detection_loss(model,image,boxes):
    from .yolo_frozen import loss
    return loss(model,image,boxes)

def predict(model,image):
    from .yolo_frozen import predict as infer_detector
    return infer_detector(model,image)

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    xy=np.maximum(a[:2],b[:2]);end=np.minimum(a[2:],b[2:])
    intersection=np.prod(np.maximum(end-xy,0))
    area_a=np.prod(np.maximum(a[2:]-a[:2],0));area_b=np.prod(np.maximum(b[2:]-b[:2],0))
    return float(intersection/max(area_a+area_b-intersection,1e-12))

def metrics(records,threshold=.35):
    total=sum(len(r['ground_truth']) for r in records)
    ranked=sorted([(score,r['id'],box) for r in records for score,box in zip(r['prediction']['scores'],r['prediction']['boxes'])],key=lambda x:x[0],reverse=True)
    truths={r['id']:r['ground_truth'] for r in records};used={key:set() for key in truths}
    tp=[];fp=[];operating_tp=operating_fp=0
    for score,key,box in ranked:
        candidates=[(iou(box,gt),i) for i,gt in enumerate(truths[key]) if i not in used[key]]
        overlap,index=max(candidates,default=(0,-1))
        correct=overlap>=.5
        if correct:used[key].add(index)
        tp.append(int(correct));fp.append(int(not correct))
        if score>=threshold:
            operating_tp+=int(correct);operating_fp+=int(not correct)
    if total and tp:
        ct=np.cumsum(tp);cf=np.cumsum(fp)
        recall=ct/total;precision=ct/np.maximum(ct+cf,1)
        r=np.r_[0,recall,1];p=np.r_[0,precision,0]
        p=np.maximum.accumulate(p[::-1])[::-1]
        ap=float(np.sum((r[1:]-r[:-1])*p[1:]))
    else:ap=0.
    return {'AP50':ap,'precision':operating_tp/max(operating_tp+operating_fp,1),
            'recall':operating_tp/max(total,1),'true_positives':operating_tp,'false_positives':operating_fp,
            'missed':total-operating_tp,'ground_truth_count':total,'images':len(records),'score_threshold':threshold,'IoU':.5}

def evaluate(head,model,data,method='network',progress=None):
    records=[]
    for index,row in enumerate(data,1):
        item=cached(row)
        with torch.no_grad():image=blend(head,item) if method=='network' else .5*item['visible']+.5*item['thermal']
        prediction=predict(model,image)
        records.append({'id':row['id'],'ground_truth':item['boxes'].tolist(),'prediction':prediction})
        if progress and index%10==0:progress(f"Detection {method}: {index}/{len(data)}")
    operating=RUN/'operating_point.json'
    threshold=json.loads(operating.read_text())['threshold'] if operating.exists() else .35
    return metrics(records,threshold),records

def train(epochs=3,progress=print):
    RUN.mkdir(parents=True,exist_ok=True)
    head=build_cache(progress)
    torch.set_num_threads(2)
    data=rows();training=[r for r in data if r['split']=='train'];validation=[r for r in data if r['split']=='validation']
    fixed=detector()
    original=copy.deepcopy(head).eval().requires_grad_(False)
    initial,_=evaluate(head,fixed,validation,progress=progress)
    best_ap=initial['AP50'];best_state=copy.deepcopy(head.state_dict());selected_epoch=0
    history=[{'epoch':0,'validation':initial}]
    optimizer=torch.optim.Adam(head.parameters(),lr=1e-4)
    rng=np.random.default_rng(17)
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(training));losses=[]
        for step,index in enumerate(order,1):
            item=cached(training[int(index)])
            output,alpha=blend(head,item,True)
            with torch.no_grad():_,anchor=blend(original,item,True)
            optimizer.zero_grad(set_to_none=True)
            loss=detection_loss(fixed,output,item['boxes'])+.1*F.mse_loss(alpha,anchor)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite supervised loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
            if step%10==0:progress(f'Epoch {epoch}/{epochs}: {step}/{len(training)}, loss={np.mean(losses):.4f}')
        score,_=evaluate(head,fixed,validation)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation':score})
        torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'history':history},RUN/'last.pt')
        if score['AP50']>best_ap:
            best_ap=score['AP50'];best_state=copy.deepcopy(head.state_dict());selected_epoch=epoch
        progress(f'Validation epoch {epoch}: AP50={score["AP50"]:.4f}; selected={selected_epoch}')
    evaluated_head=copy.deepcopy(head).eval()
    torch.save(evaluated_head.state_dict(),RUN/'evaluated_head.pt')
    head.load_state_dict(best_state)
    metadata={'base_sha256':sha(BASE),'dataset_sha256':sha(DATA),'detector_sha256':sha(ROOT/'pretrained/yolov8n.pt'),
              'domain_detector_sha256':sha(RUN/'domain_detector.pt') if (RUN/'domain_detector.pt').exists() else None,
              'train_ids':[r['id'] for r in training],'validation_ids':[r['id'] for r in validation],
              'test_ids':[r['id'] for r in data if r['split']=='test'],'epochs':epochs,'selected_epoch':selected_epoch,
              'evaluated_epoch':epochs,'recommended_epoch':selected_epoch,
              'history':history,'trainable_parameters':sum(p.numel() for p in head.parameters()),
              'architecture':'AMG-guided coarse neural weights, full-resolution visible/thermal blending',
              'objective':'Frozen YOLOv8n COCO classification+box+DFL loss on annotated pedestrian GT; 0.1 alpha regularization',
              'rain':'Simulated on visible only; severity train 0.4-0.9, validation/test 0.7',
              'selection':'AP50 validation recommends checkpoint including epoch zero; comparison evaluates predeclared fixed final epoch without choosing by test'}
    preparation=RUN/'detector_training.json'
    if preparation.exists():
        metadata['common_detector_preparation']=json.loads(preparation.read_text(encoding='utf-8'))
    operating=RUN/'operating_point.json'
    if operating.exists():
        metadata['operating_point']=json.loads(operating.read_text(encoding='utf-8'))
    metadata['detector_input_size']=[512,640]
    torch.save({'state':best_state,'metadata':metadata},RUN/'best.pt')
    torch.save(best_state,RUN/'best_head.pt')
    (RUN/'training.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    # Test is evaluated once, after all checkpoint selection has finished.
    test=[r for r in data if r['split']=='test'];results={}
    for label,candidate,method in [('media_fissa',original,'fixed'),('pretrained',original,'network'),('finetuned',evaluated_head,'network')]:
        summary,records=evaluate(candidate,fixed,test,method,progress)
        results[label]={'summary':summary,'records':records}
    report={'metadata':metadata,'test':results,
            'recall_difference':results['finetuned']['summary']['recall']-results['pretrained']['summary']['recall'],
            'extra_true_positives':results['finetuned']['summary']['true_positives']-results['pretrained']['summary']['true_positives']}
    from .yolo_frozen import load as load_detector
    raw_score,raw_records=evaluate(original,load_detector(domain=False),test,progress=progress)
    report['system_before_any_adaptation']={'summary':raw_score,'records':raw_records,
        'definition':'Original pretrained fusion and original COCO detector before common detector preparation. Same final images, input processing and shared threshold.'}
    destination=RUN/'results.json';pending=RUN/'results.pending.json'
    assert destination.resolve().is_relative_to(ROOT.resolve()) and pending.resolve().is_relative_to(ROOT.resolve())
    pending.write_text(json.dumps(report,indent=2),encoding='utf-8')
    pending.replace(destination)
    progress(json.dumps({key:value['summary'] for key,value in results.items()},indent=2))
    return report
