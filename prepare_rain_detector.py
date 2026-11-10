"""Fit a common YOLO head on train only, then freeze it for fusion comparisons."""
import copy
import json
import numpy as np
import torch
from weatherfusion.rain_detection import RUN,CACHE,rows,cached,predict,metrics,make_head,blend,build_cache
from weatherfusion.yolo_frozen import load,loss

def prepare(epochs=6):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    data=rows();training=[r for r in data if r['split']=='train'];validation=[r for r in data if r['split']=='validation']
    if any(not (CACHE/f"{r['id']}.pt").exists() for r in training+validation):
        build_cache()
    model=load(domain=False)
    # Only YOLO's detection head is adapted; its feature extractor and BN buffers stay frozen.
    model.model[-1].requires_grad_(True)
    for name,parameter in model.model[-1].named_parameters():
        if name.startswith('dfl.'):
            parameter.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    adapted_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    def validation_records():
        records=[]
        for row in validation:
            item=cached(row);image=.5*item['visible']+.5*item['thermal']
            records.append({'id':row['id'],'ground_truth':item['boxes'].tolist(),'prediction':predict(model,image)})
        return records
    history=[]
    initial=metrics(validation_records())
    best_ap=initial['AP50'];best_state=copy.deepcopy(model.state_dict());selected=0
    history.append({'epoch':0,'validation':initial})
    print(f'Detector initial validation AP50={best_ap:.4f}',flush=True)
    rng=np.random.default_rng(17)
    for epoch in range(1,epochs+1):
        losses=[]
        for step,index in enumerate(rng.permutation(len(training)),1):
            item=cached(training[int(index)])
            kind=(int(index)+epoch)%3
            image=item['visible'] if kind==0 else item['thermal'].repeat(3,1,1) if kind==1 else .5*item['visible']+.5*item['thermal']
            optimizer.zero_grad(set_to_none=True)
            value=loss(model,image,item['boxes'])
            if not torch.isfinite(value):raise RuntimeError('Non-finite detector calibration loss')
            value.backward();torch.nn.utils.clip_grad_norm_(model.model[-1].parameters(),5.);optimizer.step()
            losses.append(float(value.detach()))
            if step%20==0:print(f'Detector epoch {epoch}/{epochs}: {step}/100 loss={np.mean(losses):.3f}',flush=True)
        score=metrics(validation_records())
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation':score})
        if score['AP50']>best_ap:
            best_ap=score['AP50'];best_state=copy.deepcopy(model.state_dict());selected=epoch
        print(f'Detector validation epoch {epoch}: AP50={score["AP50"]:.4f}, selected={selected}',flush=True)
        torch.save({'state':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history},RUN/'detector_last.pt')
    model.load_state_dict(best_state);model.eval().requires_grad_(False)
    torch.save(best_state,RUN/'domain_detector.pt')
    original=make_head()
    records=[]
    for row in validation:
        item=cached(row)
        with torch.no_grad():image=blend(original,item)
        records.append({'id':row['id'],'ground_truth':item['boxes'].tolist(),'prediction':predict(model,image)})
    options=[]
    for threshold in np.arange(.05,.76,.05):
        score=metrics(records,round(float(threshold),2))
        precision,recall=score['precision'],score['recall']
        score['F1']=2*precision*recall/max(precision+recall,1e-12)
        options.append(score)
    chosen=max(options,key=lambda r:(r['F1'],r['score_threshold']))
    operating={'threshold':chosen['score_threshold'],'selection':'Maximum F1 on validation, original fusion only, shared by all methods',
               'validation_metrics':chosen,'candidates':options}
    (RUN/'operating_point.json').write_text(json.dumps(operating,indent=2),encoding='utf-8')
    (RUN/'detector_training.json').write_text(json.dumps({'epochs':epochs,'selected_epoch':selected,'history':history,
        'adapted_parameters':adapted_parameters,
        'train_ids':[r['id'] for r in training],'validation_ids':[r['id'] for r in validation],
        'test_used':False,'inputs':'Seeded visible / thermal / fixed fusion mix; same 100 annotated training pairs',
        'scope':'COCO-pretrained YOLO detection head adapted; backbone and BN buffers fixed. Entire detector frozen for subsequent fusion training.'},indent=2),encoding='utf-8')
    print(f'Detector ready, selected epoch={selected}, shared threshold={chosen["score_threshold"]:.3f}',flush=True)

if __name__=='__main__':prepare()
