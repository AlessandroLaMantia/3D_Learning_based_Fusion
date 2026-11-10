import copy
import numpy as np
import torch
import json
from pathlib import Path
import pytest
from weatherfusion.rain_detection import (rows,metrics,rain,blend,detector,detection_loss,ROOT)

def test_annotated_subset_and_scene_separation():
    data=rows()
    assert {s:sum(r['split']==s for r in data) for s in ('train','validation','test')}=={'train':100,'validation':20,'test':50}
    assert sum(len(r['boxes']) for r in data)>170
    for row in data:
        assert row['boxes'] and all(0<=b[0]<b[2]<=row['width'] and 0<=b[1]<b[3]<=row['height'] for b in row['boxes'])

def test_one_to_one_matching_rejects_duplicate_boxes_and_keeps_false_negatives():
    record={'id':'a','ground_truth':[[0,0,10,10],[20,20,30,30]],
            'prediction':{'boxes':[[0,0,10,10],[0,0,10,10],[40,40,50,50]],'scores':[.9,.8,.7]}}
    score=metrics([record])
    assert score['true_positives']==1 and score['false_positives']==2 and score['missed']==1
    assert score['recall']==.5 and score['precision']==1/3
    assert score['AP50']==.5

def test_ap_uses_score_ranking_and_iou():
    record={'id':'b','ground_truth':[[0,0,10,10]],'prediction':{'boxes':[[20,20,30,30],[0,0,10,10]],'scores':[.9,.2]}}
    score=metrics([record])
    assert score['AP50']==.5
    assert score['true_positives']==0 and score['missed']==1

def test_simulated_rain_is_seeded_and_bounded():
    image=np.full((64,80,3),.4,dtype=np.float32)
    a=rain(image,.7,17)
    assert np.array_equal(a,rain(image,.7,17))
    assert not np.array_equal(a,rain(image,.7,18))
    assert a.min()>=0 and a.max()<=1

def test_supervised_gradient_updates_fusion_but_keeps_detector_identical():
    torch.set_num_threads(2)
    model=detector()
    state=torch.load(ROOT/'pretrained/amgfuse_haze.pth',map_location='cpu',weights_only=True)['AWF']
    head=torch.nn.Conv2d(96,3,3,padding=1,bias=False)
    head.load_state_dict({'weight':state['module.output_f.weight']},strict=True)
    del state
    item={'features':torch.randn(1,96,32,40),
          'visible_low':torch.rand(1,3,32,40),'thermal_low':torch.rand(1,1,32,40),
          'visible':torch.rand(3,128,160),'thermal':torch.rand(1,128,160)}
    before={k:v.clone() for k,v in model.state_dict().items()}
    initial=head.weight.detach().clone()
    optimizer=torch.optim.Adam(head.parameters(),lr=1e-4)
    loss=detection_loss(model,blend(head,item),torch.tensor([[40.,20.,80.,110.]]))
    loss.backward()
    assert head.weight.grad.abs().sum()>0
    optimizer.step()
    assert not torch.equal(initial,head.weight)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())

def test_final_comparison_uses_same_truth_and_frozen_common_detector():
    from weatherfusion.rain_detection import RUN
    from weatherfusion.core import sha
    path=RUN/'results.json'
    if not path.exists():pytest.skip('Final run is not complete yet')
    report=json.loads(path.read_text(encoding='utf-8'));meta=report['metadata']
    assert meta['domain_detector_sha256']==sha(RUN/'domain_detector.pt')
    assert meta['evaluated_epoch']==meta['epochs']
    original=torch.load(RUN/'original_head.pt',weights_only=True)
    adapted=torch.load(RUN/'evaluated_head.pt',weights_only=True)
    assert any(not torch.equal(value,adapted[name]) for name,value in original.items())
    assert set(meta['test_ids']).isdisjoint(meta['train_ids']+meta['validation_ids'])
    preparation=meta['common_detector_preparation']
    assert preparation['train_ids']==meta['train_ids']
    assert preparation['test_used'] is False
    truths=[]
    for group in report['test'].values():
        truths.append({r['id']:r['ground_truth'] for r in group['records']})
        actual=metrics(group['records'],group['summary']['score_threshold'])
        assert abs(actual['AP50']-group['summary']['AP50'])<1e-12
    assert truths[0]==truths[1]==truths[2]
