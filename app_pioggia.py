"""Annotated pedestrian experiment: fixed, pretrained and fine-tuned fusion."""
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import streamlit as st
import streamlit.components.v1 as components
import torch
from weatherfusion.rain_detection import ROOT,RUN,CACHE,rows,cached,make_head,blend,iou

st.set_page_config(page_title='WeatherFusion · pedestrians in the rain',page_icon='🌧️',layout='wide')
st.title('Detecting pedestrians in the rain')

report_path=RUN/'results.json'
if not report_path.exists():
    st.info('Preparation or fine-tuning is in progress. Results will appear after all 50 test images have been evaluated.')
    count=sum((CACHE/f"{row['id']}.pt").exists() for row in rows()) if CACHE.exists() else 0
    st.progress(min(count/170,1.),text=f'Cached features prepared: {count}/170')
    log=RUN/'progress.log'
    if log.exists():st.code('\n'.join(log.read_text(encoding='utf-8').splitlines()[-12:]),language=None)
    if st.button('Refresh status'):st.rerun()
    st.stop()

report=json.loads(report_path.read_text(encoding='utf-8'))
if 'evaluated_epoch' not in report['metadata']:
    st.info('Training is complete. The comparison using the adapted weights is being finalised.')
    if st.button('Refresh results'):st.rerun()
    st.stop()
labels={'media_fissa':'Fixed average','pretrained':'Pretrained fusion','finetuned':'Fine-tuned fusion'}
summary=[]
for key,label in labels.items():
    value=report['test'][key]['summary']
    summary.append({'Method':label,'Detected pedestrians':value['true_positives'],'Missed pedestrians':value['missed'],
                    'False positives':value['false_positives'],'Recall (%)':round(value['recall']*100,2),
                    'Precision (%)':round(value['precision']*100,2),'AP50 (%)':round(value['AP50']*100,2)})

before=report['test']['pretrained']['summary'];after=report['test']['finetuned']['summary']

def array(tensor):return tensor.detach().clamp(0,1).permute(1,2,0).numpy()

@st.cache_data
def images_for(identifier,modified):
    row=next(r for r in rows() if r['id']==identifier)
    item=cached(row)
    original=make_head();adapted=make_head(RUN/'evaluated_head.pt')
    torch.set_num_threads(2)
    with torch.no_grad():
        a=blend(original,item);b=blend(adapted,item)
    outputs={'media_fissa':array(.5*item['visible']+.5*item['thermal']),
             'pretrained':array(a),'finetuned':array(b)}
    return array(item['visible']),array(item['thermal']).squeeze(),item['boxes'].tolist(),outputs

def overlay(image,truth,prediction,threshold):
    canvas=Image.fromarray((np.clip(image,0,1)*255).astype('uint8'))
    draw=ImageDraw.Draw(canvas)
    for gt in truth:draw.rectangle(gt,outline='#00aaff',width=1)
    matched=set()
    for score,box in sorted(zip(prediction['scores'],prediction['boxes']),reverse=True):
        if score<threshold:continue
        candidates=[(iou(box,gt),i) for i,gt in enumerate(truth) if i not in matched]
        overlap,index=max(candidates,default=(0,-1))
        correct=overlap>=.5
        if correct:matched.add(index)
        color='#00ee55' if correct else '#ff8800'
        draw.rectangle(box,outline=color,width=2)
        draw.text((box[0],max(0,box[1]-12)),f'{score:.2f}',fill=color,stroke_width=1,stroke_fill='black')
    return canvas

st.subheader('Explore the 50 test scenes')
identifiers=sorted(r['id'] for r in rows() if r['split']=='test')
selected=st.selectbox('Test scene',identifiers)
vi,ir,gt,outputs=images_for(selected,(RUN/'evaluated_head.pt').stat().st_mtime_ns)
st.markdown('<style>[data-testid="stImage"] { margin-inline:auto; }</style>',unsafe_allow_html=True)
scene,controls=st.columns([5,1.35],gap='large')
with controls:
    vertical_threshold=components.declare_component('vertical_threshold',path=str(ROOT/'components/vertical_threshold'))
    current_threshold=st.session_state.get('display_threshold',float(after['score_threshold']))
    threshold=vertical_threshold(value=current_threshold,default=current_threshold,key='vertical_threshold')
    st.session_state['display_threshold']=float(threshold)
    st.markdown('<div style="height:24px" aria-hidden="true"></div>',unsafe_allow_html=True)
    st.markdown('**Legend**')
    st.markdown('''
- <span style="color:#007bb8;font-weight:700">Blue</span>: ground truth pedestrian boxes.
- <span style="color:#16803c;font-weight:700">Green</span>: correct detections (IoU ≥ 0.50).
- <span style="color:#b85c00;font-weight:700">Orange</span>: unmatched predictions.
''',unsafe_allow_html=True)
with scene:
    _,visible_column,thermal_column,_=st.columns([.3,1,1,.3])
    visible_column.image(vi,caption='Visible image with simulated rain',width=320)
    thermal_column.image(ir,caption='Thermal infrared image',width=320,clamp=True)
    columns=st.columns(3)
    for column,(key,label) in zip(columns,labels.items()):
        record=next(r for r in report['test'][key]['records'] if r['id']==selected)
        column.image(overlay(outputs[key],gt,record['prediction'],threshold),caption=label,width=320)
st.divider()
st.subheader('How detection performance changes')
before=report['test']['pretrained']['summary'];after=report['test']['finetuned']['summary']
columns=st.columns(3)
columns[0].metric('Pedestrians detected after fine-tuning',after['true_positives'],f"{after['true_positives']-before['true_positives']:+d} vs before fine-tuning")
columns[1].metric('Recall',f"{after['recall']*100:.1f}%",f"{(after['recall']-before['recall'])*100:+.1f} percentage points")
columns[2].metric('False positives',after['false_positives'],f"{after['false_positives']-before['false_positives']:+d} vs before fine-tuning",delta_color='inverse')
st.caption(f"{after['ground_truth_count']} annotated pedestrian boxes across {after['images']} test pairs. One-to-one matching with IoU ≥ 0.50. Counts, precision and recall use the shared threshold of {after['score_threshold']:.2f}, selected on validation; AP50 integrates thresholds over scores ≥ 0.05.")
st.write('Recall measures how many actual pedestrians were detected. Precision measures how many predictions are correct. Both matter: producing more boxes can also increase false alarms.')
st.caption(f"The comparison evaluates the model after the planned {report['metadata']['evaluated_epoch']} epochs. Validation recommends epoch {report['metadata']['recommended_epoch']}; epoch zero means the initial version. The chart comparing all three methods is on slide 12 of the presentation.")
st.download_button('Download results CSV',pd.DataFrame(summary).to_csv(index=False).encode(),'rain_results.csv','text/csv')
st.download_button('Download full JSON report',report_path.read_bytes(),'rain_results.json','application/json')
if 'system_before_any_adaptation' in report:
    with st.expander('How much does preparing the shared detector help?'):
        raw=report['system_before_any_adaptation']['summary']
        st.write(f"With the same original fusion: unadapted COCO detector → {raw['true_positives']} detected pedestrians, AP50 {raw['AP50']*100:.1f}%; detector adapted on the 100 training pairs → {before['true_positives']} detected pedestrians, AP50 {before['AP50']*100:.1f}%.")
        st.caption('This comparison changes the detector to measure the effect of its adaptation. The comparison of the three fusion methods keeps the detector fixed to isolate the contribution of fusion.')
