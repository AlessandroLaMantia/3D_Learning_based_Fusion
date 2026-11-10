from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

def test_rain_app_displays_real_held_out_results():
    path=ROOT/'runs/pioggia_llvip_yolo/results.json'
    if not path.exists():pytest.skip('Experiment is still running')
    app=AppTest.from_file(ROOT/'app_pioggia.py').run(timeout=60)
    assert not app.exception
    assert len(app.tabs)==0
    assert app.title[0].value=='Detecting pedestrians in the rain'
    assert [section.value for section in app.subheader]==['Explore the 50 test scenes','How detection performance changes']
    assert app.selectbox[0].label=='Test scene'
    assert [metric.label for metric in app.metric]==['Pedestrians detected after fine-tuning','Recall','False positives']
    assert len(app.dataframe)==0
    assert len(app.get('arrow_vega_lite_chart'))==0
    report=json.loads(path.read_text(encoding='utf-8'))
    assert int(app.metric[0].value)==report['test']['finetuned']['summary']['true_positives']
    assert len(app.selectbox[0].options)==50
