"""Package the current rain project; include cached test features for an instant demo."""
from pathlib import Path
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
files=[]
for name in ['README.md','Spiegazione_slides.txt','requirements-meteo.txt','requirements-meteo-lock.txt','pytest.ini','AVVIA_DEMO.cmd','app_pioggia.py','train_rain_detection.py','prepare_rain_detector.py']:
    files.append(ROOT/name)
for directory in ['weatherfusion','presentazione','.streamlit']:
    files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for name in ['VERIFICA_PIOGGIA.md']:
    files.append(ROOT/'docs'/name)
for name in ['amgfuse_haze.pth','yolov8n.pt','YOLO_LICENSE.txt','rain_sources.json']:
    files.append(ROOT/'pretrained'/name)
for name in ['download_llvip_subset.py','build_rain_presentation.py','slide_design.py','package_rain_project.py','write_rain_verification.py']:
    files.append(ROOT/'tools'/name)
for name in ['test_rain_detection.py','test_app_pioggia.py']:
    files.append(ROOT/'tests'/name)
run=ROOT/'runs/pioggia_llvip_yolo'
files.extend(p for p in run.iterdir() if p.is_file() and p.name not in ('last.pt','detector_last.pt'))
manifest=json.loads((ROOT/'data/llvip_subset/manifest.json').read_text(encoding='utf-8'))
for row in manifest:
    files.extend(ROOT/row[key] for key in ('visible','thermal','annotation'))
files.extend(p for p in (ROOT/'data/llvip_subset').iterdir() if p.is_file() and p.suffix in ('.json','.md'))
files.append(run/'cache/provenance.json')
files.extend(run/'cache'/f"{r['id']}.pt" for r in manifest if r['split']=='test')
target=ROOT/'WeatherFusion_Pioggia.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
    for path in sorted(set(files)):
        if not path.exists():raise FileNotFoundError(path)
        archive.write(path,path.relative_to(ROOT))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
print(json.dumps({'archive':target.name,'files':len(set(files)),'megabytes':round(target.stat().st_size/1e6,1),'cache':'test only; training cache can be rebuilt'}),flush=True)
