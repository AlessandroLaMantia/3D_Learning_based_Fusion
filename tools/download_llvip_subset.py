"""Read a remote ZIP by byte ranges and fetch only a reproducible LLVIP subset."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import struct
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/llvip_subset'
URL='https://huggingface.co/datasets/McCheng/LLVIP/resolve/e50976a/LLVIP.zip'
SIZE=4003858172

def request_bytes(start,end):
    for attempt in range(5):
        try:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end-1}'})
            with urllib.request.urlopen(request,timeout=90) as response:
                if response.status!=206:raise ValueError('Server does not support ranges')
                data=response.read()
            if len(data)!=end-start:raise ValueError('Wrong byte range')
            return data
        except Exception:
            if attempt==4:raise

class RemoteZip(io.RawIOBase):
    def __init__(self):self.position=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else SIZE+offset
        return self.position
    def read(self,n=-1):
        end=SIZE if n<0 else min(SIZE,self.position+n)
        if end<=self.position:return b''
        block=request_bytes(self.position,end);self.position=end
        return block

def download_member(info):
    destination=OUT/info.filename
    if not destination.resolve().is_relative_to(OUT.resolve()):raise ValueError('Unsafe path')
    if destination.exists() and destination.stat().st_size==info.file_size:
        data=destination.read_bytes()
        if zlib.crc32(data)&0xffffffff==info.CRC:return destination
    # One request for local header, variable name/extra and the compressed member.
    length=30+len(info.filename.encode('utf-8'))+len(info.extra)+info.compress_size+4096
    block=request_bytes(info.header_offset,min(SIZE,info.header_offset+length))
    values=struct.unpack('<4s5H3I2H',block[:30])
    if values[0]!=b'PK\x03\x04':raise ValueError('Invalid local ZIP header')
    start=30+values[-2]+values[-1]
    compressed=block[start:start+info.compress_size]
    data=zlib.decompress(compressed,-15) if info.compress_type==zipfile.ZIP_DEFLATED else compressed
    if len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:raise ValueError('Member CRC mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(data)
    return destination

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(RemoteZip()) as archive:
        infos={i.filename:i for i in archive.infolist()}
    visible=sorted(name for name in infos if '/visible/' in '/'+name and name.endswith('.jpg'))
    print('Archive images',len(visible),'sample',visible[:3],flush=True)
    rng=random.Random(17)
    train=[p for p in visible if '/train/' in p]
    test=[p for p in visible if '/test/' in p]
    # Keep scene identifiers separate between train and validation.
    validation_groups={'16','17','18'}
    train_pool=[p for p in train if Path(p).stem[:2] not in validation_groups]
    val_pool=[p for p in train if Path(p).stem[:2] in validation_groups]
    if not val_pool:raise ValueError('Cannot construct scene-separated validation')
    selected=[(p,'train') for p in rng.sample(train_pool,100)]+[(p,'validation') for p in rng.sample(val_pool,20)]
    if '--fresh-test' in sys.argv:
        old=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        excluded={r['id'] for r in old if r['split']=='test'}
        excluded_file=OUT/'initial_trial_test_ids.json'
        if excluded_file.exists():
            excluded.update(json.loads(excluded_file.read_text(encoding='utf-8')))
        excluded_file.write_text(json.dumps(sorted(excluded),indent=2),encoding='utf-8')
        test=[p for p in test if Path(p).stem not in excluded]
        selected += [(p,'test') for p in random.Random(2028).sample(test,50)]
    else:
        # Reproduce the final test split independently of removed diagnostic runs.
        excluded_file=OUT/'initial_trial_test_ids.json'
        if not excluded_file.exists():
            raise FileNotFoundError('Keep initial_trial_test_ids.json to reproduce the final split')
        excluded=set(json.loads(excluded_file.read_text(encoding='utf-8')))
        selected += [(p,'test') for p in random.Random(2028).sample([p for p in test if Path(p).stem not in excluded],50)]
    entries=[];needed={}
    for visible_name,split in selected:
        stem=Path(visible_name).stem
        thermal_name=visible_name.replace('/visible/','/infrared/')
        annotation=next((p for p in infos if Path(p).name==stem+'.xml'),None)
        if thermal_name not in infos or annotation is None:raise ValueError(f'Missing paired annotation: {stem}')
        entry={'id':stem,'split':split,'scene':stem[:2],
               'visible':str((OUT/visible_name).relative_to(ROOT)),
               'thermal':str((OUT/thermal_name).relative_to(ROOT)),
               'annotation':str((OUT/annotation).relative_to(ROOT))}
        entries.append(entry)
        for name in (visible_name,thermal_name,annotation):needed[name]=infos[name]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(download_member,info) for info in needed.values()]
        for index,future in enumerate(concurrent.futures.as_completed(futures),1):
            future.result()
            if index%30==0:print(f'Downloaded {index}/{len(futures)} members',flush=True)
    for entry in entries:
        root=ET.parse(ROOT/entry['annotation']).getroot()
        width=int(root.findtext('size/width'));height=int(root.findtext('size/height'))
        boxes=[]
        for obj in root.findall('object'):
            box=obj.find('bndbox')
            # VOC 1-based inclusive -> continuous 0-based xyxy, clipped below.
            coords=[float(box.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
            coords[0]-=1;coords[1]-=1
            coords=[max(0,min(v,width if i%2==0 else height)) for i,v in enumerate(coords)]
            if coords[2]>coords[0] and coords[3]>coords[1]:boxes.append(coords)
        entry.update({'width':width,'height':height,'boxes':boxes,'class':'person'})
        entry['sha256']={key:hashlib.sha256((ROOT/entry[key]).read_bytes()).hexdigest() for key in ('visible','thermal','annotation')}
    license_url='https://raw.githubusercontent.com/bupt-ai-cz/LLVIP/main/Term%20of%20Use%20and%20License.md'
    (OUT/'LICENSE.md').write_bytes(urllib.request.urlopen(license_url,timeout=60).read())
    (OUT/'manifest.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    (OUT/'source.json').write_text(json.dumps({'source':URL,'official':'https://github.com/bupt-ai-cz/LLVIP',
        'download':'Only 170 paired samples and XML labels, remote ZIP member CRC verified',
        'seed':17,'test_seed':2028,'excluded_test_ids':sorted(excluded),'counts':{'train':100,'validation':20,'test':50},'validation_scene_ids':sorted(validation_groups),
        'weather':'Rain is simulated; original dataset is low-light, not recorded rain.'},indent=2),encoding='utf-8')
    print('Subset ready: 100 train / 20 validation / 50 test pairs',flush=True)

if __name__=='__main__':main()
