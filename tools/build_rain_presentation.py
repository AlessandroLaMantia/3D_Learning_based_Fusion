"""Build the annotated rain experiment deck, oral guide and current README."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from weatherfusion.rain_detection import ROOT,RUN,rows,cached,make_head,blend,iou
from tools.slide_design import fit,width as text_width,write_pdf,INK,TEAL,GRAY,WHITE,GOLD

OUT=ROOT/'presentazione';FIG=OUT/'figure'
FIG.mkdir(parents=True,exist_ok=True)
report=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
data=rows();meta=report['metadata']
summary={key:value['summary'] for key,value in report['test'].items()}
before=summary['pretrained'];after=summary['finetuned']
raw_system=report['system_before_any_adaptation']['summary']
score_threshold=after['score_threshold']
extra=after['true_positives']-before['true_positives']
delta=(after['recall']-before['recall'])*100
torch.set_num_threads(2)
original=make_head();tuned=make_head(RUN/'evaluated_head.pt')

def save(name,t):
    a=t.detach().clamp(0,1).numpy()
    if a.ndim==3:a=a.transpose(1,2,0)
    if a.ndim==3 and a.shape[2]==1:a=a[:,:,0]
    Image.fromarray((a*255).astype('uint8')).convert('RGB').save(FIG/f'{name}.png')

# A fixed, label-independent example; the app exposes all 50 test scenes.
example=sorted((r for r in data if r['split']=='test'),key=lambda r:r['id'])[0]
item=cached(example)
with torch.no_grad():
    a,alpha_a=blend(original,item,True);b,alpha_b=blend(tuned,item,True)
outputs={'media_fissa':.5*item['visible']+.5*item['thermal'],'pretrained':a,'finetuned':b}
save('visible',item['visible']);save('thermal',item['thermal'])
with Image.open(ROOT/example['visible']) as source:
    source.convert('RGB').resize((320,256)).save(FIG/'visible_original.png')
save('alpha_before',alpha_a[0]);save('alpha_after',alpha_b[0])
for key,output in outputs.items():
    save(key,output)
    canvas=Image.open(FIG/f'{key}.png');draw=ImageDraw.Draw(canvas)
    for box in item['boxes'].tolist():draw.rectangle(box,outline='#00aaff',width=1)
    prediction=next(r for r in report['test'][key]['records'] if r['id']==example['id'])['prediction']
    matched=set()
    for score,box in zip(prediction['scores'],prediction['boxes']):
        if score>=score_threshold:
            candidates=[(iou(box,gt),index) for index,gt in enumerate(item['boxes'].tolist()) if index not in matched]
            overlap,index=max(candidates,default=(0,-1))
            correct=overlap>=.5
            if correct:matched.add(index)
            color='#00ef55' if correct else '#ff8800'
            draw.rectangle(box,outline=color,width=2)
            draw.text((box[0],max(0,box[1]-12)),f'{score:.2f}',fill=color,stroke_width=1,stroke_fill='black')
    canvas.save(FIG/f'{key}_boxes.png')

fig,axes=plt.subplots(1,2,figsize=(10,4),facecolor='#edf3f8')
names=['Media fissa','Prima','Dopo'];colors=['#94a9b9',TEAL,GOLD]
for ax,key,title,scale in [(axes[0],'recall','Pedoni presenti trovati (%)',100),(axes[1],'false_positives','Falsi positivi',1)]:
    values=[summary[k][key]*scale for k in ['media_fissa','pretrained','finetuned']]
    ax.set_facecolor('#edf3f8');ax.bar(names,values,color=colors,width=.6)
    for i,v in enumerate(values):ax.text(i,v+max(values+[1])*.03,f'{v:.1f}' if scale==100 else str(int(v)),ha='center',color=INK)
    ax.set_ylim(0,max(values+[1])*1.22);ax.set_title(title,color=INK);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(FIG/'metrics.png',dpi=140);plt.close(fig)

slides=[]
def text(s,x,y,w,h,value,size=23,color=INK,bold=False,align='left'):
    raw=str(value);value,size=fit(raw,w,h,size,bold)
    while max(text_width(line,size,bold) for line in value.split('\n'))>w*72-4:
        size-=1
        if size<12:raise ValueError(raw)
        value,size=fit(raw,w,h,size,bold)
    s['elements'].append(dict(kind='text',x=x,y=y,w=w,h=h,text=value,size=size,color=color,bold=bold,align=align))
def rect(s,x,y,w,h,color=WHITE):s['elements'].append(dict(kind='rect',x=x,y=y,w=w,h=h,color=color))
def image(s,x,y,w,h,name):s['elements'].append(dict(kind='image',x=x,y=y,w=w,h=h,path=str(FIG/name)))
def card(s,x,y,w,h,label,body):
    rect(s,x,y,w,h);text(s,x+.2,y+.18,w-.4,.6,label,22,TEAL,True);text(s,x+.2,y+.95,w-.4,h-1.1,body,22)
def slide(title,subtitle,note,layout):
    s={'title':title,'subtitle':subtitle,'note':note,'layout':layout,'elements':[]};slides.append(s)
    text(s,.75,.9,11.8,.85,title,31,INK,True);text(s,.75,1.85,11.8,.6,subtitle,17,GRAY)
    text(s,.75,7.05,11.8,.23,f'Alessandro La Mantia                                             {len(slides):02d}',10,GRAY)
    return s

s=slide('Riconoscimento pedoni con 3 tipi di data fusion','',
 'L’obiettivo è capire se adattare la fusione tra visibile e termico aiuta a trovare più pedoni quando il visibile è degradato dalla pioggia. Usiamo immagini reali e pioggia simulata. Il confronto considera media fissa, fusione guidata da AMG-Fuse prima dell’adattamento e la stessa fusione dopo. Il detector resta identico. Questa versione sostituisce il piccolo esperimento precedente con pseudo-riferimenti.','cover')
s['elements'].clear()
text(s,1.1,1.6,11.133,.45,'LEARNING-BASED DATA FUSION',15,TEAL,True,align='center')
text(s,1.1,2.65,11.133,1.65,'Riconoscimento pedoni\ncon 3 tipi di data fusion',38,INK,True,align='center')
text(s,1.1,4.9,11.133,.65,'Alessandro La Mantia',25,TEAL,False,align='center')

s=slide('Obiettivo: confrontare tre fusioni','',
 'Confrontiamo tre modi di combinare le stesse immagini visibili e termiche con pioggia simulata: media fissa, rete preaddestrata e stessa rete dopo fine-tuning. Valutiamo tutti i metodi sulle stesse 50 immagini di test, con lo stesso detector congelato. Il confronto misura pedoni trovati e falsi positivi: il miglioramento va verificato, non dato per scontato.','objective-comparison')
for x,label in [(0.8,'no learning-based\ndata fusion'),(4.8,'learning-based\ndata fusion'),(8.8,'fine tuned\nlearning-based\ndata fusion')]:
    rect(s,x,2.7,3.8,2.5)
    line_count=label.count('\n')+1
    text(s,x+.2,2.7+(2.5-line_count*24*1.25/72)/2,3.4,1.35,label,24,TEAL,True,align='center')
objective_parts=[('Capire se la rete e il fine-tuning aiutano a ',False),('riconoscere più pedoni',True)]
objective_x=1+(11.3-sum(text_width(value,18,bold) for value,bold in objective_parts)/72)/2
for value,bold in objective_parts:
    part_width=text_width(value,18,bold)/72
    text(s,objective_x,5.6,part_width+.08,.45,value,18,INK,bold)
    objective_x+=part_width
text(s,1,5.6+.45,11.3,.6,'rispetto alla no learning-based data fusion',18,INK,False,align='center')

s=slide('Dataset: AWMM-100K e due tipi di immagini','',
 'Usiamo LLVIP, un dataset di coppie visibili e termiche della stessa scena, soprattutto in scarsa illuminazione. Le immagini mostrate sono la coppia originale della stessa scena: a sinistra la normale telecamera a colori, a destra la termocamera a infrarossi. Sono già allineate e accompagnate da riquadri annotati dei pedoni. Dal dataset completo di 15.488 coppie ne usiamo 170: 100 per aggiornare i pesi, 20 per la validation e 50 per il test. Nell’esperimento aggiungiamo pioggia artificiale soltanto al visibile; il termico resta invariato. LLVIP non è un dataset di pioggia reale.','dataset-inputs')
# Emphasize the dataset size and split while retaining the two input images.
for element in s['elements']:
    if element['kind']=='text' and element['y']==1.85:
        element['text'],element['size']=fit(s['subtitle'],11.8,.6,24)
image(s,.9,2.5,5.45,3.1,'visible_original.png');image(s,6.95,2.5,5.45,3.1,'thermal.png')
text(s,1,5.75,5.3,.5,'Visibile: telecamera a colori',22,TEAL,True)
text(s,7,5.75,5.3,.5,'Termico: telecamera a infrarossi',22,TEAL,True)
text(s,1,6.45,11.3,.4,'AWMM-100K: pretraining. Immagini mostrate: LLVIP, usato nel nostro fine-tuning.',16,GRAY,False,align='center')

s=slide('Pioggia artificiale nel nostro esperimento','Nel nostro esperimento aggiungiamo pioggia artificiale solo alle immagini della telecamera a colori.',
 'Mostriamo la stessa immagine a colori della slide precedente dopo l’applicazione della pioggia artificiale usata nell’esperimento. Non è una fotografia scattata sotto pioggia reale. Il termico della stessa scena rimane invariato. L’obiettivo è confrontare i tre tipi di data fusion — media fissa, rete preaddestrata e rete dopo fine-tuning — usando sempre questa coppia di ingressi: immagine a colori con pioggia simulata e immagine della telecamera a infrarossi. Il detector resta identico nei tre confronti.','rain-input')
for element in s['elements']:
    if element['kind']=='text' and element['y']==1.85:
        element['text']=s['subtitle']
        element['size']=14
image(s,4.16,2.65,5,3.1,'visible.png')
text(s,1,5.95,11.3,.4,'Obiettivo: confrontare i 3 tipi di data fusion tra',20,INK,False,align='center')
text(s,1,6.4,11.3,.4,'immagine con la pioggia e immagine della telecamera a infrarossi',20,INK,True,align='center')

s=slide('Il percorso completo è questo:','',
 'Le immagini visibile con pioggia e termica entrano nel processo di fusione. Il blocco fusion process (1/2/3 method) rappresenta uno dei tre metodi confrontati: media fissa, fusione preaddestrata o fusione dopo fine-tuning. Ogni metodo produce una sola immagine fusa, che viene analizzata dallo stesso YOLO per rilevare i pedoni.','complete-pipeline')
for x,y,w,h,label in [
    (.75,3.15,2.65,1.2,'Immagine visibile\ncon pioggia'),
    (.75,4.75,2.65,1.2,'Immagine termica'),
    (4.05,3.95,2.65,1.2,'fusion process\n(1/2/3 method)'),
    (7.15,3.95,1.65,1.2,'Immagine\nfusa'),
    (9.25,3.95,1.1,1.2,'YOLO'),
    (10.8,3.95,1.8,1.2,'Pedoni\nrilevati')]:
    rect(s,x,y,w,h)
    text(s,x+.08,y+(.43 if label=='YOLO' else .28),w-.16,.8,label,17,TEAL,True,align='center')
for x,y,w,h in [(3.4,3.74,.3,.025),(3.4,5.34,.3,.025),(3.68,3.74,.025,1.625),(3.68,4.54,.1,.025)]:
    rect(s,x,y,w,h,TEAL)
for x in [3.73,6.72,8.82,10.37]:
    text(s,x,4.29,.4,.55,'→',22,TEAL,True,align='center')

s=slide('1) No learning-based data fusion','',
 'Questo metodo combina immagini già allineate senza imparare dai dati. Per ogni posizione prende il valore del visibile e quello del termico, entrambi normalizzati, e calcola la loro media: F = 0,5 V + 0,5 T. Il termico ha un solo canale, replicato per combinarlo con i tre canali a colori. L’esempio 100 e 200 che danno 150 usa valori illustrativi su una scala comune: non sono temperature in gradi. I pesi restano 50% e 50% in ogni zona e per ogni scena, anche quando un ingresso è degradato. Non ci sono pesi da addestrare nella fusione. L’immagine fusa va poi allo stesso detector YOLO usato dagli altri due metodi. “No learning-based” riguarda la fusione, non il detector.','fixed-average')
text(s,.75,2.05,11.8,.6,'Regola: media dei pixel corrispondenti delle due immagini',23,INK,True,align='center')
rect(s,1,2.95,11.3,1.8,'#dcebf0')
text(s,1.45,3.55,4.1,.65,'Immagine fusa =',28,TEAL,True,align='center')
text(s,5.75,3.2,5.8,.6,'Visibile + Termico',28,TEAL,True,align='center')
rect(s,5.95,3.86,5.4,.025,TEAL)
text(s,5.75,4.05,5.8,.6,'2',28,TEAL,True,align='center')
text(s,1,5.15,11.3,1.6,'Ogni pixel riceve metà contributo da\nciascuna immagine.\nI pesi sono uguali ovunque: la fusione\nnon impara.',19,INK,False,align='center')

s=slide('2) Learning-based data fusion','',
 'Usiamo MSAKNet di AMG-Fuse, caricando il checkpoint Haze degli autori, già addestrato per la fusione visibile–infrarosso in presenza di foschia. Non partiamo da pesi casuali e in questo secondo metodo non eseguiamo nostro fine-tuning. La rete contiene convoluzioni, che elaborano strutture locali, e blocchi Transformer con attenzione, che mettono in relazione le caratteristiche. I rami del visibile, del termico e della fusione scambiano informazioni a più scale. Le feature sono mappe numeriche apprese, non etichette di pedoni. Nel progetto, per lavorare sulla CPU, i due ingressi vengono ridotti a 40 × 32 pixel: estraiamo le feature prima della convoluzione finale output_f e applichiamo questo strato originale per ottenere una ricostruzione RGB ridotta. La slide seguente spiega come questa ricostruzione guida la fusione degli ingressi a 320 × 256. È una variante guidata da AMG-Fuse, distinta dalla sua uscita nativa a piena risoluzione.','neural-fusion')
network_prefix='• Rete preaddestrata usata: '
network_prefix_width=text_width(network_prefix,19,False)/72
text(s,1,1.95,network_prefix_width+.15,.55,network_prefix,19,INK)
text(s,1+network_prefix_width+.06,1.95,2,.55,'AMG-Fuse',19,INK,True)
text(s,1,2.4,11.3,.55,'• Dataset utilizzato: AWMM-100K con 1.000 coppie di immagini',19,INK)
intro_prefix='• AMG-Fuse elabora visibile e termico e produce direttamente un’immagine '
intro_width=text_width(intro_prefix,19,False)/72
text(s,1,2.85,intro_width+.2,.55,intro_prefix,19,INK)
text(s,1+intro_width+.18,2.85,.35,.55,'R',19,INK,True)
for x,label,body in [
    (.8,'1. Estrae feature','Individua contorni e contrasti nei due ingressi.'),
    (4.8,'2. Le combina','Unisce le informazioni tramite l’attenzione.'),
    (8.8,'3. Ricostruisce','Produce una prima immagine combinata')]:
    rect(s,x,3.85,3.7,.9)
    text(s,x+.12,4.1,3.46,.5,label,18,TEAL,True,align='center')
    if label=='3. Ricostruisce':
        text(s,x+.12,5.05,3.46,.45,'Produce una prima',17,INK,align='center')
        last_line='immagine combinata'
        advance=text_width(last_line,17,False)/72
        line_x=x+(3.7-advance-.35)/2
        text(s,line_x,5.4,advance+.1,.45,last_line,17)
        text(s,line_x+advance+.08,5.4,.3,.45,'R',17,INK,True)
    else:
        text(s,x+.12,5.05,3.46,1.15,body,17,INK,align='center')
for x in [4.5,8.5]:
    text(s,x,4.06,.3,.55,'→',22,TEAL,True,align='center')

s=slide('Come la rete guida la fusione','Il codice prende R e calcola α dopo, confrontandola con i due ingressi.',
 'La rete non produce direttamente la mappa alfa usata nella nostra combinazione finale. Prima ricostruiamo un risultato R a 40 × 32 tramite output_f. Per ogni posizione calcoliamo la distanza quadratica media sui tre canali fra R e il visibile ridotto, dV, e fra R e il termico ridotto, dT. Il termico viene esteso ai tre canali. Definiamo alfa = sigmoid(4 × (dT − dV)). Se R è più vicino al visibile, dV è minore di dT e alfa supera 0,5; se è più vicino al termico, alfa è inferiore a 0,5. La sigmoid mantiene i coefficienti fra zero e uno. Ingrandiamo la mappa con interpolazione bilineare a 320 × 256 e calcoliamo F = alfa V + (1 − alfa) T, usando gli ingressi alla risoluzione maggiore per conservarne i dettagli. La media fissa corrisponde ad alfa sempre uguale a 0,5; qui alfa dipende dagli ingressi e varia tra le posizioni. Per esempio alfa = 0,8 significa 80% visibile e 20% termico: è un esempio di interpretazione, non un valore misurato. Alfa non è una probabilità di affidabilità e un beneficio per il riconoscimento va misurato con YOLO. Questa procedura è la nostra variante guidata da AMG-Fuse, non la sua uscita nativa a piena risoluzione.','neural-fusion-weights')
for y,value in [
    (2.55,'• R più simile al visibile → α > 0,5 → più contributo del visibile.'),
    (3.1,'• R più simile al termico → α < 0,5 → più contributo del termico.')]:
    text(s,1,y,11.3,.7,value,21,INK)
rect(s,1,4.15,11.3,.75,'#dcebf0')
text(s,1.2,4.33,10.9,.5,'Immagine fusa = α × Visibile + (1 − α) × Termico',25,TEAL,True,align='center')
text(s,1,5.15,11.3,.35,'Media fissa: α = 0,5 ovunque. Qui: il peso cambia in base alla scena e alla posizione.',16,INK,True,align='center')

s=slide('Fine-tuned learning-based data fusion','',
 f'Il fine-tuning parte dai pesi preaddestrati, non da una rete casuale. Usiamo 100 coppie LLVIP con pioggia simulata sul visibile e le vere annotazioni dei pedoni per {meta["epochs"]} epoche. La fusione produce un’immagine e il detector YOLO propone i pedoni. La loss confronta classificazione e posizione con i riquadri veri; la backpropagation attraversa il detector e aggiorna soltanto i {meta["trainable_parameters"]} pesi dell’ultimo strato di AMG-Fuse. Il resto della fusione e tutto il detector restano congelati. Una piccola regolarizzazione limita lo scostamento della mappa da quella iniziale. Le 20 coppie validation raccomandano un checkpoint; confrontiamo comunque i pesi realmente aggiornati dopo tre epoche sulle 50 coppie test. Il detector comune è preparato prima del confronto e resta identico nei tre metodi. La validation raccomanda l’originale: il fine-tuning non è presentato come un miglioramento garantito.','fine-tuning-method')
for y,parts in [
    (1.95,[('• Partiamo dalla stessa AMG-Fuse e adattiamo soltanto l’',False),('ultimo strato',True)]),
    (2.5,[('• ',False),('Dataset LLVIP',True),(': 100 coppie di immagini',False)]),
    (3.05,[('• 3 epoche: ',False),('2.592',True),(' pesi aggiornati',False)])]:
    # Use PDF font metrics so adjacent normal/bold runs do not overlap.
    import io
    from matplotlib.backends.backend_pdf import RendererPdf
    from matplotlib.font_manager import FontProperties
    pdf_metrics=RendererPdf(io.BytesIO(),72,7.5,13.333)
    run_x=1
    for value,bold in parts:
        prop=FontProperties(family='DejaVu Sans',size=21,weight='bold' if bold else 'normal')
        run_width=pdf_metrics.get_text_width_height_descent(value,prop,False)[0]/72
        # text() trims whitespace: retain it through the measured run positions.
        leading=value[:len(value)-len(value.lstrip())]
        leading_width=pdf_metrics.get_text_width_height_descent(leading,prop,False)[0]/72
        text(s,run_x+leading_width,y,run_width+.15,.55,value,21,INK,bold)
        run_x+=run_width
for x,label,body in [
    (.8,'1. Riconoscimento','Immagine fusa → YOLO → riquadri dei pedoni.'),
    (4.8,'2. Errore','Confronto con i riquadri veri presenti nel dataset.'),
    (8.8,'3. Aggiornamento','L’errore modifica solo l’ultimo strato della fusione.')]:
    rect(s,x,4.15,3.7,.9)
    text(s,x+.12,4.4,3.46,.5,label,18,TEAL,True,align='center')
    text(s,x+.12,5.35,3.46,1.05,body,17,INK,align='center')
for x in [4.5,8.5]:
    text(s,x,4.36,.3,.55,'→',22,TEAL,True,align='center')

s=slide('YOLO: dalle immagini fuse ai pedoni',
 'Ogni immagine fusa dei tre metodi entra separatamente nello stesso YOLO.',
 'Le immagini prodotte da media fissa, fusione preaddestrata e fusione dopo fine-tuning vengono analizzate separatamente dallo stesso detector YOLO, mantenendo identici i suoi parametri. YOLO cerca i pedoni e propone rettangoli con un punteggio di confidenza. La soglia è il punteggio minimo per mantenere una rilevazione: a soglia 0,30 una proposta con punteggio 0,75 viene mostrata, una con 0,20 viene nascosta. Alzando la soglia si mostrano meno proposte: si possono eliminare falsi allarmi, ma anche perdere pedoni corretti. Abbassandola si mostrano più proposte: si possono recuperare pedoni, ma anche aggiungere falsi allarmi. Il punteggio non è una probabilità calibrata. Nella demo lo slider modifica solo le box visualizzate, mentre i risultati aggregati sono calcolati alla soglia condivisa 0,30 scelta sulla validation.','yolo-threshold')
rect(s,1,2.65,11.3,.85,'#dcebf0')
text(s,1.2,2.87,10.9,.5,'Immagine fusa → YOLO → riquadri dei pedoni + punteggi',24,TEAL,True,align='center')
text(s,1,3.75,11.3,.55,'Soglia = punteggio minimo per mostrare una rilevazione.',23,INK,True,align='center')
for x,label,body in [
    (1,'Soglia più alta ↑','Meno rettangoli mostrati.\nPuò ridurre i falsi allarmi,\nma anche perdere pedoni veri.'),
    (6.95,'Soglia più bassa ↓','Più rettangoli mostrati.\nPuò recuperare pedoni,\nma anche aggiungere falsi allarmi.')]:
    rect(s,x,4.65,5.35,1.8)
    text(s,x+.2,4.85,4.95,.5,label,22,TEAL,True,align='center')
    text(s,x+.2,5.4,4.95,.95,body,18,INK,False,align='center')

s=slide('Interfaccia','',
 'Ora confrontiamo i tre metodi sulle stesse 50 coppie di test, separate dal training, usando lo stesso YOLO. L’interfaccia locale è disponibile su http://127.0.0.1:64001 dopo aver avviato AVVIA_DEMO.cmd. Mostriamo prima una scena con i due ingressi, le tre fusioni e le rilevazioni, poi i risultati aggregati.','demo-transition')
for y,parts in [
    (2.6,[('Ora confrontiamo i tre metodi sulle stesse ',False),('50',True),(' coppie di immagini',False)]),
    (3.15,[('test (dataset: LLVIP) usando lo ',False),('STESSO YOLO',True)])]:
    measured=[]
    for value,bold in parts:
        prop=FontProperties(family='DejaVu Sans',size=24,weight='bold' if bold else 'normal')
        measured.append((value,bold,prop,pdf_metrics.get_text_width_height_descent(value,prop,False)[0]/72))
    run_x=(13.333-sum(run[3] for run in measured))/2
    for value,bold,prop,run_width in measured:
        leading=value[:len(value)-len(value.lstrip())]
        leading_width=pdf_metrics.get_text_width_height_descent(leading,prop,False)[0]/72
        text(s,run_x+leading_width,y,run_width+.15,.6,value,24,INK,bold)
        run_x+=run_width
rect(s,1.5,4.35,10.3,1.05,'#dcebf0')
text(s,1.7,4.65,9.9,.6,'http://127.0.0.1:64001',28,TEAL,True,align='center')

s=slide('Tre difficoltà e come sono state risolte','Dati annotati, costo sulla CPU e misura corretta del riconoscimento.',
 'La prima difficoltà era il costo del dataset: l’archivio completo pesa circa 4 GB, quindi abbiamo letto l’indice remoto e scaricato un sottoinsieme con XML, verificando i singoli file. L’esperimento finale usa 170 coppie, oltre alle coppie diagnostiche escluse dal test finale. La seconda era il costo della rete sulla CPU: abbiamo congelato e memorizzato le feature, adattando solo l’ultimo strato e conservando i dettagli degli ingressi tramite la mappa di fusione. La terza era il detector generico, poco adatto alle scene notturne e termiche: prepariamo una testa comune sulle 100 coppie, poi la congeliamo per il confronto. Le box vere e un nuovo test separato permettono di misurare recall, precision, AP50 e falsi positivi.','challenges')
for y,label,body in [(2.8,'Dati','Sottoinsieme riproducibile, annotazioni vere, file verificati.'),(4.1,'CPU','Feature riutilizzate, solo ultimo strato aggiornato.'),(5.4,'Detection','Detector preparato prima, poi congelato. Annotazioni e test nuovo.')]:
    text(s,1,y,2.4,.65,label,27,TEAL,True);text(s,3.8,y,8.2,1,body,24)

s=slide('Grazie per l’attenzione','Domande e discussione',
 f"Il messaggio finale è che la fusione può essere adattata usando un obiettivo di riconoscimento concreto, mantenendo fisso il detector. Abbiamo usato 100 coppie per il fine-tuning e 50 per il test, con pioggia simulata. Il cambiamento misurato è {extra:+d} pedoni correttamente trovati e {delta:+.2f} punti di recall. Va interpretato insieme a falsi positivi e AP50. Le fonti e i limiti sono nel rapporto, senza una slide finale di riferimenti.",'thanks')
text(s,1,3,11.2,1.6,'Una fusione che impara\nquanto aiuta il riconoscimento.',38,TEAL,True);text(s,1,5.8,10,.65,'Alessandro La Mantia',27,INK,True)

from tools.performance_slide import insert_performance_slide
insert_performance_slide(slides,OUT)
from tools.closing_slides import update_closing_slides
update_closing_slides(slides)
write_pdf(slides,OUT)
from tools.english_presentation import write_english
write_english(slides,OUT)
(OUT/'slides.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2),encoding='utf-8')
intro="""OBIETTIVO GENERALE
Misurare se il fine-tuning della fusione visibile–termico permette di riconoscere più pedoni in presenza di pioggia simulata. Confrontiamo media fissa, fusione guidata dalla rete preaddestrata e stessa fusione dopo adattamento. Il detector resta identico.

DA COSA PARTIAMO
170 coppie reali LLVIP già allineate, con annotazioni dei pedoni: 100 train, 20 validation e 50 test. Riutilizziamo i pesi ufficiali AMG-Fuse Haze e un detector YOLOv8n preaddestrato. La pioggia è aggiunta al visibile; il termico resta invariato nella simulazione.

PREPARAZIONE COMUNE DEL DETECTOR
Prima del confronto adattiamo la testa di YOLOv8n alle stesse 100 coppie, usando visibile, termico e media fissa. La validation sceglie i pesi. Poi congeliamo l'intero detector: da quel momento resta identico prima e dopo il fine-tuning della fusione. Il termine 'prima' indica quindi prima dell'adattamento della fusione, dopo la preparazione comune del detector. La soglia delle box è scelta sulla validation con la fusione originale e condivisa da tutti i metodi.

COME IMPARA LA RETE
La parte congelata di AMG-Fuse produce feature a 40×32 pixel. Adattiamo solo i 2.592 pesi del suo ultimo strato. Dall'uscita ricaviamo una mappa morbida che combina le immagini originali a 320×256 pixel. Il detector cerca pedoni nell'immagine fusa e confrontiamo le proposte con le box annotate. L'errore di classificazione e posizione attraversa il detector congelato e aggiorna soltanto lo strato di fusione. La validation raccomanda quali pesi conservare; il confronto mostra comunque l'effetto reale delle tre epoche prefissate, anche se peggiora.

ATTENZIONE ALLA DIFFERENZA RISPETTO ALLA VERSIONE PRECEDENTE
Questo esperimento usa vere box per imparare e misurare la detection. Non usa il precedente PSNR verso un'immagine prodotta da un docente. Per funzionare sulla CPU la nuova pipeline è guidata da AMG-Fuse, ma non coincide con la sua uscita nativa a piena risoluzione. 'Senza NN' significa senza NN nella fusione: il detector resta neurale in tutti i confronti.

"""
explanation=f'WEATHERFUSION — SPIEGAZIONE DELLE {len(slides)} SLIDE\nPresentazione attuale: presentazione/WeatherFusion.pdf\n\n'+intro
for i,s in enumerate(slides,1):explanation+=f"SLIDE {i} — {s['title']}\n\n{s['note']}\n\n"
explanation+="INTRODUZIONE PRONTA DA DIRE\n\nIl mio progetto studia se una fusione tra telecamera visibile e termica può essere adattata per riconoscere meglio i pedoni nella pioggia. Parto da una rete già addestrata e da immagini reali annotate; simulo la pioggia sul visibile e aggiorno solo un piccolo strato della fusione. Mantengo fisso il detector e confronto i risultati prima e dopo su 50 immagini tenute fuori dal training. Misuro pedoni trovati, persone mancate e falsi positivi, insieme a recall, precision e AP50.\n"
(ROOT/'Spiegazione_slides.txt').write_text(explanation,encoding='utf-8')
sheet=Image.new('RGB',(1600,225*((len(slides)+3)//4)),'#edf3f8')
for i in range(len(slides)):
    im=Image.open(OUT/f'anteprima_{i+1:02d}.png').convert('RGB');im.thumbnail((400,225));sheet.paste(im,((i%4)*400,(i//4)*225))
sheet.save(OUT/'panoramica.png')
table='| Fusione | Pedoni trovati | Mancati | Falsi positivi | Recall | Precision | AP50 |\n|---|---:|---:|---:|---:|---:|---:|\n'
for key,label in [('media_fissa','Media fissa'),('pretrained','Prima del fine-tuning'),('finetuned','Dopo il fine-tuning')]:
    v=summary[key]
    table+=f"| {label} | {v['true_positives']} | {v['missed']} | {v['false_positives']} | {v['recall']*100:.2f}% | {v['precision']*100:.2f}% | {v['AP50']*100:.2f}% |\n"
readme=f'''# WeatherFusion — pedoni nella pioggia

**Alessandro La Mantia · Learning-based Data Fusion**

Obiettivo: misurare se adattare una fusione visibile–termico aiuta a riconoscere più pedoni sotto pioggia simulata. Il confronto usa vere annotazioni, tenendo identico e congelato il detector.

## Avvio

Apri `AVVIA_DEMO.cmd`, oppure:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app_pioggia.py --server.port 64001
```

Demo: http://127.0.0.1:64001. La prima scheda mostra risultati aggregati; la seconda permette di esplorare tutte le 50 scene con box vere, previsioni e mappe di fusione; la terza spiega l'apprendimento. I risultati e le feature sono già salvati: la demo non ripete il training.

## Dati e fine-tuning realmente eseguiti

Sono state scaricate **170 coppie reali LLVIP** con annotazioni XML dei pedoni: **100 train, 20 validation, 50 test**. Il conteggio delle box si trova nel manifest e nel report. Il dataset completo contiene 15.488 coppie; abbiamo letto l'indice dell'archivio remoto e scaricato solo i file necessari, verificandone CRC e SHA-256. Manifest e provenienza: `data/llvip_subset/`.

Gli ID di scena scelti per train, validation e test sono disgiunti, come verificato dal codice. Non è certificata l'assenza di LLVIP dal pretraining degli autori di AMG-Fuse. La separazione garantita riguarda il nostro fine-tuning.

**Il meteo è simulato:** strisce di pioggia, leggera sfocatura e attenuazione del visibile; il termico resta invariato. Intensità 0,4–0,9 nel train e 0,7 in validation e test. LLVIP è soprattutto un dataset di scene poco illuminate, non un dataset di pioggia reale. Non generalizziamo questi risultati a riprese piovose reali.

Abbiamo eseguito **{meta['epochs']} epoche**, aggiornando **{meta['trainable_parameters']:,} pesi** dello strato finale originale di AMG-Fuse. Il confronto valuta il modello realmente adattato dopo le **{meta['evaluated_epoch']} epoche prefissate**, anche se peggiora. La validation raccomanda l'**epoca {meta['recommended_epoch']}** tramite AP50: zero significa conservare l'originale. La raccomandazione resta distinta dall'effetto sperimentale del fine-tuning. Il test non sceglie il modello da mostrare.

## Come funziona la rete

Il punto di partenza è il checkpoint ufficiale **AMG-Fuse Haze**, già specializzato nella fusione per foschia, trasferito qui alla pioggia simulata. Non viene addestrata una rete casuale. Tutte le feature congelate e i coefficienti iniziali dell'ultimo strato provengono dal checkpoint originale.

Prima del confronto viene preparato un detector comune: la sola testa di YOLOv8n COCO è adattata per sei epoche sulle stesse 100 coppie, alternando visibile, termico e media fissa. La validation sceglie anche i suoi pesi. Poi l'intero detector viene congelato per tutti e tre i confronti e per il fine-tuning della fusione. Non vengono aggiunte immagini di training. Il test finale esclude i 100 ID usati nelle prove diagnostiche iniziali; l'elenco necessario a riprodurre questa separazione è in `data/llvip_subset/initial_trial_test_ids.json`.

Controllo sul test finale, mantenendo la fusione originale: senza la preparazione comune del detector si trovano **{raw_system['true_positives']} pedoni**, con AP50 **{raw_system['AP50']*100:.2f}%**; dopo la sua preparazione si trovano **{before['true_positives']} pedoni**, con AP50 **{before['AP50']*100:.2f}%**. Questo controllo cambia il detector e non va attribuito al fine-tuning della fusione. Il confronto principale sottostante mantiene invece il detector fisso.

Per lavorare su questa CPU usiamo una **variante guidata da AMG-Fuse**, distinta dalla sua uscita nativa a piena risoluzione:

1. La parte congelata di AMG-Fuse estrae feature a 40 × 32 pixel, salvate in cache FP16 e rilette in FP32.
2. Lo strato finale originale, una convoluzione 96 → 3, ricostruisce un risultato ridotto. Solo i suoi 2.592 coefficienti vengono adattati.
3. Dalle distanze quadratiche di quel risultato rispetto ai due ingressi ridotti ricaviamo α = sigmoid(4 × (distanza_termico − distanza_visibile)). Ingrandiamo α con interpolazione bilineare.
4. Combiniamo gli ingressi originali a 320 × 256 pixel: F = αV + (1−α)T. Questo conserva i dettagli delle misure. α è un coefficiente di combinazione, non una probabilità di affidabilità dei sensori.
5. Un **YOLOv8n COCO preaddestrato** cerca i pedoni. La loss di classificazione e posizione, calcolata sulle vere box, attraversa il detector congelato e aggiorna solo lo strato della fusione. Si aggiunge 0,1 volte l'errore quadratico tra α e la sua versione iniziale come regolarizzazione.

La media fissa impone α = 0,5. Le due varianti neurali usano identica architettura e identiche feature congelate: cambiano soltanto i coefficienti dello strato finale. Il detector mantiene sempre identici pesi e statistiche, verificati da un test con aggiornamento reale del gradiente. **“Senza NN” riguarda la fusione:** il detector è neurale in tutti e tre i casi.

## Risultati sulle 50 immagini di test

{table}

Differenza dopo − prima: **{extra:+d} pedoni correttamente trovati** e **{delta:+.2f} punti percentuali di recall**. Va letta insieme ai falsi positivi e ad AP50; un maggior numero di proposte non è automaticamente un miglioramento.

Matching uno a uno, IoU ≥ 0,50. Conteggi, precision e recall alla soglia condivisa {score_threshold:.2f}, scelta sulla validation massimizzando F1 con la fusione originale, prima del fine-tuning della fusione. YOLO riceve le immagini ingrandite a 640 × 512; le box sono riportate nelle coordinate 320 × 256. AP50 integra l'inviluppo della curva precision–recall sulle proposte con score ≥ 0,05 e massimo 100 proposte per immagine. È AP50 di questo esperimento a una classe, non COCO mAP 0,50:0,95. La confidence di una box non è l'accuratezza del sistema.

`runs/pioggia_llvip_yolo/results.json` conserva tutte le box vere e previste, split, metadati e metriche. `training.json` contiene la cronologia della validation. `evaluated_head.pt` conserva lo strato dopo le tre epoche realmente confrontato. `best.pt` e `best_head.pt` conservano invece la raccomandazione della validation, che può restare l'epoca zero. Non si sostituisce il risultato negativo di un modello adattato con una copia dell'originale.

## Presentazione e spiegazione

- `presentazione/WeatherFusion.pdf`: {len(slides)} slide aggiornate, sfondo chiaro, autore in apertura e ringraziamento finale.
- `Spiegazione_slides.txt`: obiettivo, punto di partenza e spiegazione di tutte le slide attuali.


## Ripetere l'esperimento su questo PC

Ambiente configurato: Python 3.11, PyTorch CPU e Torchvision compatibile. Le risorse sono già presenti.

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-meteo.txt
.\\.venv\\Scripts\\python.exe tools/download_llvip_subset.py
.\\.venv\\Scripts\\python.exe prepare_rain_detector.py
.\\.venv\\Scripts\\python.exe train_rain_detection.py --epochs 3
.\\.venv\\Scripts\\python.exe tools/build_rain_presentation.py
```

Il download viene ripreso usando i file già verificati; le feature congelate vengono riutilizzate. Il training registra il progresso in `runs/pioggia_llvip_yolo/progress.log`. Per un nuovo ambiente occorrono anche i pesi AMG-Fuse e YOLOv8n; le loro fonti e gli hash sono nel report e nei file di provenienza. Non cambiare manifest o pesi conservando una cache precedente: il codice ne verifica la corrispondenza.

## Fonti

- [LLVIP, dataset e annotazioni](https://github.com/bupt-ai-cz/LLVIP): Jia et al., ICCV Workshops 2021. Uso non commerciale per ricerca e didattica; licenza in `data/llvip_subset/LICENSE.md`.
- [Mirror LLVIP utilizzato](https://huggingface.co/datasets/McCheng/LLVIP), revisione e50976a; seed 17 per train/validation e 2028 per il test finale, escludendo gli ID dei test diagnostici.
- [AMG-Fuse, codice e checkpoint ufficiali](https://github.com/ixilai/AMG-Fuse); codice originale con licenza nella cartella vendor.
- [YOLOv8n e pesi ufficiali](https://docs.ultralytics.com/models/yolov8/).
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
print(f'{len(slides)} slides complete; extra true positives={extra}, recall delta={delta:+.2f} pp',flush=True)
