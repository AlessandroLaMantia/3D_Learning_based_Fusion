"""Build the bilingual test-performance slide from the saved experiment report."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tools.slide_design import BG, INK, GRAY, TEAL, GOLD, fit

ROOT = Path(__file__).resolve().parents[1]

def performance_slide(out, language='it'):
    report = json.loads((ROOT/'runs/pioggia_llvip_yolo/results.json').read_text(encoding='utf-8'))
    keys = ['media_fissa', 'pretrained', 'finetuned']
    data = [report['test'][key]['summary'] for key in keys]
    en = language == 'en'
    labels = ['Fixed average', 'Pretrained fusion', 'Fine-tuned fusion'] if en else ['Media fissa', 'Fusione preaddestrata', 'Fusione dopo fine-tuning']
    colors = ['#8299ac', TEAL, GOLD]
    chart = out/'figure'/f'performance_{language}.png'
    chart.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.25), facecolor=BG)
    panels = [
        (['true_positives', 'missed', 'false_positives'],
         ['Detected', 'Missed', 'False positives'] if en else ['Trovati', 'Mancati', 'Falsi positivi'], 1,
         'Pedestrian counts' if en else 'Conteggi dei pedoni'),
    ]
    for metrics, ticks, scale, title in panels:
        ax.set_facecolor(BG)
        for i, (row, label, color) in enumerate(zip(data, labels, colors)):
            values = [row[m]*scale for m in metrics]
            bars = ax.bar(np.arange(3)+(i-1)*.25, values, .23, color=color, label=label)
            values_text = [f'{v:.1f}' if scale == 100 else str(int(v)) for v in values]
            if not en:
                values_text = [v.replace('.', ',') for v in values_text]
            ax.bar_label(bars, labels=values_text, padding=3, fontsize=13, color=INK)
        ax.set_xticks(np.arange(3), ticks, fontsize=14, color=INK)
        ax.set_ylim(0, 100 if scale == 100 else max(row[m] for row in data for m in metrics)*1.2)
        ax.tick_params(axis='y', labelsize=9, colors=GRAY, length=0)
        ax.tick_params(axis='x', length=0)
        ax.set_title(title, fontsize=13, color=INK, weight='bold', pad=14)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#bdced9')
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.15)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, frameon=False, fontsize=11, labelcolor=INK)
    fig.subplots_adjust(left=.055, right=.98, top=.87, bottom=.22)
    fig.savefig(chart, dpi=180, facecolor=BG)
    plt.close(fig)
    sample = data[0]
    title = 'Performance of the three fusion methods' if en else 'Prestazioni dei tre metodi di fusione'
    subtitle = (f"{sample['images']} LLVIP test pairs · {sample['ground_truth_count']} annotated pedestrians · same frozen YOLO" if en else
                f"{sample['images']} coppie LLVIP di test · {sample['ground_truth_count']} pedoni annotati · stesso YOLO congelato")
    gain = data[2]['true_positives']-data[1]['true_positives']
    fp = data[2]['false_positives']-data[1]['false_positives']
    threshold = sample['score_threshold']
    notes = (f'The chart compares the three fusion methods on the same held-out test set using the same frozen detector. Recall measures how many annotated pedestrians are detected; precision measures how many predictions are correct. AP50 summarises the precision–recall curve at IoU 0.50 over proposals scoring at least 0.05. Counts, recall and precision use the shared validation-selected score threshold of {threshold:.2f}. The counts show detected pedestrians, missed pedestrians and false positives. Fine-tuning yields {gain:+d} detected pedestrians and {fp:+d} false positives compared with pretrained fusion. The experiment evaluates epoch {report["metadata"]["evaluated_epoch"]}; validation recommends epoch {report["metadata"]["recommended_epoch"]}. The rain is simulated.' if en else
             f'Il grafico confronta le tre fusioni sullo stesso test separato dal training, usando lo stesso detector congelato. Il recall misura quanti pedoni annotati vengono trovati; la precision misura quante previsioni sono corrette. AP50 riassume la curva precision–recall con IoU 0,50 sulle proposte con score almeno 0,05. Conteggi, recall e precision usano la soglia condivisa {threshold:.2f}, scelta sulla validation. I conteggi mostrano pedoni trovati, mancati e falsi positivi. Il fine-tuning produce {gain:+d} pedoni trovati e {fp:+d} falsi positivi rispetto alla fusione preaddestrata. Si valuta l’epoca {report["metadata"]["evaluated_epoch"]}; la validation raccomanda l’epoca {report["metadata"]["recommended_epoch"]}. La pioggia è simulata.')
    elements = []
    def text(x,y,w,h,value,size,color=INK,bold=False,align='left'):
        value,size = fit(value,w,h,size,bold)
        elements.append(dict(kind='text',x=x,y=y,w=w,h=h,text=value,size=size,color=color,bold=bold,align=align))
    text(.75,.9,11.8,.85,title,31,bold=True)
    text(.75,1.85,11.8,.6,subtitle,17,GRAY)
    text(.75,7.05,11.8,.23,'Alessandro La Mantia 12',10,GRAY)
    elements.append(dict(kind='image',x=.65,y=2.45,w=12.03,h=4.25,path=str(chart.resolve())))
    return dict(title=title,subtitle=subtitle,note=notes,layout='performance-comparison',elements=elements)

def insert_performance_slide(slides, out):
    slides[:] = [s for s in slides if s['layout'] != 'performance-comparison']
    index = next(i for i,s in enumerate(slides) if s['layout'] == 'demo-transition')
    slides.insert(index+1, performance_slide(out))
    for i,s in enumerate(slides,1):
        for a in s['elements']:
            if a['kind']=='text' and a['y']==7.05:
                a['text'] = f'Alessandro La Mantia {i:02d}'
