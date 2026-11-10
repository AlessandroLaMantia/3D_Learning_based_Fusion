"""Render the saved deck and its English edition without rerunning the experiment."""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
from tools.slide_design import write_pdf
from tools.english_presentation import overview, write_english
from tools.performance_slide import insert_performance_slide
from tools.closing_slides import update_closing_slides

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'presentazione'

def main():
    slides = json.loads((OUT/'slides.json').read_text(encoding='utf-8'))
    slides = [s for s in slides if s['layout'] != 'code']
    insert_performance_slide(slides, OUT)
    update_closing_slides(slides)
    for i, s in enumerate(slides, 1):
        for a in s['elements']:
            if a['kind'] == 'text' and a['y'] == 7.05:
                a['text'] = f'Alessandro La Mantia {i:02d}'
    write_pdf(slides, OUT)
    (OUT/'slides.json').write_text(json.dumps(slides, ensure_ascii=False, indent=2), encoding='utf-8')
    # Discard only superseded generated previews inside this output directory.
    for preview in OUT.glob('anteprima_*.png'):
        if int(preview.stem.rsplit('_', 1)[1]) > len(slides):
            preview.unlink()
    overview(slides, OUT)
    write_english(slides, OUT)
    guide = f'WEATHERFUSION — SPIEGAZIONE DELLE {len(slides)} SLIDE\nPresentazione attuale: presentazione/WeatherFusion.pdf\n\n'
    for i, s in enumerate(slides, 1):
        guide += f"SLIDE {i} — {s['title']}\n\n{s['note']}\n\n"
    (ROOT/'Spiegazione_slides.txt').write_text(guide, encoding='utf-8')
    readme = ROOT/'README.md'
    text = readme.read_text(encoding='utf-8')
    text = re.sub(r'(WeatherFusion.pdf`: )\d+ slide', rf'\g<1>{len(slides)} slide', text)
    english_line = '- `presentazione/english/WeatherFusion.pdf`: stessa presentazione in inglese, con note in `presentazione/english/Speaker_notes.txt`.\n'
    if english_line not in text:
        text = text.replace('- `Spiegazione_slides.txt`:', english_line+'- `Spiegazione_slides.txt`:')
    readme.write_text(text, encoding='utf-8')
    print(f'Italian and English editions rendered: {len(slides)} slides each.')

if __name__ == '__main__':
    main()
