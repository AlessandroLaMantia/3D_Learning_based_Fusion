"""Simple bilingual challenges, lessons learned and closing slides."""
from tools.slide_design import fit, INK, TEAL, GRAY

CONTENT = {
    'it': {
        'challenges': ('Due problemi nella data fusion e le soluzioni', [
            ('1. Dare lo stesso peso a due immagini di qualità diversa',
             'Ho usato pesi variabili per zona, guidati dalla rete.'),
            ('2. Rischiare di perdere dettagli dei pedoni',
             'Ho applicato la mappa dei pesi ingrandita alle immagini originali.'),
        ]),
        'lessons-learned': ('Tre cose imparate in questo progetto', [
            ('1. Combinare due sensori',
             'Visibile e termico danno informazioni diverse: unirle può aiutare a trovare i pedoni.'),
            ('2. Adattare una rete già addestrata',
             'Si può partire da pesi esistenti e aggiornare solo una piccola parte della rete.'),
            ('3. Valutare davvero un miglioramento',
             'Trovare più pedoni non basta: bisogna controllare anche i falsi positivi.'),
        ]),
        'thanks': ("Grazie per l'attenzione", []),
    },
    'en': {
        'challenges': ('Two data fusion problems and their solutions', [
            ('1. Giving equal weight to images of different quality',
             'I used spatially varying weights guided by the network.'),
            ('2. Risking the loss of pedestrian details',
             'I applied the enlarged weight map to the original images.'),
        ]),
        'lessons-learned': ('Three things I learned from this project', [
            ('1. Combining two sensors',
             'Visible and thermal images provide different information: combining them can help detect pedestrians.'),
            ('2. Adapting a pretrained network',
             'Existing weights can be reused while updating only a small part of the network.'),
            ('3. Evaluating an improvement',
             'Detecting more pedestrians is not enough: false positives must also be checked.'),
        ]),
        'thanks': ('Thank you for your attention', []),
    },
}

def closing_slide(layout, number, language='it'):
    title, items = CONTENT[language][layout]
    elements = []
    def text(x, y, w, h, value, size, color=INK, bold=False, align='left'):
        value, size = fit(value, w, h, size, bold)
        elements.append(dict(kind='text', x=x, y=y, w=w, h=h, text=value,
                             size=size, color=color, bold=bold, align=align))
    if layout == 'thanks':
        text(.75, 3.25, 11.8, 1, title, 38, TEAL, True, 'center')
    else:
        text(.75, .9, 11.8, .85, title, 31, bold=True)
        for i, (heading, body) in enumerate(items):
            y = (2.65 + i*1.8) if len(items) == 2 else (2.35 + i*1.4)
            text(1, y, 11.3, .55, heading, 24, TEAL, True)
            if layout != 'lessons-learned':
                text(1, y+.63, 11.3, .85, body, 21)
        text(.75, 7.05, 11.8, .23, f'Alessandro La Mantia {number:02d}', 10, GRAY)
    note = '\n\n'.join(f'{heading}. {body}' for heading, body in items)
    return dict(title=title, subtitle='', note=note, layout=layout, elements=elements)

def update_closing_slides(slides):
    slides[:] = [s for s in slides if s['layout'] != 'lessons-learned']
    index = next(i for i,s in enumerate(slides) if s['layout'] == 'challenges')
    slides[index] = closing_slide('challenges', index+1)
    slides.insert(index+1, closing_slide('lessons-learned', index+2))
    for i,s in enumerate(slides,1):
        if s['layout'] == 'thanks':
            slides[i-1] = closing_slide('thanks', i)
        else:
            for a in s['elements']:
                if a['kind'] == 'text' and a['y'] == 7.05:
                    a['text'] = f'Alessandro La Mantia {i:02d}'
