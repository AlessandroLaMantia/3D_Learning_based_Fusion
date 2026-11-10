"""Impaginazioni editoriali condivise da PDF, PowerPoint e HTML."""
from pathlib import Path
import base64
import html
import json
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.patches import FancyBboxPatch, Rectangle
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

BG, WHITE, INK, GRAY, TEAL, GOLD = "#edf3f8", "#ffffff", "#183249", "#51677b", "#087d94", "#ad610d"
W,H = 13.333,7.5
METRICS = TextToPath()


@lru_cache(maxsize=8192)
def width(text,size,bold):
    return METRICS.get_text_width_height_descent(text,FontProperties(family="DejaVu Sans",size=size,weight="bold" if bold else "normal"),False)[0]


def fit(text,w,h,size,bold=False):
    for fs in range(size,min(size,14)-1,-1):
        lines = []
        for para in text.split("\n"):
            line = ""
            for word in para.split():
                candidate = (line+" "+word).strip()
                if line and width(candidate,fs,bold) > w*72-4:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        if len(lines)*fs*1.25 <= h*72:
            return "\n".join(lines),fs
    raise ValueError(f"Blocco troppo lungo: {text}")


def write_pptx(slides,out):
    prs=Presentation(); prs.slide_width=Inches(W); prs.slide_height=Inches(H)
    prs.core_properties.title="WeatherFusion — Pedoni nella pioggia"
    prs.core_properties.author="Alessandro La Mantia"
    for s in slides:
        slide=prs.slides.add_slide(prs.slide_layouts[6]); fill=slide.background.fill
        fill.solid(); fill.fore_color.rgb=RGBColor.from_string(BG[1:])
        for a in s["elements"]:
            x,y,w,h=(a[k] for k in ("x","y","w","h"))
            if a["kind"]=="rect":
                shape=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
                shape.fill.solid(); shape.fill.fore_color.rgb=RGBColor.from_string(a["color"][1:]); shape.line.fill.background()
            elif a["kind"]=="image":
                with Image.open(a["path"]) as im: ratio=im.width/im.height
                iw=min(w,h*ratio); ih=iw/ratio
                slide.shapes.add_picture(a["path"],Inches(x+(w-iw)/2),Inches(y+(h-ih)/2),width=Inches(iw),height=Inches(ih))
            else:
                shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=shape.text_frame
                tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0; tf.word_wrap=False
                for n,line in enumerate(a["text"].split("\n")):
                    p=tf.paragraphs[0] if n==0 else tf.add_paragraph(); p.text=line
                    p.font.name="Segoe UI"; p.font.size=Pt(a["size"]); p.font.bold=a["bold"]
                    p.font.color.rgb=RGBColor.from_string(a["color"][1:]); p.line_spacing=1.2; p.space_after=Pt(0)
                    from pptx.enum.text import PP_ALIGN
                    p.alignment=PP_ALIGN.RIGHT if a["align"]=="right" else PP_ALIGN.LEFT
        slide.notes_slide.notes_text_frame.text=s["note"]
    prs.save(out/"WeatherFusion.pptx")


def write_pdf(slides,out):
    checks=[]
    with PdfPages(out/"WeatherFusion.pdf") as pdf:
        for i,s in enumerate(slides):
            fig=plt.figure(figsize=(W,H),facecolor=BG); areas=[]
            for a in s["elements"]:
                x,y,w,h=(a[k] for k in ("x","y","w","h"))
                if a["kind"]=="rect":
                    fig.add_artist(Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,facecolor=a["color"],edgecolor="none",zorder=0))
                elif a["kind"]=="image":
                    ax=fig.add_axes([x/W,1-(y+h)/H,w/W,h/H]); ax.imshow(plt.imread(a["path"])); ax.axis("off")
                else:
                    tx=(x+w)/W if a["align"]=="right" else (x+w/2)/W if a["align"]=="center" else x/W
                    item=fig.text(tx,1-y/H,a["text"],fontsize=a["size"],fontfamily="DejaVu Sans",color=a["color"],
                                  weight="bold" if a["bold"] else "normal",va="top",ha=a["align"],linespacing=1.2)
                    if a.get('url'):item.set_url(a['url'])
                    areas.append((item,(x/W,1-(y+h)/H,(x+w)/W,1-y/H)))
            fig.canvas.draw(); renderer=fig.canvas.get_renderer()
            for item,(x0,y0,x1,y1) in areas:
                box=item.get_window_extent(renderer).transformed(fig.transFigure.inverted())
                if box.x0<x0-.003 or box.x1>x1+.003 or box.y0<y0-.003 or box.y1>y1+.003:
                    raise ValueError(f"Testo fuori area, slide {i+1}: {item.get_text()}")
            checks.append(dict(slide=i+1,title=s["title"],layout=s["layout"],checked_text_blocks=len(areas),overflow=False))
            pdf.savefig(fig,facecolor=BG)
            fig.savefig(out/f"anteprima_{i+1:02d}.png",dpi=85,facecolor=BG)
            plt.close(fig)
    (out/"layout_checks.json").write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding="utf-8")


def write_html(slides,out):
    sections=[]
    for i,s in enumerate(slides):
        content=[]
        for a in s["elements"]:
            pos=f"left:{a['x']/W*100:.4f}%;top:{a['y']/H*100:.4f}%;width:{a['w']/W*100:.4f}%;height:{a['h']/H*100:.4f}%;"
            if a["kind"]=="rect": content.append(f'<div class="el" style="{pos}background:{a["color"]}"></div>')
            elif a["kind"]=="image":
                data=base64.b64encode(Path(a["path"]).read_bytes()).decode()
                content.append(f'<img class="el" style="{pos}object-fit:contain" src="data:image/png;base64,{data}" alt="{html.escape(s["title"])}">')
            else:
                size=a["size"]/(W*72)*100
                style=pos+f'font-size:{size:.4f}cqw;color:{a["color"]};font-weight:{700 if a["bold"] else 400};text-align:{a["align"]};'
                content.append(f'<div class="el text" style="{style}">'+html.escape(a["text"]).replace("\n","<br>")+"</div>")
        sections.append(f'<section class="slide" aria-label="{html.escape(s["title"])}"><div class="canvas">'+"".join(content)+f'</div><aside>{html.escape(s["note"])}</aside></section>')
    page='''<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WeatherFusion</title>
<style>*{box-sizing:border-box}body{margin:0;background:#dfe8ef;color:#183249;font-family:"DejaVu Sans","Segoe UI",sans-serif}.slide{display:none;width:min(96vw,1280px);margin:2vh auto}.slide.active{display:block}.canvas{aspect-ratio:16/9;container-type:inline-size;background:#edf3f8;position:relative;box-shadow:0 6px 24px #18324915}.el{position:absolute}.text{line-height:1.2}aside{display:none;padding:24px;background:white;line-height:1.6}.notes aside{display:block}nav{display:flex;gap:16px;justify-content:center;align-items:center;margin:16px;color:#51677b}button{padding:10px 18px;border:1px solid #b5cdd9;border-radius:8px;background:white;color:#087d94;cursor:pointer}button:focus-visible{outline:2px solid #ad610d}@media print{.slide{display:block;break-after:page;width:100%;margin:0}nav,aside{display:none!important}}</style>'''+"".join(sections)+'''<nav><button id="prev">←</button><span>Frecce per navigare · N per le note</span><button id="next">→</button><button id="notes">Note relatore</button></nav>
<script>const slides=[...document.querySelectorAll('.slide')];let i=0;function show(n){i=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,k)=>s.classList.toggle('active',k===i));history.replaceState(null,'','#'+(i+1))}function notes(){document.body.classList.toggle('notes')}document.querySelector('#prev').onclick=()=>show(i-1);document.querySelector('#next').onclick=()=>show(i+1);document.querySelector('#notes').onclick=notes;document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();show(i+1)}if(e.key==='ArrowLeft')show(i-1);if(e.key.toLowerCase()==='n')notes()});show(Number(location.hash.slice(1)||1)-1)</script></html>'''
    (out/"WeatherFusion.html").write_text(page,encoding="utf-8")
