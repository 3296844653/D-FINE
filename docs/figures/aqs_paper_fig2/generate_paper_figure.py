"""Adapt the user-supplied D-FINE Figure 2, preserving the original FDR art.

PDF preserves the original PDF drawing; SVG/draw.io use a rendered background.
New AQS shapes, labels, connections and the enlarged inset are editable vectors.
Only figure files are written. The original paper and training code are untouched.
"""
from __future__ import annotations

import base64
import copy
import io
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
PAPER = Path('/Users/wq/Desktop/D-FINE.pdf')
PDF_OUT = PROJECT / 'output/pdf/dfine_paper_fig2_with_aqs_l2.pdf'
sys.path.insert(0, str(OUT.parent / 'aqs_l2_r1'))
import generate_diagrams as drawing

W, H = 2110, 1160
LEFT, TOP, RIGHT, BOTTOM = 90., 79., 548., 240.
X, Y, SCALE = 80., 115., 4.25
FIG_W, FIG_H = (RIGHT-LEFT)*SCALE, (BOTTOM-TOP)*SCALE
ORANGE, ORANGE_FILL = '#B85A1D', '#FFF1E3'
GRAY = '#64748B'


def map_point(x, y):
    return X + (x-LEFT)*SCALE, Y + (y-TOP)*SCALE


def make_scene():
    s = drawing.Scene('D-FINE Figure 2 + AQS-L2', W, H,
        'Adaptation of D-FINE paper Figure 2. Original decoder heads, residual refined distributions, weighting function and example boxes are retained. AQS is added to the query path after layer 2, not to the FDR distribution path. Enlarged inset adds the classification-score gate input.')
    s.text('title',80,27,1900,48,'D-FINE with AQS-L2 and Fine-grained Distribution Refinement',size=31,color=drawing.INK,bold=True)
    s.text('subtitle',80,78,1900,29,'Adapted from the original paper Fig. 2: FDR retained; AQS added after Decoder Layer 2.',size=19)
    # Coordinates measured from the original page: L2 bottom=181.30,
    # L3 top=191.17; their vertical query connection is at x=198.10.
    cx, l2_end = map_point(198.10,181.30)
    _, l3_start = map_point(198.10,191.17)
    marker_x, marker_y = map_point(189.60,182.70)
    s.group('old_query_arrow_mask',cx-8,l2_end+.5,16,l3_start-l2_end-1,
            fill=drawing.WHITE,stroke=drawing.WHITE)
    s.text('layer2_query_anchor',cx-1,l2_end-1,2,2,'',size=1)
    s.text('layer3_query_anchor',cx-1,l3_start-1,2,2,'',size=1)
    s.node('aqs_insert',marker_x,marker_y,17*SCALE,7*SCALE,'AQS',
           fill=ORANGE_FILL,stroke=ORANGE,title_size=18)
    s.edge('l2_aqs','layer2_query_anchor','aqs_insert',source_side='bottom',target_side='top',color=ORANGE)
    s.edge('aqs_l3','aqs_insert','layer3_query_anchor',source_side='bottom',target_side='top',color=ORANGE)
    s.text('zoom_title',80,822,560,43,'New AQS-L2 branch (enlarged)',size=24,color=ORANGE,bold=True)
    s.node('q2',220,936,235,106,'Layer 2 query Q2',['300 queries'],title_size=22,body_size=19)
    s.node('scores2',685,822,300,82,'ClsHead2 + LQE2',['Layer 2 logits S2'],fill=drawing.WHITE,stroke=ORANGE,title_size=22,body_size=18)
    s.node('aqs_enlarged',685,936,300,106,'AQS',['Gate + Context + Residual'],fill=ORANGE_FILL,stroke=ORANGE,title_size=29,body_size=20)
    s.node('q2_refined',1150,936,250,106,'Refined query Q2\u2032',['300 queries retained'],fill=ORANGE_FILL,stroke=ORANGE,title_size=22,body_size=18)
    s.node('layer3',1555,936,245,106,'Decoder Layer 3',['Original D-FINE head'],title_size=22,body_size=18)
    s.edge('q2_scores','q2','scores2',bends=[(560,989),(560,863)],color=ORANGE)
    s.edge('scores_gate','scores2','aqs_enlarged',source_side='bottom',target_side='top',color=ORANGE)
    s.edge('q2_aqs','q2','aqs_enlarged',color=ORANGE)
    s.edge('aqs_refined','aqs_enlarged','q2_refined',color=ORANGE)
    s.edge('refined_l3','q2_refined','layer3',color=ORANGE)
    s.text('fdr_note',1030,825,950,70,'Original L2 FDR is computed before AQS; distribution refinement is not routed through AQS.',size=18)
    s.text('caption',80,1082,1920,31,'Orange: added AQS. Original FDR / GO-LSD, losses and query count are unchanged.',size=20,color=drawing.INK)
    s.text('source',80,1125,1920,25,'Original illustrations are retained for architecture explanation, not SCB-S predictions or evidence of an AQS gain.',size=16)
    s.validate()
    return s


def render_original_background():
    doc=pdfium.PdfDocument(str(PAPER))
    page=doc[4]
    width,height=page.get_size()
    image=page.render(scale=6,crop=(LEFT,height-BOTTOM,width-RIGHT,TOP)).to_pil()
    buf=io.BytesIO();image.save(buf,format='PNG')
    (OUT/'original_fig2_background.png').write_bytes(buf.getvalue())
    return base64.b64encode(buf.getvalue()).decode('ascii')


def write_svg_and_drawio(scene,b64):
    svg=scene.svg()
    image=f'<image x="{X:g}" y="{Y:g}" width="{FIG_W:g}" height="{FIG_H:g}" preserveAspectRatio="none" href="data:image/png;base64,{b64}"/>'
    tag=f'<g font-family="{drawing.escape(drawing.FONT,quote=True)}">'
    assert svg.count(tag)==1
    svg=svg.replace(tag,tag+'\n'+image,1)
    ET.fromstring(svg)
    (OUT/'dfine_paper_fig2_with_aqs_l2.svg').write_text(svg,encoding='utf-8')
    file=ET.Element('mxfile',host='app.diagrams.net',type='device')
    scene.drawio(file)
    root=file.find('diagram/mxGraphModel/root')
    # Original paper art is one locked image; only the added diagram layer is
    # editable. The PDF separately retains the source drawing, not this image.
    cell=ET.Element('mxCell',id='original_paper_fig2',value='',style=
        f'shape=image;imageAspect=0;aspect=fixed;image=data:image/png,{b64};locked=1;movable=0;resizable=0;',vertex='1',parent='1')
    ET.SubElement(cell,'mxGeometry',x=str(X),y=str(Y),width=str(FIG_W),height=str(FIG_H),attrib={'as':'geometry'})
    root.insert(2,cell)
    ET.indent(file,space='  ')
    ET.ElementTree(file).write(OUT/'dfine_paper_fig2_with_aqs_l2.drawio',encoding='utf-8',xml_declaration=True)


def pdf_node(c,n):
    if not n.text_only:
        c.setFillColor(n.fill);c.setStrokeColor(n.stroke);c.setLineWidth(2)
        c.setDash([7,5] if n.dashed else [])
        if n.ellipse:
            c.ellipse(n.x,H-n.y-n.h,n.x+n.w,H-n.y,fill=1,stroke=1)
        else:
            c.roundRect(n.x,H-n.y-n.h,n.w,n.h,8,fill=1,stroke=1)
    heights=[size*1.5 for _,size,_,_ in n.lines]
    y=n.y+(n.h-sum(heights))/2
    for (label,size,bold,color),lh in zip(n.lines,heights):
        y+=lh/2;c.setFillColor(color);c.setFont('DiagramBold' if bold else 'Diagram',size)
        baseline=H-y-size*.33
        if n.text_only and n.align=='left':c.drawString(n.x,baseline,label)
        else:c.drawCentredString(n.x+n.w/2,baseline,label)
        y+=lh/2


def pdf_edges(c,scene):
    for e in scene.edges:
        points=scene.points(e);c.setStrokeColor(e.color);c.setLineWidth(2.2)
        c.setDash([7,5] if e.dashed else [])
        path=c.beginPath();path.moveTo(points[0][0],H-points[0][1])
        for x,y in points[1:]:path.lineTo(x,H-y)
        c.drawPath(path)
        if e.arrow:
            import math
            x,y=points[-1];px,py=points[-2]
            angle=math.atan2(y-py,x-px);length=9;half=4.5
            ux,uy=math.cos(angle),math.sin(angle)
            ax,ay=x-length*ux,y-length*uy
            c.setFillColor(e.color);tip=c.beginPath();tip.moveTo(x,H-y)
            tip.lineTo(ax-half*uy,H-(ay+half*ux));tip.lineTo(ax+half*uy,H-(ay-half*ux));tip.close()
            c.drawPath(tip,fill=1,stroke=0)


def write_pdf(scene):
    pdfmetrics.registerFont(TTFont('Diagram','/System/Library/Fonts/Supplemental/Arial.ttf'))
    pdfmetrics.registerFont(TTFont('DiagramBold','/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
    buf=io.BytesIO();c=canvas.Canvas(buf,pagesize=(W,H))
    c.setTitle('D-FINE original Figure 2 adapted with AQS-L2')
    for n in scene.groups.values():pdf_node(c,n)
    pdf_edges(c,scene)
    for n in scene.nodes.values():pdf_node(c,n)
    c.showPage();c.save()
    writer=PdfWriter();target=writer.add_blank_page(width=W,height=H)
    source=copy.copy(PdfReader(str(PAPER)).pages[4]);page_height=float(source.mediabox.height)
    crop=RectangleObject([LEFT,page_height-BOTTOM,RIGHT,page_height-TOP])
    source.cropbox=crop;source.trimbox=crop
    transform=Transformation().translate(-LEFT,-(page_height-BOTTOM)).scale(SCALE).translate(X,H-Y-FIG_H)
    target.merge_transformed_page(source,transform,expand=False)
    target.merge_page(PdfReader(buf).pages[0])
    writer.add_metadata({'/Title':'D-FINE Figure 2 with AQS-L2','/Subject':'Adapted architecture; original FDR artwork retained; AQS on L2 query path.'})
    PDF_OUT.parent.mkdir(parents=True,exist_ok=True)
    with PDF_OUT.open('wb') as f:writer.write(f)
    check=PdfReader(str(PDF_OUT));assert len(check.pages)==1
    text=check.pages[0].extract_text()
    assert 'AQS' in text and 'Decoder Layer 3' in text
    out=pdfium.PdfDocument(str(PDF_OUT))
    out[0].render(scale=2).to_pil().save(OUT/'dfine_paper_fig2_with_aqs_l2.png')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scene=make_scene();background=render_original_background()
    write_svg_and_drawio(scene,background);write_pdf(scene)
    print('Figure files:',OUT)
    print('PDF:',PDF_OUT)


if __name__=='__main__':main()
