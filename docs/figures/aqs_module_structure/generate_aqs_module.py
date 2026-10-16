"""Portrait, native editable AQS-L2 figure; reference informs style only."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pypdfium2 as pdfium
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[2]
PDF=PROJECT/'output/pdf/aqs_module_structure.pdf'
sys.path.insert(0,str(OUT.parent/'aqs_reference_style'))
import generate_reference_figure as ref

W,H=680,1035
INK='#202020'
LINE='#414141'
BLUE=ref.base.BLUE
GATE=ref.base.ORANGE
PINK='#FDEEF2'
TEAL='#31B6AC'
TOKEN_BLUE='#416EA1'
ROSE='#EB718B'


class PortraitScene(ref.ReferenceScene):
    @staticmethod
    def svg_node(n):
        svg=ref.ReferenceScene.svg_node(n)
        if getattr(n,'italic',False):
            svg=svg.replace('<text ','<text font-style="italic" ')
        return svg

    def drawio(self,parent):
        super().drawio(parent)
        cells={c.attrib['id']:c for c in parent.findall('diagram')[-1].findall('mxGraphModel/root/mxCell')}
        for n in self.nodes.values():
            if getattr(n,'italic',False):
                cell=cells[n.id]
                cell.set('value',cell.attrib['value'].replace('line-height:','font-style:italic;line-height:'))


def node(s,id,x,y,w,h,title='',body=(),fill='#FFFFFF',size=24,body_size=18):
    return ref.node(s,id,x,y,w,h,title,body,fill=fill,size=size,body_size=body_size)


def edge(s,id,src,dst,**kwargs):
    kwargs.setdefault('color',ref.base.LINE)
    s.edge(id,src,dst,**kwargs)


def build():
    s=PortraitScene('AQS internal structure',W,H,
        'Portrait AQS-L2 diagram: confidence-based hard gates with STE, raw-query '
        'mean pooling, LayerNorm/context fusion MLP, and gated residual update. '
        'All 300 queries are retained. Eight displayed queries and confidences '
        'are schematic, not experimental data. No quantizer or regularization loss.')
    s.group('panel',12,12,656,1011,fill=PINK,stroke=PINK)
    s.groups['panel'].radius=0
    s.groups['panel'].no_stroke=True
    ref.text(s,'title',24,19,632,45,'Adaptive Query Selection (AQS)',size=28,bold=True)
    s.nodes['title'].italic=True

    # Schematic maximum class confidences, not attention maps or quantization.
    s.group('confidence',65,90,550,110,fill=PINK,stroke='#888888',dashed=True)
    s.groups['confidence'].radius=0
    s.groups['confidence'].outline_only=True
    ref.text(s,'confidence_title',80,95,520,39,'Class Confidence',size=25)
    confidence=[.21,.83,.74,.36,.92,.68,.27,.89]
    conf_members=['confidence_title']
    for i,value in enumerate(confidence):
        id=f'confidence_{i}'
        tile=node(s,id,100.5+i*62,145,45,36,
                  fill='#EB9DAE' if value>=.5 else '#DCE6F0')
        tile.lines=[]
        tile.radius=0;tile.square=True;tile.stroke='#FFFFFF'
        conf_members.append(id)
    s.editable_groups['confidence']=conf_members

    node(s,'gate',350,245,250,68,'Adaptive Gating',fill=ROSE,size=25)
    node(s,'threshold',80,245,210,68,'Threshold',fill='#F8CCA9',size=25)

    # Binary gate positions correspond to the confidence row and query tokens.
    s.group('binary_gate',350,350,250,44,fill=PINK,stroke=PINK)
    s.groups['binary_gate'].no_stroke=True
    gate_members=[]
    for i,value in enumerate(confidence):
        id=f'gate_value_{i}'
        tile=node(s,id,358+i*30,357,24,30,
                  fill=TEAL if value>=.5 else '#F2CCD5')
        tile.lines=[]
        tile.radius=0;tile.square=True;tile.stroke='#FFFFFF'
        gate_members.append(id)
    s.editable_groups['binary_gate']=gate_members

    # Separate original query features feed both pooling and local fusion.
    s.group('query',80,420,210,98,fill=PINK,stroke=PINK)
    s.groups['query'].no_stroke=True
    ref.text(s,'query_title',80,420,210,39,'Query Features',size=25)
    colors=['#F5D9DF',TEAL,TEAL,'#C0D1E6','#BA7122','#BA7122','#C0D1E6',TOKEN_BLUE]
    query_members=['query_title']
    for i,color in enumerate(colors):
        id=f'query_token_{i}'
        tile=node(s,id,85+i*26,466,18,32,fill=color)
        tile.lines=[];tile.radius=3;tile.stroke='#75818D'
        query_members.append(id)
    s.editable_groups['query']=query_members
    node(s,'pool',350,435,250,68,'Query Pooling',fill='#E6E6E6',size=25)
    node(s,'fusion',350,555,250,68,'Context Fusion',fill='#E6E6E6',size=25)
    node(s,'residual',350,675,250,68,'Gated Residual',fill='#F1BBC8',size=25)
    n=node(s,'sum',452,780,46,46,'+',fill='#FFFFFF',size=29)
    n.ellipse=True

    s.group('output',65,852,550,106,fill=PINK,stroke='#888888',dashed=True)
    s.groups['output'].radius=0
    s.groups['output'].outline_only=True
    ref.text(s,'output_title',80,856,520,39,'Refined Queries',size=25)
    out_members=['output_title']
    for i,color in enumerate(colors):
        id=f'output_token_{i}'
        tile=node(s,id,115+i*60,907,30,34,fill=color)
        tile.lines=[];tile.radius=3;tile.stroke='#75818D'
        # A short stripe marks the residual only on gated-on query positions.
        out_members.append(id)
        if confidence[i]>=.5:
            stripe=node(s,id+'_update',118+i*60,910,24,4,fill='#FFFFFF')
            stripe.lines=[];stripe.radius=0;stripe.no_stroke=True
            out_members.append(stripe.id)
    s.editable_groups['output']=out_members
    ref.text(s,'l3',170,978,340,39,'Decoder L3',size=25)

    edge(s,'confidence_gate','confidence','gate',source_side='bottom',target_side='top',source_fraction=410/550)
    edge(s,'threshold_gate','threshold','gate')
    edge(s,'gate_binary','gate','binary_gate',source_side='bottom',target_side='top')
    edge(s,'gate_pool','binary_gate','pool',source_side='bottom',target_side='top')
    edge(s,'gate_residual','binary_gate','residual',
         target_side='right',bends=[(632,372),(632,709)])
    edge(s,'query_pool','query','pool')
    edge(s,'pool_fusion','pool','fusion',source_side='bottom',target_side='top')
    edge(s,'query_fusion','query','fusion',source_side='bottom',
         bends=[(185,589)])
    edge(s,'fusion_residual','fusion','residual',source_side='bottom',target_side='top')
    edge(s,'residual_sum','residual','sum',source_side='bottom',target_side='top')
    edge(s,'identity','query','sum',source_side='left',target_side='left',
         bends=[(42,469),(42,803)])
    edge(s,'sum_output','sum','output',source_side='bottom',target_side='top',target_fraction=410/550)
    edge(s,'output_l3','output','l3',source_side='bottom',target_side='top')

    ids=set(s.nodes)|set(s.groups)
    for e in s.edges:
        assert e.source in ids and e.target in ids
        for (x1,y1),(x2,y2) in zip(s.points(e),s.points(e)[1:]):
            assert abs(x1-x2)<1e-7 or abs(y1-y2)<1e-7,e.id
    assert {e.target for e in s.edges if e.source=='binary_gate'}=={'pool','residual'}
    assert {e.source for e in s.edges if e.target=='sum'}=={'residual','query'}
    assert len([n for n in s.nodes if n.startswith('query_token_')])==8
    assert len([n for n in s.nodes if n.startswith('output_token_') and not n.endswith('_update')])==8
    assert {(s.nodes[id].x,s.nodes[id].w,s.nodes[id].h) for id in ('gate','pool','fusion','residual')}=={(350,250,68)}
    assert s.nodes['fusion'].y-s.nodes['pool'].y==s.nodes['residual'].y-s.nodes['fusion'].y==120
    assert all(size>=25 for n in s.nodes.values() for _,size,_,_ in n.lines)
    assert all(len(n.lines)<=1 for n in s.nodes.values())
    s.validate()
    return s


def pdf_node(c,n):
    """Keep dashed reference-style frames and italic heading native in PDF."""
    c.saveState()
    if not n.text_only:
        c.setStrokeColor(n.stroke);c.setFillColor(n.fill);c.setLineWidth(1.4)
        c.setDash([7,5] if n.dashed else [])
        fill=0 if getattr(n,'outline_only',False) else 1
        stroke=0 if getattr(n,'no_stroke',False) else 1
        radius=getattr(n,'radius',8)
        if n.ellipse:
            c.ellipse(n.x,H-n.y-n.h,n.x+n.w,H-n.y,fill=fill,stroke=stroke)
        elif radius:
            c.roundRect(n.x,H-n.y-n.h,n.w,n.h,radius,fill=fill,stroke=stroke)
        else:
            c.rect(n.x,H-n.y-n.h,n.w,n.h,fill=fill,stroke=stroke)
    height=sum(size*1.5 for _,size,_,_ in n.lines)
    y=n.y+(n.h-height)/2
    for label,size,bold,color in n.lines:
        font='DiagramBoldItalic' if getattr(n,'italic',False) else 'DiagramBold' if bold else 'Diagram'
        c.setFont(font,size);c.setFillColor(color)
        baseline=H-y-size*.75-size*.33
        if n.text_only and n.align=='left':c.drawString(n.x,baseline,label)
        else:c.drawCentredString(n.x+n.w/2,baseline,label)
        y+=size*1.5
    c.restoreState()


def export(s):
    OUT.mkdir(parents=True,exist_ok=True);PDF.parent.mkdir(parents=True,exist_ok=True)
    ref.W,ref.H=W,H;ref.pub.W,ref.pub.H=W,H
    pdfmetrics.registerFont(TTFont('Diagram','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
    pdfmetrics.registerFont(TTFont('DiagramBold','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))
    pdfmetrics.registerFont(TTFont('DiagramBoldItalic','/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf'))
    for n in s.nodes.values():
        for label,size,bold,_ in n.lines:
            font='DiagramBoldItalic' if getattr(n,'italic',False) else 'DiagramBold' if bold else 'Diagram'
            assert pdfmetrics.stringWidth(label,font,size)<=n.w-2,n.id
            assert all(ord(char) in pdfmetrics.getFont(font).face.charToGlyph for char in label),n.id
        assert sum(size*1.5 for _,size,_,_ in n.lines)<=n.h+1e-7,n.id
    svg=s.svg();ET.fromstring(svg);assert '<image' not in svg
    (OUT/'aqs_module_structure.svg').write_text(svg,encoding='utf-8')
    root=ET.Element('mxfile',host='app.diagrams.net',type='device')
    s.drawio(root);ref.verify_drawio(root);ET.indent(root,space='  ')
    ET.ElementTree(root).write(OUT/'aqs_module_structure.drawio',encoding='utf-8',xml_declaration=True)
    c=canvas.Canvas(str(PDF),pagesize=(W,H))
    c.setTitle('AQS-L2 internal structure - source-checked editable diagram')
    for n in s.groups.values():pdf_node(c,n)
    ref.pdf_edges(c,s)
    for n in s.nodes.values():pdf_node(c,n)
    c.showPage();c.save()
    r=PdfReader(PDF);assert len(r.pages)==1
    assert '/XObject' not in r.pages[0]['/Resources']
    for label in ('Query Pooling','Context Fusion','Adaptive Gating','Gated Residual','Decoder L3'):
        assert label in r.pages[0].extract_text(),label
    pdfium.PdfDocument(str(PDF))[0].render(scale=2).to_pil().save(OUT/'aqs_module_structure.png')
    print('Editable draw.io:',OUT/'aqs_module_structure.drawio')
    print('Vector PDF:',PDF)


if __name__=='__main__':
    export(build())
