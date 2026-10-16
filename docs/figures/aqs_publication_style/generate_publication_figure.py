"""Native-vector D-FINE-S + AQS figure; paper references inform layout only."""
from __future__ import annotations

import copy
import io
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import pypdfium2 as pdfium
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[2]
PDF_OUT=PROJECT/'output/pdf/dfine_s_aqs_publication_style.pdf'
sys.path.insert(0,str(OUT.parent/'aqs_l2_r1'))
import generate_diagrams as base

W,H=2460,1190
INK='#252525'
LINE='#555555'
BLUE='#7C97B0'
ORANGE='#B85A1D'
FONT="'Times New Roman', 'Noto Serif', serif"
GREEN=['#D8E3C1','#E7EDDA','#F3F5E9']
STAGES=['#FFF0B8','#D8E6F3','#E6DFF0']


def polygon_points(n):
    if getattr(n,'kind','')=='plane':
        d=n.w*.14
        return [(n.x+d,n.y),(n.x+n.w,n.y),(n.x+n.w-d,n.y+n.h),(n.x,n.y+n.h)]
    if getattr(n,'kind','')=='trapezoid':
        return [(n.x+n.w*.15,n.y),(n.x+n.w*.85,n.y),(n.x+n.w,n.y+n.h),(n.x,n.y+n.h)]
    return None


class PublicationScene(base.Scene):
    @staticmethod
    def svg_node(n):
        if getattr(n,'vertical',False):
            blank=copy.copy(n);blank.lines=[]
            shape=base.Scene.svg_node(blank)
            label,size,bold,color=n.lines[0]
            return shape+f'<text transform="translate({n.x+n.w/2:g} {n.y+n.h/2:g}) rotate(-90)" text-anchor="middle" dominant-baseline="central" font-size="{size:g}" font-weight="700" fill="{color}">{base.escape(label)}</text>'
        svg=base.Scene.svg_node(n)
        if getattr(n,'square',False):svg=svg.replace('rx="8"','rx="0"')
        pts=polygon_points(n)
        if pts:
            poly=f'<polygon points="{" ".join(f"{x:g},{y:g}" for x,y in pts)}" fill="{n.fill}" stroke="{n.stroke}" stroke-width="1.5"/>'
            svg=re.sub(r'<rect[^>]+/>',poly,svg,count=1)
        return svg

    def svg(self):
        return super().svg().replace(base.escape(base.FONT,quote=True),base.escape(FONT,quote=True))

    def drawio(self,parent):
        super().drawio(parent)
        root=parent.findall('diagram')[-1].find('mxGraphModel/root')
        nodes={**self.groups,**self.nodes}
        for cell in root.findall('mxCell'):
            id=cell.attrib['id'];style=cell.attrib.get('style','').replace('fontFamily=Arial','fontFamily=Times New Roman')
            if id in nodes:
                n=nodes[id]
                if getattr(n,'kind','')=='plane':style='shape=parallelogram;size=0.14;'+style.replace('rounded=1;','')
                if getattr(n,'kind','')=='trapezoid':style='shape=trapezoid;size=0.15;'+style.replace('rounded=1;','')
                if getattr(n,'vertical',False):style+='horizontal=0;'
                if getattr(n,'square',False):style=style.replace('rounded=1;','rounded=0;')
            if style:cell.set('style',style)


def tensor(s,id,x,y,w,h,title,dimensions):
    # Three offset parallelograms denote a tensor, not observed activations.
    s.text(id,x,y-12,w+16,h+24,'',size=1)
    for j in range(2,-1,-1):
        s.group(f'{id}_plane{j}',x+j*5,y-j*6,w,h,fill=GREEN[j],stroke='#8F997B')
        s.groups[f'{id}_plane{j}'].kind='plane'
    s.text(id+'_name',x-10,y+h+8,w+38,24,title,size=18,color=INK,bold=True,align='center')
    if dimensions:s.text(id+'_size',x-25,y+h+35,w+70,22,dimensions,size=15,color='#5B6451',align='center')


def tokens(s,id,x,y,count,w=30,cell_h=12,gap=4,title='',subtitle=''):
    total=count*(cell_h+gap)-gap
    s.text(id,x,y,w,total,'',size=1)
    for i in range(count):
        s.group(f'{id}_{i}',x,y+i*(cell_h+gap),w,cell_h,fill='#FFF3CF',stroke='#6F6F64')
        s.groups[f'{id}_{i}'].square=True
    if title:s.text(id+'_title',x-65,y-46,w+130,35,title,size=17,color=INK,bold=True,align='center')
    if subtitle:s.text(id+'_subtitle',x-65,y+total+13,w+130,29,subtitle,size=16,align='center')


def make_scene():
    s=PublicationScene('D-FINE-S + AQS',W,H,
        'Publication-style architecture of the actual SCB-S D-FINE-S AQS-L2 model. Tensor pyramids and encoder partitions follow HF-D-FINE-style presentation, and token columns/decoder heads follow DETRPose-style presentation. Reference-paper modules are not transplanted.')
    s.text('title',45,26,2200,50,'Overall architecture of D-FINE-S with Adaptive Query Selection (AQS)',size=33,color=INK,bold=True)
    s.text('subtitle',45,83,2200,30,'HGNetV2-B0  /  original Hybrid Encoder  /  AQS after Decoder Layer 2  /  original FDR heads',size=21)
    # Original backbone and three feature levels, not HF-D-FINE's extra P2.
    s.node('backbone',55,295,175,90,'Backbone',['HGNetV2-B0'],fill='#D8E7EB',stroke=LINE,title_size=25,body_size=19)
    s.nodes['backbone'].kind='trapezoid'
    tensor(s,'input',60,555,135,90,'Input RGB','640 x 640')
    s.edge('input_backbone','input','backbone',source_side='top',target_side='bottom',bends=[(135.5,510),(142.5,510)])
    s.group('encoder_bg',290,150,950,700,fill='#F0F1F2',stroke='#AAB0B4')
    s.text('encoder_title',310,162,890,43,'Hybrid Encoder (unchanged)',size=27,color=INK,bold=True,align='center')
    s.text('input_levels',320,214,320,25,'Backbone features',size=18,color=INK,align='center')
    tensor(s,'p5',360,275,65,35,'P5','20 x 20 x 1024')
    tensor(s,'p4',350,480,80,46,'P4','40 x 40 x 512')
    tensor(s,'p3',335,705,100,57,'P3','80 x 80 x 256')
    for id in ['p5','p4','p3']:
        ty=s.get(id).point('left')[1]
        s.edge('bb_'+id,'backbone',id,bends=[(257,340),(257,ty)])
    for id,y in [('conv5',265),('conv4',480),('conv3',705)]:
        s.node(id,485,y,110,55,'1 x 1 Conv',['C = 256'],fill='#FAFAF8',stroke=LINE,title_size=19,body_size=16)
    for p,c in [('p5','conv5'),('p4','conv4'),('p3','conv3')]:s.edge(p+'_'+c,p,c)
    s.node('aifi',730,265,150,70,'AIFI',['P5 self-attention'],fill='#F8E2CB',stroke=LINE,title_size=25,body_size=17)
    s.node('fpn4',730,480,150,70,'FPN fusion',['Concat + CSP'],fill='#F2D7B8',stroke=LINE,title_size=23,body_size=17)
    s.node('fpn3',730,695,150,70,'FPN fusion',['Concat + CSP'],fill='#F2D7B8',stroke=LINE,title_size=23,body_size=17)
    s.node('up54',750,370,110,50,'Up 2x',fill='#FFF9EC',stroke=LINE,title_size=21)
    s.node('up43',750,585,110,50,'Up 2x',fill='#FFF9EC',stroke=LINE,title_size=21)
    s.edge('conv_aifi','conv5','aifi')
    s.edge('conv_fpn4','conv4','fpn4')
    s.edge('conv_fpn3','conv3','fpn3')
    s.edge('aifi_up','aifi','up54',source_side='bottom',target_side='top')
    s.edge('up_fpn4','up54','fpn4',source_side='bottom',target_side='top')
    s.edge('fpn4_up','fpn4','up43',source_side='bottom',target_side='top')
    s.edge('up_fpn3','up43','fpn3',source_side='bottom',target_side='top')
    s.node('pan5',985,265,150,70,'PAN fusion',['Concat + CSP'],fill='#D3E5E7',stroke=LINE,title_size=23,body_size=17)
    s.node('pan4',985,480,150,70,'PAN fusion',['Concat + CSP'],fill='#D3E5E7',stroke=LINE,title_size=23,body_size=17)
    s.node('down34',910,585,90,50,'Down 2x',fill='#EFF6F6',stroke=LINE,title_size=18)
    s.node('down45',1015,385,90,50,'Down 2x',fill='#EFF6F6',stroke=LINE,title_size=18)
    s.edge('aifi_pan5','aifi','pan5')
    s.edge('fpn4_pan4','fpn4','pan4')
    s.edge('fpn3_down','fpn3','down34',bends=[(900,730),(900,610)])
    s.edge('down_pan4','down34','pan4',source_side='right',target_side='bottom',bends=[(1060,610)])
    s.edge('pan4_down','pan4','down45',source_side='top',target_side='bottom')
    s.edge('down_pan5','down45','pan5',source_side='top',target_side='bottom')
    tensor(s,'f5',1160,276,58,36,'F5','256 ch.')
    tensor(s,'f4',1152,490,66,42,'F4','256 ch.')
    tensor(s,'f3',1140,710,78,53,'F3','256 ch.')
    s.edge('pan5_out','pan5','f5');s.edge('pan4_out','pan4','f4');s.edge('fpn3_out','fpn3','f3')
    s.text('encoder_note',490,799,625,30,'Three scales retained: strides 8, 16, 32',size=19,align='center')
    # Schematic columns, NOT one displayed rectangle per actual token/query.
    tokens(s,'memory',1284,354,19,title='Memory',subtitle='Flatten + concat')
    for id in ['f5','f4','f3']:
        sy=s.get(id).point('right')[1]
        s.edge(id+'_memory',id,'memory',bends=[(1252,sy),(1252,504)])
    s.node('selection',1350,389,62,232,'Query selection',fill='#F8D4B2',stroke=LINE,title_size=22)
    s.nodes['selection'].vertical=True
    tokens(s,'queries',1442,448,7,cell_h=12,gap=5,title='Top300',subtitle='Q0')
    s.edge('memory_selection','memory','selection')
    s.edge('selection_queries','selection','queries')
    s.group('decoder_bg',1507,146,898,1000,fill='#FFFFFF',stroke='#B8BDC2',dashed=True)
    s.text('decoder_title',1530,151,810,37,'Decoder and prediction heads',size=26,color=INK,bold=True,align='center')
    # The three-layer S configuration. FDR residuals are on a separate path.
    for layer,y in [(1,295),(2,485),(3,785)]:
        s.node('l'+str(layer),1560,y,170,75,'Decoder Layer '+str(layer),['SA + MSCA + FFN'],fill=STAGES[layer-1],stroke=LINE,title_size=23,body_size=17)
        s.node('fdr'+str(layer),1810,y,150,75,'FDR Head '+str(layer),['Residual logits'],fill=STAGES[layer-1],stroke=LINE,title_size=22,body_size=17)
        s.node('sum'+str(layer),2010,y+21.5,32,32,'+',fill='#FFFFFF',stroke=LINE,ellipse=True,title_size=23)
        s.node('pr'+str(layer),2120,y,180,75,'Softmax',['Pr'+str(layer)+'(n)'],fill='#F9F9F7',stroke=LINE,title_size=22,body_size=20)
        s.edge('l_fdr'+str(layer),'l'+str(layer),'fdr'+str(layer))
        s.edge('fdr_sum'+str(layer),'fdr'+str(layer),'sum'+str(layer))
        s.edge('sum_pr'+str(layer),'sum'+str(layer),'pr'+str(layer))
        s.text('delta'+str(layer),1970,y-1,35,25,'\u0394z'+str(layer),size=17,align='center')
    s.edge('queries_l1','queries','l1',bends=[(1517,503.5),(1517,332.5)])
    # Query initialization and decoder cross-attention use the same memory,
    # but on distinct paths. AQS has no direct memory input.
    for layer,y in [(1,356.5),(2,546.5),(3,846.5)]:
        s.edge('memory_l'+str(layer),'memory','l'+str(layer),target_side='left',target_fraction=.82,
               bends=[(1328,504),(1328,881),(1490,881),(1490,y)],color=base.BLUE)
    s.text('shared_memory_label',1247,893,328,53,'Shared multi-scale memory',size=17,color=base.BLUE,align='center')
    s.edge('l1_l2','l1','l2',source_side='bottom',target_side='top')
    s.node('aqs',1560,625,170,105,'AQS-L2',['Gate + Context','Query residual'],fill='#FFE5CC',stroke=ORANGE,title_size=27,body_size=18)
    s.edge('l2_aqs','l2','aqs',source_side='bottom',target_side='top',color=ORANGE)
    s.edge('aqs_l3','aqs','l3',source_side='bottom',target_side='top',color=ORANGE)
    s.text('refined_queries',1656,746,51,26,'Q2\u2032',size=19,color=ORANGE,align='center')
    s.node('cls2',1760,625,200,85,'ClsHead2 + LQE2',['Gate logits S2'],fill='#FFF8EF',stroke=ORANGE,title_size=22,body_size=18)
    s.edge('l2_cls','l2','cls2',bends=[(1743,522.5),(1743,650.5)],target_fraction=.30,color=ORANGE)
    s.edge('fdr2_quality','fdr2','cls2',source_side='bottom',target_side='top',bends=[(1885,598),(1860,598)])
    s.edge('scores_aqs','cls2','aqs',source_side='bottom',target_side='right',target_fraction=.70,bends=[(1860,740),(1753,740),(1753,698.5)],color=ORANGE)
    s.edge('z1_residual','sum1','sum2',source_side='right',target_side='top',bends=[(2070,332.5),(2070,466),(2026,466)])
    s.edge('z2_residual','sum2','sum3',source_side='right',target_side='top',bends=[(2070,522.5),(2070,750),(2026,750)])
    s.text('z1_label',2076,391,40,30,'z1',size=18)
    s.text('z2_label',2076,625,40,30,'z2',size=18)
    s.text('z0_label',1996,273,60,28,'z0 = 0',size=16,align='center')
    s.text('fdr_refinement_label',2100,687,250,31,'Residual distribution refinement',size=17,align='center')
    # Initial box from L1. No AQS on this branch.
    s.node('traditional',1749,200,181,65,'Traditional Head',['Layer 1 bbox'],fill='#EEEAF4',stroke=LINE,title_size=22,body_size=17)
    s.node('b0',1990,200,160,65,'Initial box b0',fill='#F3F3EC',stroke=LINE,title_size=21)
    s.edge('l1_traditional','l1','traditional',source_side='top',target_side='left',bends=[(1645,232.5)])
    s.edge('traditional_b0','traditional','b0')
    s.node('integral',2120,900,180,90,'FDR integral',['\u03a3 W(n)Pr3(n)','+ initial box b0'],fill='#F4F1E9',stroke=LINE,title_size=23,body_size=18)
    s.edge('pr3_integral','pr3','integral',source_side='bottom',target_side='top')
    s.edge('b0_integral','b0','integral',source_side='right',target_side='right',bends=[(2353,232.5),(2353,945)],dashed=True)
    s.text('b0_label',2359,555,36,30,'b0',size=18)
    s.node('cls3',1810,900,200,90,'ClsHead3 + LQE3',['Final class logits'],fill='#EEEAF4',stroke=LINE,title_size=22,body_size=18)
    s.edge('l3_cls','l3','cls3',bends=[(1765,822.5),(1765,945)])
    s.edge('fdr3_quality','fdr3','cls3',source_side='bottom',target_side='top',bends=[(1885,880),(1910,880)])
    s.node('prediction',2075,1023,225,100,'Predictions',['class / score / bbox','hand-raising, read, write'],fill='#E2EAD8',stroke=LINE,title_size=24,body_size=17)
    s.edge('cls_prediction','cls3','prediction',source_side='right',target_side='left',bends=[(2040,945),(2040,1073)])
    s.edge('box_prediction','integral','prediction',source_side='bottom',target_side='top',bends=[(2210,1008),(2187.5,1008)])
    s.group('legend_aqs',60,956,24,24,fill='#FFE5CC',stroke=ORANGE)
    s.text('legend_aqs_text',98,944,560,46,'Added AQS: query enhancement, not query pruning',size=21,color=INK)
    s.text('legend2',60,1000,1320,36,'FPN / PAN, initial boxes, FDR and original losses are retained.',size=21)
    s.text('legend3',60,1048,1320,32,'L2 FDR is computed before AQS; updated queries then reach L3.',size=20)
    s.text('legend4',60,1094,1320,30,'Feature maps and token columns are schematic; no performance values are illustrated.',size=18)
    s.text('caption',60,1147,2250,28,'Inference structure of the SCB-S three-class model. DN and training-only auxiliary supervision are omitted.',size=20,color=INK)
    s.validate()
    return s


def pdf_node(c,n):
    if not n.text_only:
        c.setFillColor(n.fill);c.setStrokeColor(n.stroke);c.setLineWidth(1.6);c.setDash([7,5] if n.dashed else [])
        pts=polygon_points(n)
        if pts:
            p=c.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
            for x,y in pts[1:]:p.lineTo(x,H-y)
            p.close();c.drawPath(p,fill=1,stroke=1)
        elif n.ellipse:c.ellipse(n.x,H-n.y-n.h,n.x+n.w,H-n.y,fill=1,stroke=1)
        elif getattr(n,'square',False):c.rect(n.x,H-n.y-n.h,n.w,n.h,fill=1,stroke=1)
        else:c.roundRect(n.x,H-n.y-n.h,n.w,n.h,10,fill=1,stroke=1)
    if getattr(n,'vertical',False):
        text,size,bold,color=n.lines[0];c.saveState();c.translate(n.x+n.w/2,H-n.y-n.h/2);c.rotate(90)
        c.setFillColor(color);c.setFont('DiagramBold',size);c.drawCentredString(0,-size*.33,text);c.restoreState();return
    heights=[size*1.5 for _,size,_,_ in n.lines];y=n.y+(n.h-sum(heights))/2
    for (text,size,bold,color),lh in zip(n.lines,heights):
        y+=lh/2;c.setFillColor(color);c.setFont('DiagramBold' if bold else 'Diagram',size)
        baseline=H-y-size*.33
        if n.text_only and n.align=='left':c.drawString(n.x,baseline,text)
        else:c.drawCentredString(n.x+n.w/2,baseline,text)
        y+=lh/2


def pdf_edges(c,scene):
    for e in scene.edges:
        pts=scene.points(e);c.setStrokeColor(e.color);c.setLineWidth(2.2);c.setDash([7,5] if e.dashed else [])
        p=c.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
        for x,y in pts[1:]:p.lineTo(x,H-y)
        c.drawPath(p)
        if e.arrow:
            x,y=pts[-1];px,py=pts[-2];a=math.atan2(y-py,x-px);ux,uy=math.cos(a),math.sin(a)
            ax,ay=x-9*ux,y-9*uy;p=c.beginPath();p.moveTo(x,H-y)
            p.lineTo(ax-4.5*uy,H-(ay+4.5*ux));p.lineTo(ax+4.5*uy,H-(ay-4.5*ux));p.close()
            c.setFillColor(e.color);c.drawPath(p,fill=1,stroke=0)


def export(scene):
    svg=scene.svg();ET.fromstring(svg)
    (OUT/'dfine_s_aqs_publication_style.svg').write_text(svg,encoding='utf-8')
    root=ET.Element('mxfile',host='app.diagrams.net',type='device');scene.drawio(root);ET.indent(root,space='  ')
    ET.ElementTree(root).write(OUT/'dfine_s_aqs_publication_style.drawio',encoding='utf-8',xml_declaration=True)
    pdfmetrics.registerFont(TTFont('Diagram','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
    pdfmetrics.registerFont(TTFont('DiagramBold','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))
    c=canvas.Canvas(str(PDF_OUT),pagesize=(W,H));c.setTitle('D-FINE-S + AQS: publication-style architecture')
    for n in scene.groups.values():pdf_node(c,n)
    pdf_edges(c,scene)
    for n in scene.nodes.values():pdf_node(c,n)
    c.showPage();c.save()
    r=PdfReader(PDF_OUT);assert len(r.pages)==1
    assert '/XObject' not in r.pages[0]['/Resources'], 'This figure must not contain embedded raster images.'
    text=r.pages[0].extract_text();assert 'AQS-L2' in text and 'Hybrid Encoder' in text
    d=pdfium.PdfDocument(str(PDF_OUT));d[0].render(scale=2).to_pil().save(OUT/'dfine_s_aqs_publication_style.png')
    print('Native vector SVG/draw.io:',OUT)
    print('Vector PDF:',PDF_OUT)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);PDF_OUT.parent.mkdir(parents=True,exist_ok=True)
    export(make_scene())
