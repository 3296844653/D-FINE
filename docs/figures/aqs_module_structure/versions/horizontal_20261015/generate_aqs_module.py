"""Native editable diagram of the current legacy AQS-L2 implementation."""
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

W,H=1800,900
INK='#202020'
LINE='#414141'
BLUE=ref.base.BLUE
GATE=ref.base.ORANGE


def node(s,id,x,y,w,h,title,body=(),fill='#FFFFFF',size=24,body_size=18):
    return ref.node(s,id,x,y,w,h,title,body,fill=fill,size=size,body_size=body_size)


def edge(s,id,src,dst,**kwargs):
    kwargs.setdefault('color',ref.base.LINE)
    s.edge(id,src,dst,**kwargs)


def build():
    s=ref.ReferenceScene('AQS internal structure',W,H,
        'Source-checked AQS-L2: class confidence, learned threshold, hard forward '
        'gate with straight-through training gradients, selected raw-query mean, '
        'LayerNorm and context concatenation, MLP, gated residual update. '
        'The number of queries is unchanged. No direct feature-map or bbox input.')
    ref.text(s,'title',40,20,1720,50,'Adaptive Query Selection (AQS)',size=32,bold=True)
    ref.text(s,'placement',40,70,1720,32,'Decoder L2  →  AQS  →  Decoder L3',size=21)
    s.group('aqs_frame',230,110,1415,635,fill='#F2F3F3',stroke='#D2D6D6')
    s.groups['aqs_frame'].radius=28

    node(s,'logits',40,170,170,110,'L2 logits S2',
         ['ClsHead2 + LQE2','N x C'],fill='#FFF0C7',size=23,body_size=18)
    node(s,'confidence',260,170,200,110,'Class Confidence',
         ['s_i = max_c σ(S2,i,c)'],fill='#FFF0C7',size=23,body_size=18)
    node(s,'gate',510,155,235,140,'Hard Gate + STE',
         ['h_i = 1[s_i ≥ τ]','g_s = σ((s_i − τ)/T)','forward h; backward g_s'],
         fill='#F7D3CE',size=24,body_size=17)
    node(s,'threshold',510,340,235,105,'Learned Threshold',
         ['τ = σ(θ)','T = 0.10  (fixed)'],fill='#FFF0C7',size=23,body_size=18)
    node(s,'pool',810,155,290,140,'Selected-query Mean',
         ['c = Σ(g_i q_i) / max(Σg_i, 1)','Broadcast c to N queries'],
         fill='#DCE9DF',size=23,body_size=17)

    node(s,'query',40,500,170,110,'L2 query Q2',
         ['300 x 256'],fill='#E0EAF5',size=24,body_size=20)
    node(s,'norm',260,500,200,110,'LayerNorm',
         ['LN(q_i)'],fill='#E0EAF5',size=25,body_size=20)
    node(s,'concat',810,500,190,110,'Concat',
         ['[LN(q_i); c]','2D = 512'],fill='#E0EAF5',size=25,body_size=18)
    node(s,'mlp',1050,500,230,110,'Fusion MLP',
         ['Linear → SiLU → Linear','512 → 256 → 256'],
         fill='#E0EAF5',size=25,body_size=17)
    node(s,'scale',1415,340,180,90,'Learned Scale',
         ['β = tanh(ρ)'],fill='#FFF0C7',size=23,body_size=20)
    node(s,'residual',1325,500,210,110,'Gated Residual',
         ['β · g_i · Δq_i'],fill='#F7D3CE',size=23,body_size=21)
    n=node(s,'sum',1590,530,50,50,'+',fill='#FFFFFF',size=32)
    n.ellipse=True
    node(s,'output',1675,500,90,110,'Q2′',
         ['300 x 256'],fill='#F7D3CE',size=27,body_size=14)
    node(s,'l3',1655,665,130,74,'Decoder L3',
         ['next layer'],fill='#E0EAF5',size=19,body_size=16)

    edge(s,'logits_confidence','logits','confidence')
    edge(s,'confidence_gate','confidence','gate')
    edge(s,'threshold_gate','threshold','gate',source_side='top',target_side='bottom',color=GATE)
    edge(s,'gate_pool','gate','pool',color=GATE)
    edge(s,'gate_residual','gate','residual',source_side='right',target_side='top',
         target_fraction=.2,bends=[(775,225),(775,130),(1300,130),(1300,455),(1367,455)],color=GATE)
    ref.text(s,'gate_branch_label',1310,245,100,29,'gate g',size=17,color=GATE)
    edge(s,'raw_query_pool','query','pool',source_side='top',target_side='bottom',
         target_fraction=.15,bends=[(125,460),(853.5,460)],color=BLUE)
    ref.text(s,'raw_query_label',285,432,235,26,'raw query features',size=17,color=BLUE)
    edge(s,'pool_concat','pool','concat',source_side='bottom',target_side='top',
         bends=[(955,430),(905,430)],color=BLUE)
    ref.text(s,'context_label',966,351,190,28,'pooled context c',size=17,color=BLUE)
    edge(s,'query_norm','query','norm',color=BLUE)
    edge(s,'norm_concat','norm','concat',color=BLUE)
    edge(s,'concat_mlp','concat','mlp')
    edge(s,'mlp_residual','mlp','residual')
    edge(s,'scale_residual','scale','residual',source_side='bottom',target_side='top',
         source_fraction=78/180,target_fraction=.8,color=GATE)
    edge(s,'residual_sum','residual','sum',color=GATE)
    edge(s,'identity','query','sum',source_side='bottom',target_side='bottom',
         bends=[(125,690),(1615,690)],color=BLUE)
    ref.text(s,'identity_label',470,700,775,27,
             'Identity path: g_i = 0 leaves the original query unchanged',size=18,color=BLUE)
    edge(s,'sum_output','sum','output',color=GATE)
    edge(s,'output_l3','output','l3',source_side='bottom',target_side='top')

    ref.text(s,'equation',300,763,1200,40,
             'q_i′ = q_i + tanh(ρ) · g_i · MLP([LN(q_i); c])',size=26,bold=True)
    ref.text(s,'settings',40,813,1720,27,
             'N = 300, D = 256.  Initial τ = 0.50, initial ρ = 0.10; T = 0.10 is fixed.  Query count is unchanged.',size=18)
    ref.text(s,'training_note',40,850,1720,27,
             'Training: STE gate gradients; ordinary and DN queries pool separately. The MLP output layer is zero-initialized.',size=17)

    ids=set(s.nodes)|set(s.groups)
    for e in s.edges:
        assert e.source in ids and e.target in ids
        for (x1,y1),(x2,y2) in zip(s.points(e),s.points(e)[1:]):
            assert abs(x1-x2)<1e-7 or abs(y1-y2)<1e-7,e.id
    assert {e.target for e in s.edges if e.source=='gate'}=={'pool','residual'}
    assert {e.source for e in s.edges if e.target=='sum'}=={'residual','query'}
    s.validate()
    return s


def export(s):
    OUT.mkdir(parents=True,exist_ok=True);PDF.parent.mkdir(parents=True,exist_ok=True)
    ref.W,ref.H=W,H;ref.pub.W,ref.pub.H=W,H
    pdfmetrics.registerFont(TTFont('Diagram','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
    pdfmetrics.registerFont(TTFont('DiagramBold','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))
    for n in s.nodes.values():
        for label,size,bold,_ in n.lines:
            assert pdfmetrics.stringWidth(label,'DiagramBold' if bold else 'Diagram',size)<=n.w-2,n.id
        assert sum(size*1.5 for _,size,_,_ in n.lines)<=n.h+1e-7,n.id
    svg=s.svg();ET.fromstring(svg);assert '<image' not in svg
    (OUT/'aqs_module_structure.svg').write_text(svg,encoding='utf-8')
    root=ET.Element('mxfile',host='app.diagrams.net',type='device')
    s.drawio(root);ref.verify_drawio(root);ET.indent(root,space='  ')
    ET.ElementTree(root).write(OUT/'aqs_module_structure.drawio',encoding='utf-8',xml_declaration=True)
    c=canvas.Canvas(str(PDF),pagesize=(W,H))
    c.setTitle('AQS-L2 internal structure - source-checked editable diagram')
    for n in s.groups.values():ref.pdf_node(c,n)
    ref.pdf_edges(c,s)
    for n in s.nodes.values():ref.pdf_node(c,n)
    c.showPage();c.save()
    r=PdfReader(PDF);assert len(r.pages)==1
    assert '/XObject' not in r.pages[0]['/Resources']
    for label in ('LayerNorm','Selected-query Mean','Hard Gate + STE','Fusion MLP','Decoder L3'):
        assert label in r.pages[0].extract_text(),label
    pdfium.PdfDocument(str(PDF))[0].render(scale=2).to_pil().save(OUT/'aqs_module_structure.png')
    print('Editable draw.io:',OUT/'aqs_module_structure.drawio')
    print('Vector PDF:',PDF)


if __name__=='__main__':
    export(build())
