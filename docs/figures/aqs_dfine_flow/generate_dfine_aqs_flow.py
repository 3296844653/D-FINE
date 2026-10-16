"""Draw an editable, source-checked D-FINE-S/AQS-L2 inference overview.

The user's reference supplies presentation, not model components. Encoder
FPN/PAN boxes summarize complete multi-scale stages; the actual operators,
four fusion blocks, initial boxes, and training-only paths are not expanded.
"""
from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pypdfium2 as pdfium
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
STEM = 'dfine_s_aqs_flow'
PDF_OUT = PROJECT / 'output/pdf' / (STEM + '.pdf')
sys.path.insert(0, str(OUT.parent / 'aqs_reference_style'))
import generate_reference_figure as ref

base, pub = ref.base, ref.pub
W, H = 2900, 960
ref.W = pub.W = W
ref.H = pub.H = H
node, text, plane, tokens, classroom = ref.node, ref.text, ref.plane, ref.tokens, ref.classroom
GREEN = '#DDEAD2'
BLUE = '#C8D9EF'
CREAM = '#FFF6DB'
INK = ref.INK


def bus(s, id, x, y, h):
    n = node(s,id,x,y,2,h,fill=base.LINE)
    n.lines=[];n.radius=0;n.no_stroke=True


def edge(s,id,src,dst,**kwargs):
    kwargs.setdefault('color',base.LINE)
    s.edge(id,src,dst,**kwargs)


def make_scene():
    s = ref.ReferenceScene('D-FINE-S + AQS-L2 - source-checked flow', W,H,
        'Reference-style native-vector inference overview of the actual three-layer D-FINE-S AQS-L2 model. '
        'P5 AIFI and original FPN/PAN produce three-scale memory. Original score-based Top300 initializes '
        'queries. L2 original classifier/LQE supplies AQS gate logits, and L2 FDR is computed before AQS. '
        'Refined query content reaches L3; all 300 ordinary queries remain. Classroom tiles are schematic.')
    classroom(s,'input',50,210,225,180)
    text(s,'input_label',40,403,245,34,'Input image',size=25)
    n = node(s,'backbone',65,562,195,88,'Backbone',fill='#FFF0B7',size=29)
    n.kind='trapezoid'
    edge(s,'input_backbone','input','backbone',source_side='bottom',target_side='top')

    # Same high-level encoder flow as the reference, but the concrete stages
    # are D-FINE's projection, P5 attention, RepNCSPELAN/SCDown FPN/PAN.
    s.group('encoder',310,80,1115,730,fill='#F2F2F2',stroke='#F2F2F2')
    s.groups['encoder'].radius=48;s.groups['encoder'].no_stroke=True
    before=set(s.nodes)|set(s.groups)
    text(s,'encoder_title',337,94,1058,49,'Hybrid Encoder',size=34,bold=True)
    plane(s,'p5',454,535,103,29,'P5')
    plane(s,'p4',432,584,156,35,'P4')
    plane(s,'p3',409,638,216,42,'P3')
    node(s,'projection',425,720,220,55,'Projection',fill='#EDF1EA',size=25)
    edge(s,'projection_features','projection','p3',source_side='top',source_fraction=95.5/220,
         target_side='bottom')
    tokens(s,'image_tokens',395,435,10,width=18,height=20,gap=6,horizontal=True)
    # Keep positional input beside the feature path, not in series with P5.
    tokens(s,'pos_tokens',340,420,4,width=18,height=18,gap=7)
    s.editable_groups.pop('pos_tokens')  # The entire Encoder is the movable group.
    for i in range(10):
        s.nodes[f'image_tokens_tile{i}'].fill=GREEN
    for i in range(4):
        s.nodes[f'pos_tokens_tile{i}'].fill=BLUE
    node(s,'aifi',369,322,294,68,'AIFI',fill='#DDEBE4',size=32)
    edge(s,'p5_tokens','p5','image_tokens',source_side='top',target_side='bottom',
         target_fraction=114/234)
    edge(s,'tokens_aifi','image_tokens','aifi',source_side='top',target_side='bottom',
         target_fraction=143/294)
    edge(s,'positions_aifi','pos_tokens','aifi',source_side='left',target_side='left',
         color=base.BLUE,bends=[(327,466.5),(327,356)])
    plane(s,'f5',457,193,104,31,'F5')
    edge(s,'aifi_f5','aifi','f5',source_side='top',source_fraction=143.5/294,target_side='bottom')

    s.group('fusion_bg',760,155,610,545,fill='#DFEBE5',stroke='#C6D7CC')
    s.groups['fusion_bg'].radius=16
    text(s,'fusion_title',777,169,576,44,'Cross-scale Fusion',size=29,bold=True)
    bus(s,'fusion_inputs',708,209,450.5)
    edge(s,'f5_bundle','f5','fusion_inputs',target_fraction=0,arrow=False)
    edge(s,'p4_bundle','p4','fusion_inputs',target_fraction=393/450.5,arrow=False)
    edge(s,'p3_bundle','p3','fusion_inputs',target_fraction=1,arrow=False)
    node(s,'fpn',795,375,185,90,'FPN',fill='#FBEAD9',size=31)
    node(s,'pan',1075,375,185,90,'PAN',fill='#E3EFF3',size=31)
    edge(s,'bundle_fpn','fusion_inputs','fpn',source_fraction=211/450.5)
    edge(s,'fpn_pan','fpn','pan')
    s.editable_groups['encoder']=sorted((set(s.nodes)|set(s.groups))-before)
    edge(s,'backbone_projection','backbone','projection',source_side='bottom',target_side='left',
         bends=[(162.5,747.5)])

    n=node(s,'concat',1460,400,40,40,'C',size=28);n.ellipse=True
    edge(s,'pyramid_concat','pan','concat')
    tokens(s,'memory',1533,234.5,14)
    text(s,'memory_label',1487,628,120,35,'Memory',size=24)
    edge(s,'concat_memory','concat','memory')
    n=node(s,'query_selection',1589,235,65,370,'Query Selection',fill='#FFF0BD',size=29)
    n.vertical=True
    edge(s,'memory_selection','memory','query_selection')
    tokens(s,'queries',1685,329,7)
    text(s,'query_count',1656,531,94,34,'Top300',size=24)
    edge(s,'selection_queries','query_selection','queries')

    # Explicit actual AQS location and score input. No uncertainty-minimal
    # selection, extra feature sampler, final-only AQS, or query pruning.
    s.group('decoder',1770,112,615,708,fill='#E1EBF7',stroke='#C7D5E6')
    s.groups['decoder'].radius=16
    before=set(s.nodes)|set(s.groups)
    text(s,'decoder_title',1794,131,568,46,'Decoder & Heads',size=31,bold=True)
    node(s,'l1',1810,218,180,65,'Decoder L1',fill='#FFFFFF',size=26)
    node(s,'l2',1810,325,180,65,'Decoder L2',fill='#FFFFFF',size=26)
    node(s,'aqs',1810,442,180,75,'AQS',fill='#FFF3DC',size=32)
    node(s,'l3',1810,610,180,65,'Decoder L3',fill='#FFFFFF',size=26)
    for src,dst in [('l1','l2'),('l2','aqs'),('aqs','l3')]:
        edge(s,src+'_'+dst,src,dst,source_side='bottom',target_side='top')
    node(s,'fdr2',2160,325,150,65,'FDR2',fill='#F2F5FA',size=26)
    node(s,'score2',2075,442,235,75,'Cls2 + LQE2',fill='#FFFFFF',size=27)
    edge(s,'l2_fdr2','l2','fdr2')
    edge(s,'l2_score2','l2','score2',source_fraction=.80,
         bends=[(2035,377),(2035,479.5)])
    edge(s,'fdr2_quality','fdr2','score2',source_side='bottom',target_side='top',target_fraction=160/235)
    edge(s,'scores_aqs','score2','aqs',source_side='bottom',target_side='right',target_fraction=.75,
         bends=[(2192.5,565),(2025,565),(2025,498.25)])

    node(s,'fdr3',2160,610,150,65,'FDR3',fill='#F2F5FA',size=26)
    node(s,'score3',2075,732,235,65,'Cls3 + LQE3',fill='#FFFFFF',size=27)
    edge(s,'l3_fdr3','l3','fdr3')
    edge(s,'l3_score3','l3','score3',source_fraction=.80,
         bends=[(2035,662),(2035,764.5)])
    edge(s,'fdr3_quality','fdr3','score3',source_side='bottom',target_side='top',target_fraction=160/235)
    edge(s,'fdr_state','fdr2','fdr3',source_side='right',target_side='top',dashed=True,
         bends=[(2350,357.5),(2350,581),(2235,581)])
    s.editable_groups['decoder']=sorted((set(s.nodes)|set(s.groups))-before)
    edge(s,'queries_l1','queries','l1',bends=[(1746,420),(1746,250.5)])
    # One shared memory bus branches to each decoder MSCA, not to AQS itself.
    bus(s,'decoder_memory',1790,257,392)
    edge(s,'memory_decoder','memory','decoder_memory',source_side='top',target_side='top',
         color=base.BLUE,bends=[(1545,192),(1791,192)])
    for id,y in [('l1',257),('l2',364),('l3',649)]:
        edge(s,'memory_'+id,'decoder_memory',id,source_fraction=(y-257)/392,
             target_fraction=.60,color=base.BLUE)
    s.editable_groups['decoder'].append('decoder_memory')

    # Final score and bbox bundles (the FDR3 box includes its original
    # integral/distance2bbox decode at this overview level).
    bus(s,'prediction_bundle',2415,642.5,122)
    edge(s,'boxes_bundle','fdr3','prediction_bundle',target_fraction=0,arrow=False)
    edge(s,'scores_bundle','score3','prediction_bundle',target_fraction=1,arrow=False)
    tokens(s,'output_tokens',2470,612.5,7)
    edge(s,'bundle_predictions','prediction_bundle','output_tokens')
    classroom(s,'output',2560,599.5,250,208,detected=True)
    text(s,'output_label',2529,820,312,37,'Output (schematic)',size=25)
    edge(s,'predictions_output','output_tokens','output')

    # Color legend copied as a convention, not as extra model operations.
    for id,x,fill,label in [('image_key',470,GREEN,'Image feature'),
                            ('position_key',865,BLUE,'Position embedding'),
                            ('query_key',1370,CREAM,'Object query')]:
        n=node(s,id,x,855,25,25,fill=fill);n.lines=[];n.radius=0
        text(s,id+'_text',x+42,848,365,40,label,size=25,align='left')
    text(s,'title',310,911,2260,42,'D-FINE-S with AQS-L2',size=32,bold=True)
    validate_flow(s)
    return s


def validate_flow(s):
    s.validate()
    ids=set(s.nodes)|set(s.groups)
    assert {'projection','p3','p4','p5','aifi','fpn','pan','l1','l2','aqs','l3','fdr2','score2','fdr3','score3'} <= ids
    pairs={(e.source,e.target) for e in s.edges}
    assert {('l1','l2'),('l2','aqs'),('aqs','l3'),('score2','aqs'),
            ('l2','fdr2'),('fdr2','score2'),('l3','fdr3'),('fdr3','score3')} <= pairs
    assert ('memory','aqs') not in pairs and ('aifi','pan') not in pairs
    for e in s.edges:
        pts=s.points(e)
        for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
            assert abs(x1-x2)<1e-7 or abs(y1-y2)<1e-7,(e.id,(x1,y1),(x2,y2))
    assert s.get('fpn').point('right')[1] == s.get('pan').point('left')[1]
    assert len([e for e in s.edges if (e.source,e.target)==('fpn','pan')])==1


def verify_source():
    cfg=(PROJECT/'configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml').read_text()
    dec=(PROJECT/'src/zoo/dfine/dfine_decoder.py').read_text()
    enc=(PROJECT/'src/zoo/dfine/hybrid_encoder.py').read_text()
    small=(PROJECT/'configs/dfine/dfine_hgnetv2_s_coco.yml').read_text()
    assert 'aqs_apply_layer: 2' in cfg and 'num_layers: 3' in small
    assert 'torch.topk(outputs_logits.max(-1).values' in dec
    assert 'scores = self.lqe_layers[i](scores, pred_corners)' in dec
    assert 'output = refined_cls_output' in dec
    assert 'return query + scale * gate.unsqueeze(-1)' in dec
    assert 'use_encoder_idx=[2]' in enc and 'outs = [inner_outs[0]]' in enc
    assert 'RepNCSPELAN4(' in enc and 'SCDown(hidden_dim' in enc


def export(s):
    OUT.mkdir(parents=True,exist_ok=True);PDF_OUT.parent.mkdir(parents=True,exist_ok=True)
    svg=s.svg();ET.fromstring(svg)
    assert '<image' not in svg
    assert not any(t in svg for t in ['Uncertainty-minimal','UCS-Fusion','DCS-Fusion','CAF'])
    (OUT/(STEM+'.svg')).write_text(svg,encoding='utf-8')
    root=ET.Element('mxfile',host='app.diagrams.net',type='device')
    s.drawio(root);ref.verify_drawio(root);ET.indent(root,space='  ')
    ET.ElementTree(root).write(OUT/(STEM+'.drawio'),encoding='utf-8',xml_declaration=True)
    pdfmetrics.registerFont(TTFont('Diagram','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
    pdfmetrics.registerFont(TTFont('DiagramBold','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))
    c=canvas.Canvas(str(PDF_OUT),pagesize=(W,H))
    c.setTitle('D-FINE-S + AQS-L2: source-checked architecture')
    for n in s.groups.values():ref.pdf_node(c,n)
    pub.pdf_edges(c,s)
    for n in s.nodes.values():ref.pdf_node(c,n)
    c.showPage();c.save()
    r=PdfReader(PDF_OUT);assert len(r.pages)==1
    assert '/XObject' not in r.pages[0]['/Resources']
    content=r.pages[0].extract_text()
    for word in ['AQS','AIFI','FPN','PAN','Query Selection','Decoder L2','Decoder L3','FDR2','FDR3']:
        assert word in content,word
    pdf=pdfium.PdfDocument(str(PDF_OUT))
    pdf[0].render(scale=1.6).to_pil().save(OUT/(STEM+'.png'))
    print('Verified D-FINE-S/AQS-L2 source and orthogonal flow.')
    print('Editable:',OUT/(STEM+'.drawio'))
    print('Vector PDF:',PDF_OUT)


if __name__=='__main__':
    verify_source()
    export(make_scene())
