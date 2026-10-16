"""Full reference-style relayout, retaining native editable D-FINE + AQS.

Paper artwork is not embedded. The classroom scenes are vector schematics.
Only figure files are generated; the training implementation is not changed.
"""
from __future__ import annotations

import copy
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import pypdfium2 as pdfium
from PIL import Image, ImageChops
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

import generate_reference_figure as ref

OUT=Path(__file__).resolve().parent
PDF_OUT=ref.PDF_OUT
STEM=ref.STEM
W,H=1842,620
LINE=ref.base.LINE
INK=ref.INK

original_polygon=ref.pub.polygon_points


def flow_polygon(n):
    if getattr(n,'kind','')=='inverted_trapezoid':
        return [(n.x,n.y),(n.x+n.w,n.y),
                (n.x+.85*n.w,n.y+n.h),(n.x+.15*n.w,n.y+n.h)]
    return original_polygon(n)


class FlowScene(ref.ReferenceScene):
    def svg(self):
        svg=super().svg()
        leaders=[]
        for e in self.edges:
            if not getattr(e,'curved',False):continue
            p,a,b,q=self.points(e)
            d=f'M{p[0]:g} {p[1]:g} C{a[0]:g} {a[1]:g} {b[0]:g} {b[1]:g} {q[0]:g} {q[1]:g}'
            path=f'<path id="{e.id}" d="{d}" fill="none" stroke="{e.color}" stroke-width="2.2"/>'
            # Draw output leaders above the schematic image, as in the
            # reference, instead of letting classroom shapes cover them.
            svg=re.sub(rf'<polyline id="{e.id}"[^>]+/>','',svg)
            leaders.append(path)
        svg=svg.replace('</g>', '\n'.join(leaders)+'</g>')
        return svg

    def drawio(self,parent):
        super().drawio(parent)
        root=parent.findall('diagram')[-1].find('mxGraphModel/root')
        cells={c.attrib['id']:c for c in root.findall('mxCell')}
        b=cells['backbone']
        b.set('style','shape=trapezoid;size=0.15;flipV=1;'+b.attrib['style'].replace('rounded=1;',''))
        for e in self.edges:
            if getattr(e,'curved',False):
                cell=cells[e.id]
                cell.set('style',cell.attrib['style']+'curved=1;')


def build():
    # The previous encoder is reused as native nodes and lines, not a bitmap.
    old=ref.make_scene()
    s=FlowScene('D-FINE-S + AQS - compact reference flow',W,H,
        'Compact reference-style overview. Original D-FINE Hybrid Encoder and '
        'Top300 query initialization are retained. AQS refines query content '
        'after decoder layer 2 and before layer 3. Fusion containers summarize '
        'three scale states, not the number of executable fusion operators. '
        'Classroom images and colored output links are schematic, not measured predictions.')
    members=set(old.editable_groups['encoder'])
    dx,dy=-200,-21.2
    # Preserve the source drawing order: background first, then fusion frame
    # and stacked maps. A set iteration would obscure maps nondeterministically.
    for source,target in [(old.groups,s.groups),(old.nodes,s.nodes)]:
        for id,original in source.items():
            if id not in members|{'encoder'}:continue
            n=copy.deepcopy(original);n.x+=dx;n.y+=dy
            target[id]=n
    s.editable_groups['encoder']=sorted(members)
    for edge in old.edges:
        if edge.source in members and edge.target in members:
            e=copy.deepcopy(edge)
            e.bends=[(x+dx,y+dy) for x,y in e.bends]
            if hasattr(e,'jump_at'):
                e.jump_at=(e.jump_at[0]+dx,e.jump_at[1]+dy)
            s.edges.append(e)
    s.nodes['encoder_title'].lines=[('Efficient Hybrid Encoder',27.2,True,INK)]
    axis=s.nodes['concat'].point('right')[1]

    # Reference-like image above a downward-tapered, pale yellow backbone.
    ref.classroom(s,'input',40,170,220,160)
    ref.text(s,'input_label',40,336,220,24,'Input image',size=20)
    b=ref.node(s,'backbone',65,385,170,84,'Backbone',fill='#FFE7A1',size=26)
    b.kind='inverted_trapezoid'
    s.edge('input_backbone','input','backbone',source_side='bottom',target_side='top',color=LINE)
    p3=s.groups['p3_plane0'];y=p3.y+p3.h
    # With an inverted trapezoid the physical lower-right corner is at 85%.
    s.edge('backbone_levels','backbone','p3',source_side='bottom',source_fraction=.85,
           target_side='left',target_fraction=(y-s.get('p3').y)/s.get('p3').h,
           bends=[(209.5,y)],color=LINE)

    # Slim token columns and vertical bars, with correct original selection.
    ref.tokens(s,'memory',950,axis-192,15,width=21,height=20,gap=6)
    ref.text(s,'memory_label',913,axis+201,96,26,'Memory',size=19)
    ref.node(s,'selection',1008,70,48,426.4,'Query selection',fill='#FFE7A1',size=24)
    s.nodes['selection'].vertical=True
    ref.tokens(s,'queries',1095,axis-126.5,10,width=21,height=19,gap=7)
    ref.text(s,'query_label',1066,axis+139,80,25,'Top300',size=20)
    sf=(axis-70)/426.4
    s.edge('concat_memory','concat','memory',color=LINE)
    s.edge('memory_selection','memory','selection',target_fraction=sf,color=LINE)
    s.edge('selection_queries','selection','queries',source_fraction=sf,color=LINE)

    s.group('decoder',1150,70,170,426.4,fill='#C9D9EC',stroke='#AEBFD5')
    s.groups['decoder'].radius=8
    d=ref.node(s,'decoder_label',1154,104,33,329,'Decoder & Head',fill='#C9D9EC',size=22)
    d.vertical=True;d.no_stroke=True
    decoder_members=['decoder_label']
    for id,label,y in [('l1','Layer 1',106),('l2','Layer 2',186),
                       ('aqs','AQS',266),('l3','Layer 3',346)]:
        ref.node(s,id,1194,y,108,46,label,fill='#FFFFFF',size=22)
        decoder_members.append(id)
    ref.text(s,'heads',1194,438,108,30,'Head',size=24,bold=True)
    decoder_members.append('heads')
    s.editable_groups['decoder']=decoder_members
    for a,b in [('l1','l2'),('l2','aqs'),('aqs','l3'),('l3','heads')]:
        s.edge(a+'_'+b,a,b,source_side='bottom',target_side='top',color=LINE)
    s.edge('queries_decoder','queries','decoder',target_fraction=sf,color=LINE)
    # Cross-attention uses the same memory; AQS itself has no map input.
    s.edge('shared_memory','memory','decoder',source_side='top',target_side='top',
           target_fraction=.62,bends=[(960.5,43),(1255.4,43)],color=LINE)
    ref.tokens(s,'detections',1366,axis-101,8,width=21,height=20,gap=6,colored=True)
    s.edge('decoder_predictions','decoder','detections',source_fraction=sf,color=LINE)

    ref.classroom(s,'output',1455,218,470,255,detected=True)
    ref.text(s,'output_label',1455,479,470,26,'Detection output (schematic)',size=20)
    # These colored leaders are visual correspondences, not extra branches.
    for tile,box,color,c1,c2 in [(1,0,'#B77D3B',(1470,249.8),(1510,268)),
                               (3,1,'#6487A4',(1510,301.8),(1630,274)),
                               (6,2,'#789763',(1490,453),(1895,472))]:
        side='top' if box<2 else 'right'
        s.edge('prediction_link'+str(box),'detections_tile'+str(tile),'output_box'+str(box),
               target_side=side,target_fraction=.5 if box<2 else .8,
               bends=[c1,c2],color=color,arrow=False)
        s.edges[-1].curved=True

    # Adapt the reference's operator legends to D-FINE's actual operators.
    for id,x,color,rows in [('yellow',1470,'#FFE7A1',
                           ['1 x 1 Conv','Norm + Act','Upsample']),
                          ('blue',1715,'#BFD0E9',
                           ['SCDown','1 x 1 Conv','DW 3 x 3, s=2'])]:
        icon=ref.node(s,id+'_icon',x,104.5,24,24,fill=color)
        icon.lines=[];icon.radius=4
        bx=x+40
        s.group(id+'_legend',bx,70,175,93,fill=color,stroke='#AAB2BA')
        s.groups[id+'_legend'].radius=5
        for j,label in enumerate(rows):
            ref.text(s,id+'_legend_row'+str(j),bx+4,72+j*29,167,27,label,size=17,bold=True)
        s.edge(id+'_legend_link',id+'_icon',id+'_legend',color=LINE)

    s.group('legend',1603,511,322,94,fill='#FFFFFF',stroke='#7B8791',dashed=True)
    s.groups['legend'].radius=3
    for i,(label,color) in enumerate([('Position Embedding','#BCD0E9'),
                                     ('Image Feature','#BDD99E'),
                                     ('Object Query','#FFF6DB')]):
        n=ref.node(s,'legend_key'+str(i),1618,521+i*26,18,18,fill=color)
        n.lines=[];n.radius=0
        ref.text(s,'legend_text'+str(i),1648,516+i*26,264,27,label,size=20,bold=True,align='left')
    ref.text(s,'title',280,551,1040,35,'D-FINE-S with Adaptive Query Selection (AQS)',size=25,bold=True)

    shrink_aifi_region(s)
    narrow_encoder(s)
    align_input_backbone(s)
    enlarge_aifi_region(s)

    # Confirm real decoder placement and native encoder connections.
    assert s.nodes['l2'].y<s.nodes['aqs'].y<s.nodes['l3'].y
    assert next(iter(s.groups))=='encoder', 'Encoder background must be behind its maps'
    assert next(e.target for e in s.edges if e.id=='l2_aqs')=='aqs'
    assert next(e.target for e in s.edges if e.id=='aqs_l3')=='l3'
    assert not any(e.target=='aqs' and e.source=='memory' for e in s.edges)
    assert 'Uncertainty' not in ' '.join(t for n in s.nodes.values() for t,*_ in n.lines)
    for e in s.edges:
        if getattr(e,'feature_grid',False) or getattr(e,'curved',False):continue
        for (x1,y1),(x2,y2) in zip(s.points(e),s.points(e)[1:]):
            assert abs(x1-x2)<1e-7 or abs(y1-y2)<1e-7,e.id
    s.validate()
    return s


def shrink_aifi_region(s):
    """Scale only the annotated feature/AIFI assembly, not the CCFF or title."""
    parts={id for id in s.editable_groups['encoder']
           if id.startswith(('aifi','a5','p3','p4','p5'))}
    outside={id:copy.deepcopy(n) for id,n in {**s.groups,**s.nodes}.items() if id not in parts}
    outside_edges=[copy.deepcopy(e) for e in s.edges
                   if e.source not in parts and e.target not in parts]
    originals={id:copy.deepcopy(s.get(id)) for id in parts}
    scale=.8;cx,cy=473,306.9
    def point(x,y):return cx+(x-cx)*scale,cy+(y-cy)*scale
    for id in parts:
        n=s.get(id);n.x,n.y=point(n.x,n.y);n.w*=scale;n.h*=scale
        n.lines=[(label,size*scale,bold,color) for label,size,bold,color in n.lines]
        if hasattr(n,'radius'):n.radius*=scale
    for e in s.edges:
        if e.source in parts and e.target in parts:
            e.bends=[point(x,y) for x,y in e.bends]
        elif e.id in ('a5_coarse_input','p4_scale_input'):
            # Keep the CCFF-side horizontal segment fixed; reconnect only the
            # scaled map-side endpoint to the existing vertical routing rail.
            sy=s.points(e)[0][1];e.bends=[(645,sy),(645,s.points(e)[-1][1])]
        elif e.id=='backbone_levels':
            inlet_y=s.groups['p3_plane0'].y+s.groups['p3_plane0'].h
            e.target_fraction=(inlet_y-s.get('p3').y)/s.get('p3').h
            e.bends=[(285,s.points(e)[0][1]),(285,inlet_y)]
    # Keep the P3-to-Fusion link straight by attaching it to the actual sloping
    # map edge at the unchanged Fusion height, instead of adding an elbow.
    p=s.groups['p3_plane0'];fusion_y=s.nodes['scale_fusion3'].point('left')[1]
    v=(fusion_y-p.y)/p.h
    assert 0<=v<=1
    s.nodes['p3_exit'].x=p.x+p.w-.14*p.w*v
    s.nodes['p3_exit'].y=fusion_y
    for id,old in outside.items():assert vars(s.get(id))==vars(old),id
    for e in outside_edges:assert any(vars(e)==vars(now) for now in s.edges),e.id
    for id in parts:
        n=s.get(id);old=originals[id]
        assert abs(n.w-old.w*scale)<1e-7 and abs(n.h-old.h*scale)<1e-7,id
    print('Annotated AIFI/maps/token assembly: uniform 80% size; CCFF/decoder/output unchanged.')


def narrow_encoder(s):
    """Narrow the complete encoder, then additionally tighten the CCFF."""
    enc=set(s.editable_groups['encoder'])|{'encoder'}
    snapshot={id:copy.deepcopy(n) for id,n in {**s.groups,**s.nodes}.items()}
    left=s.groups['encoder'].x
    cf_left=s.groups['fusion_bg'].x
    cf_right=cf_left+s.groups['fusion_bg'].w
    whole=.8;ccff=.85;shift=158
    extra=(cf_right-cf_left)*whole*(1-ccff)
    def xmap(x):
        if x<=cf_left:return left+whole*(x-left)
        if x<=cf_right:return left+whole*(cf_left-left)+whole*ccff*(x-cf_left)
        return left+whole*(x-left)-extra
    right={id for id,n in {**s.groups,**s.nodes}.items() if id not in enc and n.x>=913}
    for id in enc:
        n=s.get(id);old=copy.deepcopy(n)
        n.x=xmap(old.x);n.w=xmap(old.x+old.w)-n.x
        if id.startswith('scale_fusion'):n.tight_spacing=True
        if id in {'coarse_projection','coarse_resample','fine_projection','fine_resample'}:
            # Keep operators square instead of stretching them into rectangles.
            n.x=xmap(old.x+old.w/2)-10;n.y=old.y+old.h/2-10;n.w=n.h=20
        elif '_tile' in id and id.startswith('aifi'):
            n.y=old.y+(old.h-n.w)/2;n.h=n.w
    # Preserve a circular C and its center height, with a small inset from the
    # narrower CCFF boundary. Frame and Fusion heights stay exactly unchanged.
    c=s.nodes['concat'];old=snapshot['concat']
    c.x=xmap(old.x+old.w/2)-3-12;c.y=old.y+old.h/2-12;c.w=c.h=24
    for id in right:s.get(id).x-=shift
    s.nodes['title'].w-=shift
    for e in s.edges:
        if e.source in enc and e.target in enc:
            e.bends=[(xmap(x),y) for x,y in e.bends]
            if hasattr(e,'jump_at'):e.jump_at=(xmap(e.jump_at[0]),e.jump_at[1])
        elif e.source in right and e.target in right:
            e.bends=[(x-shift,y) for x,y in e.bends]
        elif e.id=='backbone_levels':
            e.bends=[(xmap(x),y) for x,y in e.bends]
        if e.id in {'coarse_output_c','fine_output_c'}:
            e.bends=[(c.x+c.w/2,s.points(e)[0][1])]
    for id in right:
        expected=copy.deepcopy(snapshot[id]);expected.x-=shift
        assert vars(s.get(id))==vars(expected),id
    for id in set(snapshot)-enc-right-{'title'}:
        assert vars(s.get(id))==vars(snapshot[id]),id
    for id in ('encoder','fusion_bg','aifi','scale_fusion5','scale_fusion4','scale_fusion3'):
        assert s.get(id).h==snapshot[id].h,id
        assert s.get(id).lines==snapshot[id].lines,id
    assert abs(s.groups['encoder'].w-482.12)<1e-7
    assert abs(s.groups['fusion_bg'].w-169.32)<1e-7
    assert c.x+c.w<s.groups['fusion_bg'].x+s.groups['fusion_bg'].w
    print('Encoder width: 640 -> 482.12; CCFF width: 249 -> 169.32; heights/text unchanged.')


def align_input_backbone(s):
    """Raise only the annotated input cluster and replace its exit by a line."""
    parts={'input','input_label','backbone',*s.editable_groups['input']}
    outside={id:copy.deepcopy(n) for id,n in {**s.groups,**s.nodes}.items() if id not in parts}
    edge=next(e for e in s.edges if e.id=='backbone_levels')
    start,end=s.points(edge)[0],s.points(edge)[-1]
    dy=end[1]-start[1]
    assert dy<0
    for id in parts:s.get(id).y+=dy
    # Keep the caption clear of the vertical Input -> Backbone arrow.
    s.nodes['input_label'].y=s.groups['input'].y-32
    edge.bends=[]
    start,end=s.points(edge)
    assert abs(start[1]-end[1])<1e-7 and end[0]>start[0]
    input_edge=next(e for e in s.edges if e.id=='input_backbone')
    assert not input_edge.bends
    for id,n in outside.items():assert vars(s.get(id))==vars(n),id
    print(f'Input/Backbone raised {-dy:.2f} units; Backbone -> P3 is a straight horizontal arrow.')


def enlarge_aifi_region(s):
    """Slightly enlarge the selected assembly and center it beside CCFF."""
    parts={id for id in s.editable_groups['encoder']
           if id.startswith(('aifi','a5','p3','p4','p5'))}
    outside={id:copy.deepcopy(n) for id,n in {**s.groups,**s.nodes}.items() if id not in parts}
    outside_edges=[copy.deepcopy(e) for e in s.edges
                   if e.source not in parts and e.target not in parts]
    ns=[s.get(id) for id in parts]
    x1=min(n.x for n in ns);x2=max(n.x+n.w for n in ns)
    y1=min(n.y for n in ns);y2=max(n.y+n.h for n in ns)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    cf=s.groups['fusion_bg'];target_cy=cf.y+cf.h/2
    scale=1.12
    def point(x,y):return cx+scale*(x-cx),target_cy+scale*(y-cy)
    for id in parts:
        n=s.get(id);n.x,n.y=point(n.x,n.y);n.w*=scale;n.h*=scale
        n.lines=[(label,size*scale,bold,color) for label,size,bold,color in n.lines]
        if hasattr(n,'radius'):n.radius*=scale
    for e in s.edges:
        if e.source in parts and e.target in parts:
            e.bends=[point(x,y) for x,y in e.bends]
        elif e.id in ('a5_coarse_input','p4_scale_input'):
            rail=e.bends[0][0]
            e.bends=[(rail,s.points(e)[0][1]),(rail,s.points(e)[-1][1])]
    # Reattach both P3 interfaces to its physical sloping edges. This preserves
    # the previous horizontal Backbone link without moving the input cluster.
    p=s.groups['p3_plane0'];d=.14*p.w
    entry=next(e for e in s.edges if e.id=='backbone_levels')
    inlet_y=s.points(entry)[0][1];v=(inlet_y-p.y)/p.h
    assert 0<=v<=1
    s.text('p3_inlet',p.x+d*(1-v),inlet_y,0,0,'',size=1)
    s.editable_groups['encoder'].append('p3_inlet')
    entry.target='p3_inlet';entry.target_side='left';entry.target_fraction=.5;entry.bends=[]
    exit_y=s.nodes['scale_fusion3'].point('left')[1];v=(exit_y-p.y)/p.h
    assert 0<=v<=1
    s.nodes['p3_exit'].x=p.x+p.w-d*v;s.nodes['p3_exit'].y=exit_y
    for id,n in outside.items():assert vars(s.get(id))==vars(n),id
    for e in outside_edges:assert any(vars(e)==vars(now) for now in s.edges),e.id
    ns=[s.get(id) for id in parts]
    new_y1=min(n.y for n in ns);new_y2=max(n.y+n.h for n in ns)
    assert abs((new_y1+new_y2)/2-target_cy)<1e-7
    assert abs((new_y2-new_y1)-(y2-y1)*scale)<1e-7
    assert abs(s.points(entry)[0][1]-s.points(entry)[-1][1])<1e-7
    print(f'Annotated assembly enlarged 12%; height {(new_y2-new_y1):.2f}, center aligned with CCFF.')


def export(s):
    old_png=OUT/(STEM+'.png')
    previous=Image.open(old_png).convert('RGB').copy() if old_png.is_file() else None
    enlarge_backup=OUT/'versions'/'before_aifi_enlarge_20261014'
    if not enlarge_backup.exists():
        enlarge_backup.mkdir(parents=True)
        for suffix in ('.drawio','.svg','.png'):
            p=OUT/(STEM+suffix)
            if p.is_file():shutil.copy2(p,enlarge_backup/p.name)
        if PDF_OUT.is_file():shutil.copy2(PDF_OUT,enlarge_backup/PDF_OUT.name)
    input_backup=OUT/'versions'/'before_input_raise_20261014'
    if not input_backup.exists():
        input_backup.mkdir(parents=True)
        for suffix in ('.drawio','.svg','.png'):
            p=OUT/(STEM+suffix)
            if p.is_file():shutil.copy2(p,input_backup/p.name)
        if PDF_OUT.is_file():shutil.copy2(PDF_OUT,input_backup/PDF_OUT.name)
    narrow_backup=OUT/'versions'/'before_encoder_narrow_20261014'
    if not narrow_backup.exists():
        narrow_backup.mkdir(parents=True)
        for suffix in ('.drawio','.svg','.png'):
            p=OUT/(STEM+suffix)
            if p.is_file():shutil.copy2(p,narrow_backup/p.name)
        if PDF_OUT.is_file():shutil.copy2(PDF_OUT,narrow_backup/PDF_OUT.name)
    resize_backup=OUT/'versions'/'full_flow_before_aifi_resize_20261014'
    if not resize_backup.exists():
        resize_backup.mkdir(parents=True)
        for suffix in ('.drawio','.svg','.png'):
            p=OUT/(STEM+suffix)
            if p.is_file():shutil.copy2(p,resize_backup/p.name)
        if PDF_OUT.is_file():shutil.copy2(PDF_OUT,resize_backup/PDF_OUT.name)
    # Preserve the last delivered layout once before changing the full page.
    backup=OUT/'versions'/'encoder_only_20261014'
    if not backup.exists():
        backup.mkdir(parents=True)
        for suffix in ('.drawio','.svg','.png'):
            p=OUT/(STEM+suffix)
            if p.is_file():shutil.copy2(p,backup/p.name)
        if PDF_OUT.is_file():shutil.copy2(PDF_OUT,backup/PDF_OUT.name)

    ref.pub.polygon_points=flow_polygon
    ref.W,ref.H=W,H
    ref.pub.W,ref.pub.H=W,H
    svg=s.svg();ET.fromstring(svg)
    assert '<image' not in svg
    (OUT/(STEM+'.svg')).write_text(svg,encoding='utf-8')
    root=ET.Element('mxfile',host='app.diagrams.net',type='device')
    s.drawio(root);ref.verify_drawio(root);ET.indent(root,space='  ')
    ET.ElementTree(root).write(OUT/(STEM+'.drawio'),encoding='utf-8',xml_declaration=True)
    pdfmetrics.registerFont(TTFont('Diagram','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
    pdfmetrics.registerFont(TTFont('DiagramBold','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))
    for id in ('encoder_title','scale_fusion5','scale_fusion4','scale_fusion3','aifi'):
        n=s.nodes[id];label,size,bold,_=n.lines[0]
        assert pdfmetrics.stringWidth(label,'DiagramBold' if bold else 'Diagram',size)<=n.w-4,id
    c=canvas.Canvas(str(PDF_OUT),pagesize=(W,H))
    c.setTitle('D-FINE-S + AQS - compact editable reference flow')
    for n in s.groups.values():ref.pdf_node(c,n)
    straight=copy.copy(s);straight.edges=[e for e in s.edges if not getattr(e,'curved',False)]
    ref.pdf_edges(c,straight)
    for n in s.nodes.values():ref.pdf_node(c,n)
    for e in s.edges:
        if not getattr(e,'curved',False):continue
        p,a,b,q=s.points(e)
        c.setStrokeColor(e.color);c.setLineWidth(2.2);c.setDash([])
        path=c.beginPath();path.moveTo(p[0],H-p[1])
        path.curveTo(a[0],H-a[1],b[0],H-b[1],q[0],H-q[1]);c.drawPath(path)
    c.showPage();c.save()
    pdf=PdfReader(PDF_OUT)
    assert len(pdf.pages)==1 and '/XObject' not in pdf.pages[0]['/Resources']
    content=pdf.pages[0].extract_text()
    for label in ('Efficient Hybrid Encoder','CCFF','Query selection','Top300',
                  'Layer 2','AQS','Layer 3','SCDown','Position Embedding'):
        assert label in content,label
    rendered=pdfium.PdfDocument(str(PDF_OUT))[0].render(scale=2).to_pil().convert('RGB')
    if previous is not None:
        def same(a,b):
            return a.size==b.size and max(hi for lo,hi in ImageChops.difference(a,b).getextrema())<=1
        if previous.size==(4000,1240):
            # Right modules are translated by exactly 158 units, not resized.
            for old_crop,new_crop in [((1880,0,2720,1080),(1564,0,2404,1080)),
                                      ((2720,0,4000,1240),(2404,0,3684,1240))]:
                assert same(previous.crop(old_crop),rendered.crop(new_crop))
            print('AQS/right-side modules: unchanged after spacing-only translation.')
        elif previous.size==rendered.size:
            for crop in ((0,0,600,1240),(1166,0,3684,1240)):
                assert same(previous.crop(crop),rendered.crop(crop))
            print('Input cluster, CCFF and right-side workflow: visually unchanged.')
    rendered.save(OUT/(STEM+'.png'))
    print('Preserved previous layout:',backup)
    print('Editable source:',OUT/(STEM+'.drawio'))
    print('Vector PDF:',PDF_OUT)


if __name__=='__main__':
    export(build())
