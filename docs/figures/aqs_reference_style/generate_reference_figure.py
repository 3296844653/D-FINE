"""Compact, fully editable architecture matching the user's layout reference.

Only documentation is generated. Dark innovation highlights and reference-
paper UCS/DCS/CAF modules are deliberately absent. Classroom tiles are native
vector illustrations, not real model predictions.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import pypdfium2 as pdfium
from PIL import Image, ImageChops
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
STEM = 'dfine_s_aqs_reference_style'
PDF_OUT = PROJECT / 'output/pdf' / (STEM + '.pdf')
sys.path.insert(0, str(OUT.parent / 'aqs_publication_style'))
import generate_publication_figure as pub

base = pub.base
W, H = 2200, 650
pub.W, pub.H = W, H
base.LINE = '#303030'
INK = '#151515'
NEUTRAL = '#505050'
FPN = '#FFE7D1'
PAN = '#DAEEF1'
AIFI = '#FBE6D4'
FEATURE = ['#EEF1DE', '#DFE6C7', '#CAD5AE']


class ReferenceScene(pub.PublicationScene):
    """Use the same geometry in SVG, draw.io, and PDF."""

    def __init__(self, *args):
        super().__init__(*args)
        self.editable_groups = {}

    @staticmethod
    def svg_node(n):
        text = pub.PublicationScene.svg_node(n)
        if not n.text_only and not n.ellipse and not pub.polygon_points(n):
            radius = getattr(n, 'radius', 8)
            text = text.replace('rx="8"', f'rx="{radius:g}"')
        if getattr(n, 'no_stroke', False):
            text = re.sub(r'stroke="[^"]+"', 'stroke="none"', text)
        if getattr(n, 'outline_only', False):
            text = text.replace(f'fill="{n.fill}"', 'fill="none"', 1)
        return text

    def svg(self):
        svg=super().svg()
        for e in self.edges:
            if not hasattr(e,'jump_at'):continue
            x,y=e.jump_at;r=6
            pts=self.points(e)
            d=f'M{pts[0][0]:g} {pts[0][1]:g} L{x-r:g} {y:g} Q{x:g} {y+10:g} {x+r:g} {y:g}'
            d+=''.join(f' L{px:g} {py:g}' for px,py in pts[1:])
            arc=f'M{x-r:g} {y:g} Q{x:g} {y+10:g} {x+r:g} {y:g}'
            replacement=(f'<path d="{arc}" fill="none" stroke="#DFEBE5" stroke-width="6"/>'
                         f'<path id="{e.id}" d="{d}" fill="none" stroke="{e.color}" '
                         'stroke-width="2.2" marker-end="url(#arrow1)"/>')
            svg=re.sub(rf'<polyline id="{e.id}"[^>]+/>',replacement,svg)
        return svg

    def drawio(self, parent):
        super().drawio(parent)
        root = parent.findall('diagram')[-1].find('mxGraphModel/root')
        cells = {c.attrib['id']: c for c in root.findall('mxCell')}
        all_nodes = {**self.groups, **self.nodes}
        for id, n in all_nodes.items():
            cell = cells[id]
            style = cell.attrib['style']
            if getattr(n, 'radius', None) is not None:
                style += f'absoluteArcSize=1;arcSize={2*n.radius:g};'
            if getattr(n, 'no_stroke', False):
                style += 'strokeColor=none;'
            if getattr(n, 'outline_only', False):
                style += 'fillColor=none;'
            if getattr(n, 'tight_spacing', False):
                style += 'spacing=2;'
            cell.set('style', style)
        for e in self.edges:
            if hasattr(e,'jump_at'):
                cell=cells[e.id]
                cell.set('style',cell.attrib['style']+'jumpStyle=arc;jumpSize=8;')
        # Compound elements stay movable, while every child remains editable.
        for group_id, member_ids in self.editable_groups.items():
            n = all_nodes[group_id]
            group = cells[group_id]
            bg = copy.deepcopy(group)
            bg.set('id', group_id + '_background')
            bg.set('parent', group_id)
            bg_geo = bg.find('mxGeometry')
            bg_geo.set('x', '0'); bg_geo.set('y', '0')
            root.insert(list(root).index(group) + 1, bg)
            group.set('style', 'group;html=1;connectable=1;')
            group.set('value', '')
            for id in member_ids:
                cell = cells[id]
                cell.set('parent', group_id)
                geo = cell.find('mxGeometry')
                geo.set('x', f'{float(geo.attrib["x"])-n.x:g}')
                geo.set('y', f'{float(geo.attrib["y"])-n.y:g}')
            members = set(member_ids)
            for edge in self.edges:
                if edge.source in members and edge.target in members:
                    cell = cells[edge.id]
                    cell.set('parent', group_id)
                    for pt in cell.findall('mxGeometry/Array/mxPoint'):
                        pt.set('x', f'{float(pt.attrib["x"])-n.x:g}')
                        pt.set('y', f'{float(pt.attrib["y"])-n.y:g}')


def node(s, id, x, y, w, h, title='', body=(), fill='#FFFFFF', size=25, body_size=19):
    n = s.node(id, x, y, w, h, title, body, fill=fill, stroke=NEUTRAL,
               title_size=size, body_size=body_size)
    n.lines = [(t, fs, bold, INK) for t, fs, bold, _ in n.lines]
    return n


def text(s, id, x, y, w, h, label, size=19, bold=False, color=INK, align='center'):
    s.text(id, x, y, w, h, label, size=size, bold=bold, color=color, align=align)


def plane(s, id, x, y, w, h, label):
    # Native parallelograms with shallow depth, no bitmap feature maps.
    s.text(id, x, y-6, w+7, h+13, '', size=1)
    for j in range(2, -1, -1):
        s.group(f'{id}_plane{j}', x+j*3, y-j*4, w, h,
                fill=FEATURE[j], stroke='#8D947E')
        s.groups[f'{id}_plane{j}'].kind = 'plane'
    text(s, id+'_label', x-47, y+3, 40, 26, label, size=20, color='#9B6643')


def tokens(s, id, x, y, count, width=24, height=20, gap=7, horizontal=False, colored=False):
    ww = count*(width+gap)-gap if horizontal else width
    hh = height if horizontal else count*(height+gap)-gap
    s.text(id, x, y, ww, hh, '', size=1)
    members = []
    for i in range(count):
        nx, ny = (x+i*(width+gap), y) if horizontal else (x, y+i*(height+gap))
        fill = ['#FFF0B7', '#E4EFCB', '#D8E7F3'][i % 3] if colored else '#FFF6DB'
        n = node(s, f'{id}_tile{i}', nx, ny, width, height, fill=fill)
        n.lines = []; n.square = True; n.radius = 0
        members.append(n.id)
    if not horizontal:
        s.editable_groups[id] = members


def classroom(s, id, x, y, w, h, detected=False):
    """Three classroom silhouettes; any boxes are explicitly schematic."""
    s.group(id, x, y, w, h, fill='#F0F3F1', stroke='#BCC5C5')
    s.groups[id].radius = 2
    members = []

    def shape(tag, xx, yy, ww, hh, fill, ellipse=False):
        n = node(s, id+'_'+tag, x+xx*w, y+yy*h, ww*w, hh*h, fill=fill)
        n.lines = []; n.ellipse = ellipse; n.radius = 0; n.no_stroke = True
        members.append(n.id)
        return n

    shape('board', .06, .09, .48, .22, '#C4D7CC')
    shape('window', .67, .09, .25, .22, '#D6EAF0')
    shape('window_bar', .784, .09, .02, .22, '#FFFFFF')
    shape('floor', 0, .80, 1, .20, '#E1E2DC')
    for i, cx in enumerate((.21, .50, .79)):
        shape(f'body{i}', cx-.062, .52, .124, .23, '#A3B4CA')
        shape(f'head{i}', cx-.046, .38, .092, .13, '#E1C3A6', True)
        shape(f'desk{i}', cx-.12, .70, .24, .065, '#CBBCA2')
        shape(f'leg{i}', cx-.085, .765, .016, .145, '#B3A58F')
        shape(f'legb{i}', cx+.07, .765, .016, .145, '#B3A58F')
        if i == 0:
            shape('raised_arm', cx-.085, .355, .025, .25, '#A3B4CA')
            shape('raised_hand', cx-.087, .31, .028, .053, '#E1C3A6', True)
        else:
            shape(f'book{i}', cx-.047, .643, .11, .05, '#FFFDF3')
            shape(f'bookline{i}', cx+.002, .643, .008, .05, '#C6CECA')
            if i == 2:
                shape('pencil', cx+.055, .60, .009, .092, '#697984')
        if detected:
            color = ['#A6854F', '#6487A4', '#789763'][i]
            n = node(s, f'{id}_box{i}', x+(cx-.126)*w, y+.31*h, .253*w, .46*h, fill='#FFFFFF')
            n.lines = []; n.stroke = color; n.outline_only = True; n.radius = 0
            members.append(n.id)
            text(s, f'{id}_class{i}', x+(cx-.136)*w, y+.775*h, .273*w, .08*h,
                 ['hand', 'read', 'write'][i], size=14, color=color)
            members.append(f'{id}_class{i}')
    s.editable_groups[id] = members


def compact_scene():
    s = ReferenceScene('D-FINE-S + AQS - compact architecture', W, H,
                       'Reference-style left-to-right architecture with pale blocks, no dark innovation highlights. '
                       'The original three-scale D-FINE-S encoder is retained, with AQS between decoder layers 2 and 3. '
                       'Classroom drawings are schematic and are not model prediction evidence.')
    classroom(s, 'input', 55, 360, 205, 174)
    text(s, 'input_label', 40, 543, 235, 29, 'Input image', size=23)
    n = node(s, 'backbone', 66, 129, 183, 82, 'BACKBONE', fill='#DEEBEF', size=24)
    n.kind = 'trapezoid'
    s.edge('input_backbone', 'input', 'backbone', source_side='top', target_side='bottom')

    s.group('encoder', 310, 30, 980, 558, fill='#F1F1F1', stroke='#F1F1F1')
    s.groups['encoder'].radius = 66; s.groups['encoder'].no_stroke = True
    before = set(s.nodes) | set(s.groups)
    text(s, 'encoder_title', 343, 44, 530, 48, 'Hybrid Encoder', size=34, bold=True)
    # Three standard D-FINE-S levels, not the reference's extra P2 level.
    plane(s, 'p5', 418, 363, 98, 28, 'P5')
    plane(s, 'p4', 396, 405, 150, 34, 'P4')
    plane(s, 'p3', 370, 449, 224, 41, 'P3')
    # Restore the pre-simplification AIFI/token/A5 presentation. Only the FPN
    # and PAN containers are abstracted; their internal scale operations remain.
    tokens(s, 'aifi_tokens', 357, 307, 10, width=18, height=20, gap=6, horizontal=True)
    for i in range(10):
        s.nodes[f'aifi_tokens_tile{i}'].fill = '#DDE6C6'
    node(s, 'aifi', 350, 214, 250, 60, 'AIFI', fill=AIFI, size=32)
    plane(s, 'a5', 429, 153, 77, 23, 'A5')
    s.edge('p5_tokens', 'p5', 'aifi_tokens', source_side='top', target_side='bottom', target_fraction=113.5/234)
    for i in range(10):
        s.edge(f'token_aifi{i}', f'aifi_tokens_tile{i}', 'aifi', source_side='top', target_side='bottom',
               target_fraction=(366+i*24-350)/250)
    s.edge('aifi_a5', 'aifi', 'a5', source_side='top', source_fraction=121/250, target_side='bottom')
    # FPN/PAN are whole multi-scale stages, not individual scale operators.
    # One input bundle, one stage link, one complete pyramid output bundle.
    n = node(s, 'fusion_inputs', 630, 165, 2, 305, fill=base.LINE)
    n.lines = []; n.radius = 0; n.no_stroke = True
    s.edge('a5_bundle', 'a5', 'fusion_inputs', target_fraction=0, arrow=False)
    s.edge('p4_bundle', 'p4', 'fusion_inputs', target_fraction=257.5/305, arrow=False)
    s.edge('p3_bundle', 'p3', 'fusion_inputs', target_fraction=1, arrow=False)
    node(s, 'fpn', 675, 292, 195, 88, 'FPN', fill=FPN, size=32)
    node(s, 'pan', 1000, 292, 195, 88, 'PAN', fill=PAN, size=32)
    s.edge('bundle_fpn', 'fusion_inputs', 'fpn', source_fraction=171/305)
    s.edge('fpn_pan', 'fpn', 'pan')
    encoder_members = sorted((set(s.nodes) | set(s.groups))-before)
    s.editable_groups['encoder'] = encoder_members
    s.edge('backbone_levels', 'backbone', 'p3', source_side='right', target_side='bottom',
           bends=[(287,170),(287,540),(485.5,540)])

    n = node(s, 'concat', 1308, 318, 36, 36, 'C', size=26)
    n.ellipse = True
    s.edge('pan_memory', 'pan', 'concat')
    tokens(s, 'memory', 1372, 164, 13)
    text(s, 'memory_label', 1329, 530, 113, 27, 'Memory', size=20)
    s.edge('concat_memory', 'concat', 'memory')
    n = node(s, 'selection', 1440, 208, 58, 256, 'Query selection', fill='#FCE0C6', size=27)
    n.vertical = True
    s.edge('memory_selection', 'memory', 'selection')
    tokens(s, 'queries', 1533, 224.5, 8, height=20, gap=9)
    text(s, 'query_label', 1501, 473, 90, 30, 'Top300', size=21)
    s.edge('selection_queries', 'selection', 'queries')

    s.group('decoder', 1606, 135, 206, 406, fill='#DFEEF3', stroke='#CCDCE3')
    s.groups['decoder'].radius = 9
    dec_members = []
    n = node(s, 'decoder_label', 1610, 167, 39, 335, 'D-FINE Decoder & Head', fill='#DFEEF3', size=24)
    n.vertical = True; n.no_stroke = True; dec_members.append(n.id)
    for id, y, title, height in [('l1',169,'Layer 1',48),
                               ('l2',237,'Layer 2',59),
                               ('aqs',317,'AQS',59),
                               ('l3',409,'Layer 3',48)]:
        n = node(s, id, 1655, y, 133, height, title, fill='#FFFFFF', size=25)
        dec_members.append(n.id)
    text(s, 'heads', 1649, 483, 145, 29, 'Head', size=25, bold=True)
    dec_members.append('heads')
    for src, dst in [('l1','l2'),('l2','aqs'),('aqs','l3'),('l3','heads')]:
        s.edge(src+'_'+dst, src, dst, source_side='bottom', target_side='top')
    s.editable_groups['decoder'] = dec_members
    s.edge('queries_decoder', 'queries', 'decoder', target_fraction=201/406)
    # Memory also reaches decoder cross-attention. AQS has no direct feature-map input.
    s.edge('shared_memory', 'memory', 'decoder', source_side='top', target_side='top', target_fraction=.62,
           bends=[(1384,99),(1733.72,99)])
    tokens(s, 'detections', 1835, 224.5, 8, height=20, gap=9, colored=True)
    s.edge('decoder_predictions', 'decoder', 'detections', source_fraction=201/406)
    classroom(s, 'output', 1914, 270, 224, 185, detected=True)
    text(s, 'output_label', 1889, 468, 274, 29, 'Detection output (schematic)', size=21)
    s.edge('predictions_output', 'detections', 'output', target_fraction=66/185)
    text(s, 'title', 315, 603, 1580, 38, 'D-FINE-S with Adaptive Query Selection (AQS)', size=29, bold=True)
    for edge in s.edges:
        edge.color = base.LINE
    for n in s.nodes.values():
        if not n.text_only:
            assert len(n.lines) <= 1, f'Module subtitle is not allowed: {n.id}'
    all_ids = set(s.nodes) | set(s.groups)
    assert {'fpn','pan','a5','aifi_tokens'} <= all_ids
    assert not {'fpn3','fpn4','pan4','pan5','feature_outputs'} & all_ids
    assert sum(1 for n in s.nodes.values() if n.lines and n.lines[0][0] == 'FPN') == 1
    assert sum(1 for n in s.nodes.values() if n.lines and n.lines[0][0] == 'PAN') == 1
    assert len([e for e in s.edges if e.source == 'fpn' and e.target == 'pan']) == 1
    assert [e.target for e in s.edges if e.source == 'pan'] == ['concat']
    assert not any(e.id in {'a5_pan_skip','fpn3_memory','pan4_memory','pan5_memory'} for e in s.edges)
    for edge in s.edges:
        points = s.points(edge)
        for (x1,y1), (x2,y2) in zip(points,points[1:]):
            assert abs(x2-x1) < 1e-7 or abs(y2-y1) < 1e-7, f'Diagonal segment: {edge.id}'
    s.validate()
    return s


def make_scene():
    """Replace only the encoder of the user's selected compact base figure."""
    s = compact_scene()
    old_members = set(s.editable_groups['encoder'])
    outside_nodes = {id:copy.deepcopy(n) for id,n in {**s.groups,**s.nodes}.items()
                     if id not in old_members}
    outside_edges = [copy.deepcopy(e) for e in s.edges
                     if e.source not in old_members and e.target not in old_members]
    for id in old_members:
        s.nodes.pop(id,None);s.groups.pop(id,None)
    s.edges = [e for e in s.edges if e.source not in old_members and e.target not in old_members]
    before = set(s.nodes)|set(s.groups)
    text(s,'encoder_title',343,44,530,48,'Hybrid Encoder',size=34,bold=True)
    # Keep the base pyramid and its backbone attachment exactly in place.
    plane(s,'p5',418,363,98,28,'P5')
    plane(s,'p4',396,405,150,34,'P4')
    plane(s,'p3',370,449,224,41,'P3')
    node(s,'aifi',350,214,250,60,'AIFI',fill=AIFI,size=32)
    tokens(s,'aifi_tokens',357,307,10,width=18,height=20,gap=6,horizontal=True)
    for i in range(10):s.nodes[f'aifi_tokens_tile{i}'].fill='#DDE6C6'
    s.edge('p5_tokens','p5','aifi_tokens',source_side='top',target_side='bottom',target_fraction=113.5/234)
    for i in range(10):
        s.edge(f'token_aifi{i}',f'aifi_tokens_tile{i}','aifi',source_side='top',target_side='bottom',
               target_fraction=(366+i*24-350)/250)
    plane(s,'a5',429,153,77,23,'A5')
    s.edge('aifi_a5','aifi','a5',source_side='top',source_fraction=121/250,target_side='bottom')

    # The reference's compact hierarchical fusion appearance, with D-FINE's
    # true two top-down and two bottom-up fusion operations (not three copied
    # RT-DETR blocks or the reference's Conv3x3/BN/SiLU operator legend).
    s.group('fusion_bg',735,91,510,454,fill='#DFEBE5',stroke='#C6D7CC')
    s.groups['fusion_bg'].radius=14
    text(s,'fusion_title',748,101,484,35,'Cross-scale Fusion',size=26,bold=True)
    node(s,'lat5',832.5,151,55,28,'1 x 1',fill='#FFF0C7',size=16)
    node(s,'up5',832.5,205,55,28,'Up',fill='#FFF0C7',size=16)
    node(s,'fpn4',795,258,130,44,'Fusion',fill='#FBEAD9',size=24)
    node(s,'lat4',832.5,327,55,28,'1 x 1',fill='#FFF0C7',size=16)
    node(s,'up4',832.5,402,55,28,'Up',fill='#FFF0C7',size=16)
    node(s,'fpn3',795,460,130,44,'Fusion',fill='#FBEAD9',size=24)
    node(s,'down3',972.5,468,55,28,'Down',fill='#D8E8F4',size=16)
    node(s,'pan4',1090,355,130,44,'Fusion',fill='#E3EFF3',size=24)
    node(s,'down4',1127.5,264,55,28,'Down',fill='#D8E8F4',size=16)
    node(s,'pan5',1090,143,130,44,'Fusion',fill='#E3EFF3',size=24)
    s.edge('a5_lat5','a5','lat5')
    s.edge('lat5_up5','lat5','up5',source_side='bottom',target_side='top')
    s.edge('up5_fpn4','up5','fpn4',source_side='bottom',target_side='top')
    s.edge('p4_fpn4','p4','fpn4',bends=[(760,422.5),(760,280)])
    s.edge('fpn4_lat4','fpn4','lat4',source_side='bottom',target_side='top')
    s.edge('lat4_up4','lat4','up4',source_side='bottom',target_side='top')
    s.edge('up4_fpn3','up4','fpn3',source_side='bottom',target_side='top')
    s.edge('p3_fpn3','p3','fpn3',bends=[(760,470),(760,482)])
    s.edge('fpn3_down3','fpn3','down3')
    s.edge('down3_pan4','down3','pan4',source_side='right',target_side='bottom',bends=[(1155,482)])
    s.edge('lat4_pan4','lat4','pan4',bends=[(970,341),(970,377)])
    s.edge('pan4_down4','pan4','down4',source_side='top',target_side='bottom')
    s.edge('down4_pan5','down4','pan5',source_side='top',target_side='bottom')
    s.edge('lat5_pan5','lat5','pan5')
    # Three true encoder outputs gather on one bus, keeping the old exterior
    # exit to C at exactly y=336. This does not introduce a fusion operator.
    n=node(s,'encoder_outputs',1260,165,2,363,fill=base.LINE)
    n.lines=[];n.radius=0;n.no_stroke=True
    s.edge('pan5_out','pan5','encoder_outputs',target_fraction=0,arrow=False)
    s.edge('pan4_out','pan4','encoder_outputs',target_fraction=212/363,arrow=False)
    s.edge('fpn3_out','fpn3','encoder_outputs',source_side='bottom',target_side='bottom',
           bends=[(860,528)],arrow=False)
    s.editable_groups['encoder']=sorted((set(s.nodes)|set(s.groups))-before)
    s.edge('backbone_levels','backbone','p3',source_side='right',target_side='bottom',
           bends=[(287,170),(287,540),(485.5,540)])
    s.edge('pyramid_concat','encoder_outputs','concat',source_fraction=171/363)
    for e in s.edges:e.color=base.LINE
    # Structural/source checks and an exact check that every non-encoder
    # element/connection remains as in the user's chosen base version.
    for id,n in outside_nodes.items():
        current=s.get(id)
        assert vars(current)==vars(n),f'Outside encoder changed: {id}'
    for edge in outside_edges:
        assert any(vars(current)==vars(edge) for current in s.edges),edge.id
    # Additional user-requested change: input above, backbone below. Keep all
    # remaining parts, especially query selection/decoder/AQS, untouched.
    input_delta=129-s.groups['input'].y
    s.groups['input'].y=129
    for id in s.editable_groups['input']:
        s.get(id).y+=input_delta
    s.nodes['input_label'].y=89
    s.nodes['backbone'].y=360
    for e in s.edges:
        if e.id=='input_backbone':
            e.source_side='bottom';e.target_side='top';e.bends=[]
        elif e.id=='backbone_levels':
            e.source_side='bottom';e.target_side='bottom'
            e.bends=[(157.5,540),(485.5,540)]
    compact_encoder(s)
    apply_annotation_layout(s)
    for edge in s.edges:
        if getattr(edge,'feature_grid',False):continue
        pts=s.points(edge)
        for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
            assert abs(x1-x2)<1e-7 or abs(y1-y2)<1e-7,edge.id
    assert len([n for n in s.nodes.values() if n.lines and n.lines[0][0]=='Fusion'])==3
    assert {'scale_fusion5','scale_fusion4','scale_fusion3'} <= set(s.nodes)
    assert not {'fpn','pan'} & set(s.nodes)
    s.validate()
    return s


def compact_encoder(s):
    """Tighten only the annotated encoder, preserving its DAG and the rest."""
    members=set(s.editable_groups['encoder'])
    outside={id:copy.deepcopy(n) for id,n in {**s.groups,**s.nodes}.items()
             if id not in members and id!='encoder'}
    outside_edges=[copy.deepcopy(e) for e in s.edges
                   if e.source not in members and e.target not in members]
    # Remove the unnecessary white gaps first, then scale uniformly. The
    # operation topology and four actual fusion nodes are unchanged.
    s.groups['encoder'].w=800
    s.nodes['encoder_title'].x=310;s.nodes['encoder_title'].w=800
    s.groups['fusion_bg'].x-=80;s.groups['fusion_bg'].w-=100
    s.nodes['fusion_title'].x-=80;s.nodes['fusion_title'].w-=100
    for id in ['lat5','up5','fpn4','lat4','up4','fpn3']:
        s.nodes[id].x-=80
    for id in ['pan4','down4','pan5','encoder_outputs']:
        s.nodes[id].x-=180
    s.nodes['down3'].x-=100
    for e in s.edges:
        if e.id=='p4_fpn4':e.bends=[(680,422.5),(680,280)]
        elif e.id=='p3_fpn3':e.bends=[(680,470),(680,482)]
        elif e.id=='down3_pan4':e.bends=[(975,482)]
        elif e.id=='lat4_pan4':e.bends=[(840,341),(840,377)]
        elif e.id=='fpn3_out':e.bends=[(780,528)]
    scale=.80
    x0,y0=480,91.2
    def transform(x,y):
        return x0+(x-310)*scale,y0+(y-30)*scale
    for id in members|{'encoder'}:
        n=s.get(id)
        n.x,n.y=transform(n.x,n.y)
        n.w*=scale;n.h*=scale
        n.lines=[(label,size*scale,bold,color) for label,size,bold,color in n.lines]
        if getattr(n,'radius',None) is not None:n.radius*=scale
        if id in {'lat5','up5','lat4','up4','down3','down4'}:n.tight_spacing=True
    for e in s.edges:
        if e.source in members and e.target in members:
            e.bends=[transform(x,y) for x,y in e.bends]
        elif e.id=='backbone_levels':
            x=s.get('p3').point('bottom')[0]
            y=transform(0,540)[1]
            e.bends=[(157.5,y),(x,y)]
    # Keep the interface at the same y=336 to avoid moving Memory/Decoder.
    assert abs(s.get('encoder_outputs').point('right',171/363)[1]-336)<1e-7
    for id,n in outside.items():
        assert vars(s.get(id))==vars(n),f'Resizing changed outside node: {id}'
    for e in outside_edges:
        assert any(vars(current)==vars(e) for current in s.edges),e.id
    for id in members:
        n=s.get(id);bg=s.groups['encoder']
        assert bg.x-1e-7<=n.x and n.x+n.w<=bg.x+bg.w+1e-7,id
        assert bg.y-1e-7<=n.y and n.y+n.h<=bg.y+bg.h+1e-7,id
    print('Compact encoder: 640 x 446.4; widths reduced 35%, height reduced 20%.')


def apply_annotation_layout(s):
    """Reconstruct the encoder to match the full reference, not just its rows.

    The three Fusion containers summarize scale states. The middle container
    encapsulates separate FPN/PAN operators; this is not a cyclic executable
    graph. Yellow blocks summarize lateral+resize, blue blocks SCDown, and
    the blue rectangular branches retain the original lateral skips.
    """
    removed=set(s.editable_groups['encoder'])
    for id in removed:
        s.nodes.pop(id,None);s.groups.pop(id,None)
    s.edges=[e for e in s.edges if e.source not in removed and e.target not in removed]
    before=set(s.nodes)|set(s.groups)
    s.groups['encoder'].h=426.4
    text(s,'encoder_title',480,101,640,34,'Hybrid Encoder',size=27.2,bold=True)
    s.group('fusion_bg',858,145,249,355,fill='#DFEBE5',stroke='#C6D7CC')
    s.groups['fusion_bg'].radius=14
    text(s,'fusion_title',858,151,249,29,'CCFF',size=23,bold=True)

    def anchor(id,x,y):
        s.text(id,x,y,0,0,'',size=1)

    # Tight overlapping pyramid as in the reference. Labels sit on the maps,
    # not in a separate margin. Projection is still implicit in this overview.
    for id,x,y,w,h,label in [('p3',512,443.2,310,47,'P3'),
                            ('p4',536,426,257,44,'P4'),
                            ('p5',560,408,206,40,'P5'),
                            ('a5',599.04,167,172,44,'A5')]:
        plane(s,id,x,y,w,h,label)
        for j,color in enumerate(['#D2E4BE','#C1DBA8','#B1D28F']):
            s.groups[f'{id}_plane{j}'].fill=color
        label_node=s.nodes[id+'_label']
        if id=='a5':
            label_node.x=745;label_node.y=208;label_node.w=49
        else:
            label_node.x=x+14;label_node.y=y+h-26;label_node.w=43
        label_node.lines=[(label,18,True,INK)]
    anchor('a5_exit',759,189)
    anchor('a5_inlet',673,211)
    anchor('p5_token_origin',673,408)
    anchor('p4_exit',793,426)
    anchor('p3_exit',822,443.2)

    # Native editable lattice on the AIFI output map.
    a5=s.groups['a5_plane0'];d=.14*a5.w
    def grid_point(u,v):
        return a5.x+d*(1-v)+u*(a5.w-d),a5.y+v*a5.h
    for axis,count in [('u',10),('v',4)]:
        for i in range(1,count):
            f=i/count
            p,q=(grid_point(f,0),grid_point(f,1)) if axis=='u' else (grid_point(0,f),grid_point(1,f))
            src=f'a5_grid_{axis}{i}_a';dst=f'a5_grid_{axis}{i}_b'
            anchor(src,*p);anchor(dst,*q)
            s.edge(f'a5_grid_{axis}{i}',src,dst,color='#8FA47B',arrow=False)
            s.edges[-1].feature_grid=True

    node(s,'aifi',506,278,334,52,'AIFI',fill='#D9E8DE',size=28)
    for id,y,color in [('aifi_output_tokens',239,'#BDD99E'),
                       ('aifi_position_tokens',344,'#BCD0E9'),
                       ('aifi_tokens',369,'#BDD99E')]:
        tokens(s,id,512,y,11,width=22,height=20,gap=8,horizontal=True)
        for i in range(11):s.nodes[f'{id}_tile{i}'].fill=color
    for i in range(11):
        x=523+i*30;frac=(x-506)/334
        s.edge(f'aifi_feature_pos{i}',f'aifi_tokens_tile{i}',f'aifi_position_tokens_tile{i}',
               source_side='top',target_side='bottom',arrow=False,color=base.LINE)
        s.edge(f'aifi_pos_input{i}',f'aifi_position_tokens_tile{i}','aifi',
               source_side='top',target_side='bottom',target_fraction=frac,color=base.LINE)
        s.edge(f'aifi_output{i}','aifi',f'aifi_output_tokens_tile{i}',
               source_side='top',source_fraction=frac,target_side='bottom',color=base.LINE)
    s.edge('p5_tokens','p5_token_origin','aifi_tokens',source_side='top',target_side='bottom',color=base.LINE)
    s.edge('tokens_a5','aifi_output_tokens','a5_inlet',source_side='top',target_side='bottom',color=base.LINE)

    for id,y in [('scale_fusion5',210.8),('scale_fusion4',318),('scale_fusion3',425.2)]:
        node(s,id,888,y,106,36,'Fusion',fill='#FBEAD9',size=19.2)
    for id,x,y,color in [('coarse_projection',888,268,'#FFE7A1'),
                         ('coarse_resample',942,268,'#BFD0E9'),
                         ('fine_projection',888,379.6,'#FFE7A1'),
                         ('fine_resample',942,379.6,'#BFD0E9')]:
        n=node(s,id,x,y,28,28,fill=color);n.lines=[];n.radius=4
    s.edge('a5_coarse_input','a5_exit','coarse_projection',color=base.ORANGE,
           bends=[(845,189),(845,282)])
    s.edge('p4_scale_input','p4_exit','scale_fusion4',color=base.ORANGE,
           bends=[(845,426),(845,336)])
    s.edge('p3_scale_input','p3_exit','scale_fusion3',color=base.ORANGE)
    s.edge('top_down_mid','coarse_projection','scale_fusion4',source_side='bottom',
           target_side='top',target_fraction=14/106,color=base.ORANGE)
    s.edge('mid_fine_projection','scale_fusion4','fine_projection',source_side='bottom',
           source_fraction=14/106,target_side='top',color=base.ORANGE)
    s.edge('top_down_fine','fine_projection','scale_fusion3',source_side='bottom',
           target_side='top',target_fraction=14/106,color=base.ORANGE)
    s.edge('fine_resample_input','scale_fusion3','fine_resample',source_side='top',
           source_fraction=68/106,target_side='bottom',color=base.BLUE)
    s.edge('bottom_up_mid','fine_resample','scale_fusion4',source_side='top',
           target_side='bottom',target_fraction=68/106,color=base.BLUE)
    s.edge('coarse_resample_input','scale_fusion4','coarse_resample',source_side='top',
           source_fraction=68/106,target_side='bottom',color=base.BLUE)
    s.edge('bottom_up_coarse','coarse_resample','scale_fusion5',source_side='top',
           target_side='bottom',target_fraction=68/106,color=base.BLUE)
    # The reference's two right-hand blue lateral skips were missing before.
    # Crossings without dots are not joins; draw.io can display line jumps.
    anchor('coarse_skip_branch',902,307)
    anchor('fine_skip_branch',902,416)
    s.edge('coarse_lateral_skip','coarse_skip_branch','scale_fusion5',target_side='right',target_fraction=.8,
           bends=[(1020,307),(1020,239.6)],color=base.BLUE)
    s.edges[-1].jump_at=(956,307)
    s.edge('fine_lateral_skip','fine_skip_branch','scale_fusion4',target_side='right',target_fraction=.8,
           bends=[(1020,416),(1020,346.8)],color=base.BLUE)
    s.edges[-1].jump_at=(956,416)

    # Input remains above Backbone; keep its centerline and shorten the gap.
    for id in ['input',*s.editable_groups['input'],'input_label','backbone']:
        s.get(id).x+=210
    for id in ['input',*s.editable_groups['input'],'input_label']:s.get(id).y+=60
    plane0=s.groups['p3_plane0']
    inlet_y=plane0.y+plane0.h
    s.nodes['backbone'].y=inlet_y-s.nodes['backbone'].h
    s.edge('backbone_levels','backbone','p3',source_side='bottom',source_fraction=1,
           target_side='left',target_fraction=(inlet_y-s.get('p3').y)/s.get('p3').h,color=base.LINE)

    # The rest of the user's chosen query/decoder/output workflow retains its
    # previous packed placement. Only C moves inside CCFF, as in the reference.
    right={id for id,n in {**s.groups,**s.nodes}.items() if n.x>=1290}
    right_snapshot={id:copy.deepcopy(s.get(id)) for id in right if id!='concat'}
    for id in right:s.get(id).x-=160
    for e in s.edges:
        if e.source in right and e.target in right:
            e.bends=[(x-160,y) for x,y in e.bends]
    for id,old in right_snapshot.items():
        expected=copy.deepcopy(old);expected.x-=160
        assert vars(s.get(id))==vars(expected),id
    c=s.nodes['concat'];c.x=1074;c.y=321;c.w=c.h=30
    c.lines=[('C',22,True,INK)]
    s.edge('coarse_output_c','scale_fusion5','concat',target_side='top',
           bends=[(1089,228.8)],color=base.LINE)
    s.edge('middle_output_c','scale_fusion4','concat',color=base.LINE)
    s.edge('fine_output_c','scale_fusion3','concat',target_side='bottom',
           bends=[(1089,443.2)],color=base.LINE)
    s.editable_groups['encoder']=sorted(((set(s.nodes)|set(s.groups))-before)|{'concat'})

    for id in ['backbone_levels','pyramid_concat','p3_scale_input',
               'top_down_fine','bottom_up_mid','coarse_resample_input']:
        if id=='pyramid_concat':continue
        e=next(e for e in s.edges if e.id==id)
        assert not e.bends,id
    assert s.points(next(e for e in s.edges if e.id=='backbone_levels'))[1][0]-\
           s.points(next(e for e in s.edges if e.id=='backbone_levels'))[0][0]<80
    # Scene.edge's original default was bound before the publication palette
    # was changed. Keep explicitly colored scale paths, normalize only that
    # legacy neutral default so SVG/PDF/draw.io share the same palette.
    for e in s.edges:
        if e.color=='#64748B':e.color=base.LINE
    assert s.groups['fusion_bg'].x<c.x and c.x+c.w<s.groups['fusion_bg'].x+s.groups['fusion_bg'].w
    print('Full reference encoder: AIFI input/output rows, overlapping maps, CCFF skips and internal C.')


def pdf_node(c, n):
    """Native vector geometry; white shape interiors do not erase outlines."""
    if n.text_only or n.ellipse or pub.polygon_points(n) or getattr(n,'vertical',False):
        if getattr(n, 'no_stroke', False):
            clone = copy.copy(n); clone.stroke = clone.fill
            pub.pdf_node(c, clone)
        else:
            pub.pdf_node(c, n)
        return
    c.setStrokeColor(n.stroke); c.setFillColor(n.fill); c.setLineWidth(1.6); c.setDash([])
    radius = getattr(n, 'radius', 8)
    stroke = 0 if getattr(n, 'no_stroke', False) else 1
    fill = 0 if getattr(n, 'outline_only', False) else 1
    if radius:
        c.roundRect(n.x, H-n.y-n.h, n.w, n.h, radius, fill=fill, stroke=stroke)
    else:
        c.rect(n.x, H-n.y-n.h, n.w, n.h, fill=fill, stroke=stroke)
    clone = copy.copy(n); clone.text_only = True
    pub.pdf_node(c, clone)


def pdf_edges(c,s):
    """Keep the reference's line-jump crossings consistent with the SVG."""
    for e in s.edges:
        if not hasattr(e,'jump_at'):
            one=copy.copy(s);one.edges=[e]
            pub.pdf_edges(c,one)
            continue
        pts=s.points(e);x,y=e.jump_at;r=6
        c.setDash([]);c.setStrokeColor('#DFEBE5');c.setLineWidth(6)
        p=c.beginPath();p.moveTo(x-r,H-y)
        p.curveTo(x-r/3,H-y-20/3,x+r/3,H-y-20/3,x+r,H-y)
        c.drawPath(p)
        c.setStrokeColor(e.color);c.setLineWidth(2.2)
        p=c.beginPath();p.moveTo(pts[0][0],H-pts[0][1]);p.lineTo(x-r,H-y)
        p.curveTo(x-r/3,H-y-20/3,x+r/3,H-y-20/3,x+r,H-y)
        for px,py in pts[1:]:p.lineTo(px,H-py)
        c.drawPath(p)
        tx,ty=pts[-1];px,py=pts[-2];a=math.atan2(ty-py,tx-px)
        ux,uy=math.cos(a),math.sin(a);ax,ay=tx-9*ux,ty-9*uy
        p=c.beginPath();p.moveTo(tx,H-ty)
        p.lineTo(ax-4.5*uy,H-(ay+4.5*ux));p.lineTo(ax+4.5*uy,H-(ay-4.5*ux));p.close()
        c.setFillColor(e.color);c.drawPath(p,fill=1,stroke=0)


def verify_drawio(root):
    cells = root.findall('diagram/mxGraphModel/root/mxCell')
    ids = {cell.attrib['id'] for cell in cells}
    assert len(ids) == len(cells), 'Duplicate IDs'
    for cell in cells:
        assert cell.attrib.get('parent', '0') in ids
        if cell.attrib.get('edge') == '1':
            assert cell.attrib['source'] in ids and cell.attrib['target'] in ids
        assert 'data:image' not in cell.attrib.get('style', '')
    vertices = sum(c.attrib.get('vertex') == '1' for c in cells)
    edges = sum(c.attrib.get('edge') == '1' for c in cells)
    print(f'Validated {vertices} editable vertices, {edges} connected edges, no bitmap backgrounds.')


def export(s):
    OUT.mkdir(parents=True, exist_ok=True); PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    old_png=OUT/(STEM+'.png')
    previous=Image.open(old_png).convert('RGB').copy() if old_png.is_file() else None
    svg = s.svg(); ET.fromstring(svg)
    assert '<image' not in svg and all(word not in svg for word in ['UCS-Fusion','DCS-Fusion','CAF'])
    (OUT / (STEM+'.svg')).write_text(svg, encoding='utf-8')
    root = ET.Element('mxfile', host='app.diagrams.net', type='device')
    s.drawio(root); verify_drawio(root); ET.indent(root, space='  ')
    ET.ElementTree(root).write(OUT / (STEM+'.drawio'), encoding='utf-8', xml_declaration=True)
    pdfmetrics.registerFont(TTFont('Diagram','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
    pdfmetrics.registerFont(TTFont('DiagramBold','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))
    c = canvas.Canvas(str(PDF_OUT), pagesize=(W,H))
    c.setTitle('D-FINE-S + AQS - reference-style editable architecture')
    for n in s.groups.values(): pdf_node(c,n)
    pdf_edges(c,s)
    for n in s.nodes.values(): pdf_node(c,n)
    c.showPage(); c.save()
    r = PdfReader(PDF_OUT); assert len(r.pages)==1
    assert '/XObject' not in r.pages[0]['/Resources']
    for label in ['Hybrid Encoder','AQS','Layer 2','Layer 3']:
        assert label in r.pages[0].extract_text()
    for subtitle in ['Upsample + CSP','Downsample + CSP','Q2 + scores S2',
                     'Query refinement','Shared multi-scale memory',
                     '1 x 1 projections','AQS: after L2']:
        assert subtitle not in r.pages[0].extract_text()
    pdf = pdfium.PdfDocument(str(PDF_OUT))
    rendered=pdf[0].render(scale=2).to_pil().convert('RGB')
    if previous is not None and previous.size==rendered.size:
        # Right-hand workflow must be identical except for the requested
        # 160-unit horizontal packing; the bottom title itself stays fixed.
        # C is now inside CCFF; compare the unchanged Memory-to-output area.
        old_right=(2720,0,4400,1176);new_right=(2400,0,4080,1176)
        # A pure integer translation can still round an ellipse's antialias
        # by one 8-bit tone. Geometry is also checked exactly above.
        def same_pixels(a,b):
            return max(hi for lo,hi in ImageChops.difference(a,b).getextrema())<=1
        moved=same_pixels(previous.crop(old_right),rendered.crop(new_right))
        unchanged=same_pixels(previous.crop(new_right),rendered.crop(new_right))
        assert moved or unchanged, 'Right-hand workflow changed beyond spacing'
        footer=(0,1176,4400,1300)
        assert ImageChops.difference(previous.crop(footer),rendered.crop(footer)).getbbox() is None
        print('Query selection/Decoder/AQS/output: unchanged after translation (render tolerance 1/255).')
    rendered.save(OUT / (STEM+'.png'))
    print('Diagram source:', OUT / (STEM+'.drawio'))
    print('Vector PDF:', PDF_OUT)


if __name__ == '__main__':
    export(make_scene())
