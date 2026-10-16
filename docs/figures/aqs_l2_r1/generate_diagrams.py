"""Build editable draw.io pages and matching SVGs; only documentation is written."""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
INK, MUTED, LINE = "#23354B", "#52647A", "#64748B"
BLUE, BLUE_FILL = "#4274A7", "#EDF4FA"
ORANGE, ORANGE_FILL = "#B85A1D", "#FFF1E3"
WHITE = "#FFFFFF"
FONT = "Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif"


@dataclass
class Node:
    id: str
    x: float
    y: float
    w: float
    h: float
    lines: list = field(default_factory=list)
    fill: str = WHITE
    stroke: str = LINE
    dashed: bool = False
    ellipse: bool = False
    text_only: bool = False
    align: str = "center"

    def point(self, side, fraction=.5):
        return {"left": (self.x, self.y+self.h*fraction),
                "right": (self.x+self.w, self.y+self.h*fraction),
                "top": (self.x+self.w*fraction, self.y),
                "bottom": (self.x+self.w*fraction, self.y+self.h)}[side]


@dataclass
class Edge:
    id: str
    source: str
    target: str
    source_side: str = "right"
    target_side: str = "left"
    bends: list = field(default_factory=list)
    color: str = LINE
    dashed: bool = False
    arrow: bool = True
    source_fraction: float = .5
    target_fraction: float = .5


class Scene:
    def __init__(self, name, width, height, description):
        self.name, self.width, self.height, self.description = name, width, height, description
        self.nodes, self.groups, self.edges = {}, {}, []

    def get(self, id):
        return self.nodes.get(id) or self.groups[id]

    def node(self, id, x, y, w, h, title, body=(), *, fill=BLUE_FILL, stroke=BLUE,
             title_size=22, body_size=18, ellipse=False):
        n = Node(id, x, y, w, h, [(title, title_size, True, INK)] +
                 [(t, body_size, False, MUTED) for t in body], fill, stroke, ellipse=ellipse)
        assert id not in self.nodes and id not in self.groups
        self.nodes[id] = n
        return n

    def text(self, id, x, y, w, h, text, *, size=18, color=MUTED, bold=False, align="left"):
        n = Node(id, x, y, w, h, [(text, size, bold, color)], text_only=True, align=align)
        assert id not in self.nodes and id not in self.groups
        self.nodes[id] = n

    def group(self, id, x, y, w, h, *, fill=WHITE, stroke="#C6D1DC", dashed=False):
        assert id not in self.nodes and id not in self.groups
        self.groups[id] = Node(id, x, y, w, h, fill=fill, stroke=stroke, dashed=dashed)

    def edge(self, id, source, target, *, source_side="right", target_side="left",
             bends=(), color=LINE, dashed=False, arrow=True, source_fraction=.5, target_fraction=.5):
        self.get(source); self.get(target)
        self.edges.append(Edge(id, source, target, source_side, target_side, list(bends),
                               color, dashed, arrow, source_fraction, target_fraction))

    def points(self, e):
        return [self.get(e.source).point(e.source_side, e.source_fraction), *e.bends,
                self.get(e.target).point(e.target_side, e.target_fraction)]

    def validate(self):
        all_nodes = list(self.groups.values()) + list(self.nodes.values())
        assert len({n.id for n in all_nodes}) == len(all_nodes)
        for n in all_nodes:
            assert 0 <= n.x and 0 <= n.y and n.x+n.w <= self.width and n.y+n.h <= self.height, n
        for e in self.edges:
            for x, y in self.points(e):
                assert 0 <= x <= self.width and 0 <= y <= self.height, (e.id, x, y)

    @staticmethod
    def svg_node(n):
        bits = []
        dash = ' stroke-dasharray="7 5"' if n.dashed else ""
        if not n.text_only:
            if n.ellipse:
                bits.append(f'<ellipse cx="{n.x+n.w/2:g}" cy="{n.y+n.h/2:g}" rx="{n.w/2:g}" ry="{n.h/2:g}" fill="{n.fill}" stroke="{n.stroke}" stroke-width="2"{dash}/>')
            else:
                bits.append(f'<rect x="{n.x:g}" y="{n.y:g}" width="{n.w:g}" height="{n.h:g}" rx="8" fill="{n.fill}" stroke="{n.stroke}" stroke-width="2"{dash}/>')
        if n.lines:
            heights = [size*1.5 for _, size, _, _ in n.lines]
            y = n.y + (n.h-sum(heights))/2
            x = n.x if n.text_only and n.align == "left" else n.x+n.w/2
            anchor = "start" if n.text_only and n.align == "left" else "middle"
            for (label, size, bold, color), lh in zip(n.lines, heights):
                y += lh/2
                bits.append(f'<text x="{x:g}" y="{y:g}" dominant-baseline="central" text-anchor="{anchor}" font-size="{size:g}" font-weight="{700 if bold else 400}" fill="{color}">{escape(label)}</text>')
                y += lh/2
        return "\n".join(bits)

    def svg(self):
        self.validate()
        bits = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">',
                f'<title id="title">{escape(self.name)}</title><desc id="desc">{escape(self.description)}</desc>', '<defs>']
        for i, color in enumerate((LINE, BLUE, ORANGE)):
            bits.append(f'<marker id="arrow{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="{color}"/></marker>')
        bits += ['</defs>', f'<rect width="{self.width}" height="{self.height}" fill="white"/>',
                 f'<g font-family="{escape(FONT, quote=True)}">']
        bits += [self.svg_node(n) for n in self.groups.values()]
        colors = {LINE: 0, BLUE: 1, ORANGE: 2}
        for e in self.edges:
            points = " ".join(f"{x:g},{y:g}" for x,y in self.points(e))
            dash = ' stroke-dasharray="7 5"' if e.dashed else ""
            marker = f' marker-end="url(#arrow{colors[e.color]})"' if e.arrow else ""
            bits.append(f'<polyline id="{e.id}" points="{points}" fill="none" stroke="{e.color}" stroke-width="2.2" stroke-linejoin="round"{dash}{marker}/>')
        bits += [self.svg_node(n) for n in self.nodes.values()]
        return "\n".join(bits + ['</g>', '</svg>']) + "\n"

    def drawio(self, parent):
        diagram = ET.SubElement(parent, "diagram", id=self.name.replace(" ", "-"), name=self.name)
        model = ET.SubElement(diagram, "mxGraphModel", dx=str(self.width), dy=str(self.height), grid="0", gridSize="10",
                              guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1",
                              pageWidth=str(self.width), pageHeight=str(self.height), math="0", shadow="0")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        for n in list(self.groups.values()) + list(self.nodes.values()):
            value = "".join(f'<div style="font-size:{size:g}px;font-weight:{700 if bold else 400};color:{color};line-height:1.5">{escape(label)}</div>' for label,size,bold,color in n.lines)
            if n.text_only:
                style = f'text;html=1;whiteSpace=wrap;align={n.align};verticalAlign=middle;strokeColor=none;fillColor=none;fontFamily=Arial;spacing=0;'
            else:
                shape = "ellipse;" if n.ellipse else "rounded=1;arcSize=8;"
                style = f'{shape}html=1;whiteSpace=wrap;align=center;verticalAlign=middle;fillColor={n.fill};strokeColor={n.stroke};strokeWidth=2;fontFamily=Arial;spacing=8;'
                if n.dashed: style += "dashed=1;dashPattern=7 5;"
            cell = ET.SubElement(root, "mxCell", id=n.id, value=value, style=style, vertex="1", parent="1")
            ET.SubElement(cell, "mxGeometry", x=f"{n.x:g}", y=f"{n.y:g}", width=f"{n.w:g}", height=f"{n.h:g}", attrib={"as":"geometry"})
        port = {"left": lambda f:(0,f), "right": lambda f:(1,f), "top": lambda f:(f,0), "bottom": lambda f:(f,1)}
        for e in self.edges:
            ex,ey = port[e.source_side](e.source_fraction); ix,iy = port[e.target_side](e.target_fraction)
            style = f'edgeStyle=none;html=1;rounded=0;strokeColor={e.color};strokeWidth=2.2;endArrow={"block" if e.arrow else "none"};endFill=1;exitX={ex:g};exitY={ey:g};exitDx=0;exitDy=0;entryX={ix:g};entryY={iy:g};entryDx=0;entryDy=0;'
            if e.dashed: style += "dashed=1;dashPattern=7 5;"
            cell = ET.SubElement(root, "mxCell", id=e.id, value="", style=style, edge="1", parent="1", source=e.source, target=e.target)
            geo = ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as":"geometry"})
            if e.bends:
                arr = ET.SubElement(geo, "Array", attrib={"as":"points"})
                for x,y in e.bends: ET.SubElement(arr, "mxPoint", x=f"{x:g}", y=f"{y:g}")


def overview():
    s = Scene("D-FINE-S + AQS-L2", 1960, 1070,
              "D-FINE-S inference: AQS follows layer 2, updates query content, preserves 300 queries and feeds layer 3. Multi-scale memory and original iterative box paths are retained.")
    s.text("title", 60, 35, 1500, 50, "D-FINE-S + AQS-L2 整体架构", size=32, color=INK, bold=True)
    s.text("subtitle", 60, 95, 1740, 35, "当前配置：AQS-L2 R1 · SCB-S三类 · 640 × 640 · 推理主路径（训练DN及辅助监督省略）", size=19)
    s.node("image", 60, 205, 145, 180, "输入图像", ["640 × 640", "RGB"], fill="#F4F6F8", stroke=LINE)
    s.group("backbone_group", 245, 175, 270, 250)
    s.text("backbone_title", 270, 185, 220, 42, "HGNetV2-B0", size=24, color=INK, bold=True, align="center")
    s.node("p3", 270, 240, 220, 45, "P3  ·  256 × 80 × 80", title_size=18)
    s.node("p4", 270, 298, 220, 45, "P4  ·  512 × 40 × 40", title_size=18)
    s.node("p5", 270, 356, 220, 45, "P5  ·  1024 × 20 × 20", title_size=18)
    s.group("encoder_group", 560, 175, 390, 250)
    s.text("encoder_title", 580, 185, 350, 42, "Hybrid Encoder", size=24, color=INK, bold=True, align="center")
    s.node("projection", 585, 239, 340, 46, "1 × 1 投影：各尺度 → 256通道", title_size=19)
    s.node("aifi", 585, 296, 260, 46, "P5 Transformer / AIFI", title_size=18)
    s.text("p34_bypass_label", 850, 303, 47, 28, "P3/P4", size=14, color=BLUE, align="center")
    s.node("fpnpan", 585, 353, 340, 46, "跨尺度融合：FPN → PAN", title_size=20)
    s.node("memory", 990, 215, 870, 125, "多尺度特征 / Decoder Memory", ["P3: 80 × 80    ·    P4: 40 × 40    ·    P5: 20 × 20", "各尺度256通道；提供各Decoder层的跨注意力特征"], title_size=23)
    s.edge("image_backbone", "image", "backbone_group")
    for name in ("p3", "p4", "p5"):
        s.edge("backbone_"+name, name, "projection", bends=[(535,s.get(name).point("right")[1]), (535,262)])
    s.edge("projection_aifi", "projection", "aifi", source_side="bottom", target_side="top", source_fraction=130/340)
    s.edge("aifi_fpnpan", "aifi", "fpnpan", source_side="bottom", target_side="top", target_fraction=130/340)
    s.edge("p34_bypass", "projection", "fpnpan", source_side="bottom", target_side="top", source_fraction=.93, target_fraction=.93, color=BLUE)
    s.edge("encoder_memory", "fpnpan", "memory", bends=[(973,376), (973,277.5)])
    s.text("encoder_note", 245, 450, 490, 40, "骨干与编码器保持原结构", size=17, align="center")
    s.node("query_init", 560, 575, 250, 155, "原有候选初始化", ["Encoder预测 + Top300", "Q₀：300 × 256", "初始参考框 b₀"], title_size=22)
    s.edge("memory_query_init", "fpnpan", "query_init", source_side="bottom", target_side="top", bends=[(755,515), (685,515)])
    for name,x in [("l1",905), ("l2",1115), ("l3",1545)]:
        s.node(name,x,575,145,155,"Decoder "+name.upper(),["Self-attn","Cross-attn","FFN / FDR"],title_size=20,body_size=17)
    s.node("aqs",1325,575,155,155,"AQS-L2",["置信门控","上下文融合","query残差更新"],fill=ORANGE_FILL,stroke=ORANGE,title_size=25)
    s.node("heads",1745,575,160,155,"原有输出头",["Cls + LQE","FDR定位框","类别分数与bbox"],title_size=21,body_size=17)
    s.node("postprocess",1695,870,210,120,"原有后处理",["Top300 query-class","hand / read / write"],title_size=22,body_size=17)
    for id,src,dst in [("init_l1","query_init","l1"),("l1_l2","l1","l2"),("l2_aqs","l2","aqs"),("aqs_l3","aqs","l3"),("l3_heads","l3","heads")]:
        s.edge(id,src,dst,color=ORANGE if "aqs" in id else LINE)
    s.edge("heads_post","heads","postprocess",source_side="bottom",target_side="top",bends=[(1825,805),(1800,805)])
    s.node("gate_scores",1305,440,235,85,"L2 ClsHead + LQE",["S₂：AQS门控依据"],fill=WHITE,stroke=ORANGE,title_size=20,body_size=17)
    s.edge("l2_gate_scores","l2","gate_scores",bends=[(1280,652.5),(1280,482.5)],color=ORANGE)
    s.edge("scores_aqs","gate_scores","aqs",source_side="bottom",target_side="top",bends=[(1422.5,550),(1402.5,550)],color=ORANGE)
    for layer,x in [("l1",930),("l2",1140),("l3",1570)]:
        s.edge("memory_"+layer,"memory",layer,source_side="bottom",target_side="top",source_fraction=.57,
               target_fraction=(x-s.get(layer).x)/s.get(layer).w,bends=[(1485.9,430),(x,430)],color=BLUE)
    s.text("memory_bus_label",1040,374,430,32,"共享多尺度Memory → 各Decoder层",size=17,color=BLUE)
    s.edge("ref_init","query_init","l1",source_side="bottom",target_side="bottom",bends=[(685,810),(977.5,810)],dashed=True)
    s.edge("ref_12","l1","l2",source_side="bottom",target_side="bottom",bends=[(977.5,782),(1187.5,782)],dashed=True)
    s.edge("ref_23","l2","l3",source_side="bottom",target_side="bottom",bends=[(1187.5,822),(1617.5,822)],dashed=True)
    s.text("ref_init_label",545,827,365,28,"参考框初始化与迭代：原路径保留",size=17)
    s.text("ref_label",1030,837,630,30,"L2当前框 / FDR迭代不直接经AQS；L3框可间接受query更新影响",size=16)
    s.text("q_out",1481,613,64,30,"Q₂′",size=19,color=ORANGE,align="center")
    s.group("legend_blue",60,870,22,22,fill=BLUE_FILL,stroke=BLUE)
    s.text("legend_blue_label",95,861,390,40,"原D-FINE组件 / 特征路径",size=18)
    s.group("legend_aqs",60,915,22,22,fill=ORANGE_FILL,stroke=ORANGE)
    s.text("legend_aqs_label",95,906,410,40,"新增AQS / query更新路径",size=18)
    s.text("protocol",560,905,1010,65,"AQS初值：τ₀ = 0.50，ρ₀ = 0.10；T = 0.10固定；τ与ρ继续学习。",size=18)
    s.text("key_note",560,969,1230,45,"保留300个query，不新增TopK选择器，不改变原损失函数，不叠加BoxLocal或SERF。",size=18,color=INK)
    return s


def detail():
    s = Scene("AQS-L2 internal structure",1980,1100,
              "Max sigmoid score, learned threshold, hard forward gate with STE gradient, gated query mean, LayerNorm/MLP fusion, tanh residual scale and gated query addition. All 300 queries are retained.")
    s.text("title",60,35,1750,50,"AQS-L2：自适应query门控与上下文残差增强",size=31,color=INK,bold=True)
    s.text("subtitle",60,94,1750,38,"展示普通检测query的计算；与当前 AdaptiveQuerySelectionRefiner 实现一致",size=19)
    s.node("scores",60,175,200,125,"L2类别logits S₂",["ClsHead₂ + LQE₂","300 × 3"],title_size=21)
    s.node("importance",315,175,205,125,"置信度重要性",["sᵢ = max_c sigmoid(Sᵢc)"],title_size=22,body_size=17)
    s.node("gate",570,165,255,195,"硬门控 + STE",["g_h = 1[sᵢ ≥ τ]","g_s = sigmoid((sᵢ − τ)/T)","前向：硬选择；反向：软梯度","T = 0.10（固定）"],fill=ORANGE_FILL,stroke=ORANGE,title_size=24,body_size=17)
    s.node("threshold",315,365,220,110,"可学习阈值 τ",["τ = sigmoid(θ)","τ₀ = 0.50"],fill=ORANGE_FILL,stroke=ORANGE,title_size=22)
    s.edge("score_importance","scores","importance")
    s.edge("importance_gate","importance","gate",target_fraction=.372)
    s.edge("threshold_gate","threshold","gate",bends=[(551,420),(551,324)],target_fraction=.815,color=ORANGE)
    s.node("query",60,635,185,115,"L2 query Q₂",["300 × 256","保留全部query"],title_size=23)
    s.node("norm",300,635,165,115,"LayerNorm",["LN(qᵢ)"],title_size=22)
    s.node("pool",855,390,430,155,"选中query的上下文池化",["c = Σᵢ gᵢqᵢ / max(Σᵢ gᵢ, 1)","将c广播至所有query位置"],fill=ORANGE_FILL,stroke=ORANGE,title_size=23,body_size=20)
    s.node("concat",855,635,205,115,"拼接",["[ LN(qᵢ) ; c ]","256 + 256 → 512"],title_size=23)
    s.node("mlp",1115,635,230,115,"融合MLP",["Linear → SiLU → Linear","512 → 256 → 256"],title_size=23,body_size=17)
    s.node("scale",1410,420,210,115,"可学习残差缩放",["β = tanh(ρ)","ρ₀ = 0.10"],fill=ORANGE_FILL,stroke=ORANGE,title_size=22)
    s.node("multiply",1410,625,210,135,"门控残差",["β · gᵢ · Δqᵢ","Δqᵢ = MLP([LN(qᵢ); c])"],fill=ORANGE_FILL,stroke=ORANGE,title_size=24,body_size=17)
    s.node("add",1680,660,65,65,"+",fill=WHITE,stroke=ORANGE,ellipse=True,title_size=35)
    s.node("out",1790,635,160,115,"增强query Q₂′",["300 × 256"],fill=ORANGE_FILL,stroke=ORANGE,title_size=21)
    s.node("l3",1790,895,160,100,"Decoder L3",["继续分类与定位"],title_size=21,body_size=17)
    s.edge("query_norm","query","norm")
    s.edge("query_pool","query","pool",bends=[(273,692.5),(273,570),(795,570),(795,467.5)])
    s.edge("gate_pool","gate","pool",source_side="bottom",target_side="top",bends=[(697.5,378),(1070,378)],color=ORANGE)
    s.edge("gate_multiply","gate","multiply",source_side="right",target_side="top",bends=[(1660,262.5),(1660,585),(1554.9,585)],target_fraction=.69,color=ORANGE)
    s.edge("pool_concat","pool","concat",source_side="bottom",target_side="top",bends=[(1070,595),(957.5,595)],color=ORANGE)
    s.edge("norm_concat","norm","concat")
    s.edge("concat_mlp","concat","mlp")
    s.edge("mlp_multiply","mlp","multiply",color=ORANGE)
    s.edge("scale_multiply","scale","multiply",source_side="bottom",target_side="top",source_fraction=.28,target_fraction=.28,color=ORANGE)
    s.edge("multiply_add","multiply","add",color=ORANGE)
    s.edge("add_out","add","out",color=ORANGE)
    s.edge("query_skip","query","add",source_side="bottom",target_side="bottom",bends=[(152.5,827),(1712.5,827)])
    s.edge("out_l3","out","l3",source_side="bottom",target_side="top",color=ORANGE)
    s.text("gate_label",900,220,620,30,"门控g：选择池化query，同时控制残差写入",size=18,color=ORANGE,align="center")
    s.text("ln_label",490,650,320,32,"每个query的归一化特征",size=18,align="center")
    s.text("skip_label",610,835,820,30,"恒等路径：gᵢ = 0时，原query保持不变；不是删除query",size=18,align="center")
    s.text("equation",60,906,1580,45,"qᵢ′ = qᵢ + tanh(ρ) · gᵢ · MLP([LN(qᵢ); c])",size=25,color=INK,bold=True)
    s.text("ste_note",60,962,1580,34,"STE：g = stopgrad(g_h − g_s) + g_s；MLP最后一层零初始化，使初始残差为0。",size=18)
    s.text("dn_note",60,1009,1830,34,"训练时DN与普通query分别计算，不相互池化；AQS不直接输入或修改bbox，L3可受query更新间接影响。",size=18)
    return s


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scenes = [("dfine_s_aqs_l2_architecture",overview()),("aqs_l2_module",detail())]
    file = ET.Element("mxfile",host="app.diagrams.net",agent="D-FINE documentation generator",version="24.7.17",type="device")
    for stem,scene in scenes:
        scene.validate(); svg = scene.svg(); ET.fromstring(svg)
        (OUT/(stem+".svg")).write_text(svg,encoding="utf-8")
        scene.drawio(file)
    ET.indent(file,space="  ")
    target = OUT/"dfine_s_aqs_l2.drawio"
    ET.ElementTree(file).write(target,encoding="utf-8",xml_declaration=True)
    parsed = ET.parse(target).getroot()
    assert len(parsed.findall("diagram")) == 2
    for diagram in parsed.findall("diagram"):
        cells = diagram.findall("mxGraphModel/root/mxCell"); ids = {c.attrib["id"] for c in cells}
        assert len(ids) == len(cells)
        assert all(c.attrib.get("source") in ids and c.attrib.get("target") in ids for c in cells if c.attrib.get("edge") == "1")
    print("Generated two SVGs and one native two-page draw.io file:",OUT)


if __name__ == "__main__":
    main()
