"""Editable IEEE-sized conceptual CSI figure, with one shared scene per language.

All waveforms are deterministic synthetic illustrations. Scores are symbols,
not measurements. No experimental data are read or inferred by this script.
Run with Python + reportlab, then use Poppler to render PDF previews.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from pathlib import Path
import urllib.parse
import xml.etree.ElementTree as ET
import zlib

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PDF_DIR = ROOT / 'output' / 'pdf'
W, H = 1000, 765
SCALE = 7.16 * 72 / W
C = dict(ink='#252B33', muted='#525D69', rule='#C9CFD5', pale='#F5F6F8',
         blue='#0072B2', orange='#B76500', purple='#8056A3',
         blue_bg='#EDF5FA', orange_bg='#FCF3E8', purple_bg='#F4EFF8')
FONT_DIR = Path('C:/Windows/Fonts')
pdfmetrics.registerFont(TTFont('Arial', str(FONT_DIR / 'arial.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold', str(FONT_DIR / 'arialbd.ttf')))
pdfmetrics.registerFont(TTFont('CJK', str(FONT_DIR / 'msyh.ttc'), subfontIndex=0))
pdfmetrics.registerFont(TTFont('CJK-Bold', str(FONT_DIR / 'msyhbd.ttc'), subfontIndex=0))


class Scene:
    def __init__(self, lang):
        self.lang, self.items, self.warnings = lang, [], []

    def text(self, x, y, value, size=18, color=None, bold=False, align='left', width=None):
        font = ('CJK' if self.lang == 'zh' else 'Arial') + ('-Bold' if bold else '')
        tw = pdfmetrics.stringWidth(value, font, size)
        if width and tw > width + 0.5:
            self.warnings.append(f'Text overflows ({tw:.1f} > {width}): {value}')
        self.items.append(dict(kind='text', x=x, y=y, value=value, size=size,
                               color=color or C['ink'], bold=bold, align=align,
                               font=font, tw=tw))

    def line(self, pts, color=None, sw=1.6, dash=None, arrow=False):
        self.items.append(dict(kind='line', pts=pts, color=color or C['rule'],
                               sw=sw, dash=dash, arrow=arrow))

    def rect(self, x, y, w, h, fill='white', stroke=None, sw=1.3, radius=0):
        if fill == 'white': fill = '#FFFFFF'
        self.items.append(dict(kind='rect', x=x, y=y, w=w, h=h, fill=fill,
                               stroke=stroke, sw=sw, radius=radius))

    def dot(self, x, y, radius=3, fill=None):
        self.items.append(dict(kind='dot', x=x, y=y, radius=radius, fill=fill or C['ink']))

    def title(self, x, y, width, value):
        self.text(x, y, value, 20, bold=True, width=width)
        self.line([(x, y+31), (x+width, y+31)], C['rule'], 1.4)

    def badge(self, x, y, i, width=48, height=38):
        col = [C['blue'], C['orange'], C['purple']][i-1]
        bg = [C['blue_bg'], C['orange_bg'], C['purple_bg']][i-1]
        self.rect(x, y, width, height, bg, col, 1.6, 3)
        self.text(x+width/2, y+(height-21)/2, f'P{i}', 18.5, col, True, 'center')

    def wave(self, x, y, w, h, variant, color, dash=None):
        pts = []
        for j in range(121):
            t = j / 120
            env = .72 + .2 * math.sin(2*math.pi*t)
            if variant == 0:
                v = env*(.63*math.sin(2*math.pi*4*t)+.25*math.sin(2*math.pi*9*t))
            elif variant == 1:
                v = .2*math.sin(2*math.pi*4*t)+.1*math.sin(2*math.pi*t)
            else:
                v = .40*math.sin(2*math.pi*3*t+.6)+.19*math.sin(2*math.pi*7*t)
                v += .12 if t > .55 else -.1
            pts.append((x+t*w, y+h/2-v*h*.46))
        self.line([(x,y+h+5),(x+w,y+h+5)], C['rule'], .9)
        self.line(pts,color,2.2,dash)

    def score(self, cx, y, index, color=None):
        # Explicit text runs avoid unsupported Unicode subscript glyphs.
        font='CJK' if self.lang=='zh' else 'Arial'
        a=pdfmetrics.stringWidth('s',font,22)
        b=pdfmetrics.stringWidth(index,font,16.6)
        self.text(cx-(a+b)/2,y,'s',22,color)
        self.text(cx-(a+b)/2+a,y+8,index,16.6,color)


def build(lang):
    s=Scene(lang)
    def T(en, zh): return zh if lang=='zh' else en
    s.title(20,14,465,T('(a) Raw CSI contains mixed effects','(a) 原始 CSI 混合了动作信息与干扰'))
    s.title(535,14,445,T('(b) Preprocessing changes retained cues','(b) 预处理改变保留的信息'))

    # (a) Schematic amplitude: motion component, nuisance drift, impulsive noise,
    # and a gain change. Phase effects are stated separately, not mislabeled as
    # amplitude effects or universally equated with inter-carrier interference.
    s.text(66,58,T('CSI amplitude (a.u.)','CSI 幅值（任意单位）'),17.5,color=C['muted'])
    s.text(80,92,T('Motion response','动作响应'),17.5,C['blue'])
    s.text(226,92,T('Outlier','异常尖峰'),17.5,C['orange'])
    s.text(350,92,T('Gain change','增益变化'),17.5,C['muted'])
    px,py,pw,ph=66,124,393,126
    pts=[]
    base=[]
    for i in range(221):
        t=i/220
        gain=1.0 if t<.72 else 1.5
        motion=.42*math.sin(2*math.pi*6*t)+.07*math.sin(2*math.pi*12*t)
        noise=.034*math.sin(2*math.pi*53*t)+.025*math.sin(2*math.pi*81*t)
        trend=.12*t+(.12 if t>=.72 else 0)
        outlier=.70*math.exp(-((t-.46)/.006)**2)
        v=gain*motion+noise+trend+outlier
        pts.append((px+t*pw, py+ph*.65-v*ph*.60))
        base.append((px+t*pw,py+ph*.65-trend*ph*.6))
    s.line([(px,118),(px,258),(468,258)],C['muted'],1.5,arrow=True)
    s.line(base,C['muted'],1.6,[7,5])
    s.line(pts,C['blue'],2.1)
    s.line([(80+65,114),(pts[47][0],pts[47][1]-8)],C['blue'],1.3,arrow=True)
    spike=min(pts[97:106], key=lambda p:p[1])
    s.line([(251,114),(spike[0],spike[1]-7)],C['orange'],1.3,arrow=True)
    s.dot(*spike,3.4,C['orange'])
    s.line([(px+pw*.72,122),(px+pw*.72,246)],C['rule'],1.2,[4,4])
    s.line([(396,114),(px+pw*.73,160)],C['muted'],1.3,arrow=True)
    s.text(451,265,T('Time','时间'),17.5,C['muted'],align='right')
    s.line([(76,278),(109,278)],C['muted'],1.6,[7,5])
    s.text(119,269,T('Baseline drift','基线漂移'),17.5,C['muted'])
    s.rect(20,298,465,58,C['pale'],radius=3)
    s.text(32,305,T('Amplitude: gain variation, drift, and outliers','幅值：增益变化、漂移与异常值'),17.5,width=441)
    s.text(32,330,T('Phase: offsets and other distortions','相位：偏移及其他失真'),17.5,width=441)

    # (b) Alternative choices, not a mandatory order or universal quality rank.
    s.rect(535,60,445,66,C['pale'],C['rule'],1.2,3)
    s.text(757.5,68,T('Denoising / Outliers / Phase calibration','去噪 / 异常值处理 / 相位校准'),17.5,align='center',width=421)
    s.text(757.5,94,T('Normalization / Resampling','归一化 / 重采样'),17.5,align='center',width=421)
    s.text(535,136,T('Vary methods, order, and parameters','改变方法、顺序与参数'),17.5,C['muted'],width=445)
    rows=[(171,'Structure retained','保留时序结构'),(232,'Details attenuated','削弱动作细节'),(293,'Cues transformed','改变特征表征')]
    for i,(y,en,zh) in enumerate(rows,1):
        col=[C['blue'],C['orange'],C['purple']][i-1]
        dash=[None,[7,4],[7,3,2,3]][i-1]
        s.badge(535,y+3,i)
        s.text(601,y+9,T(en,zh),17.5,width=170)
        s.wave(796,y,179,37,i-1,col,dash)
    s.text(535,339,T('Possible outcomes; schematic traces','不同处理的可能结果；波形仅为示意'),17,C['muted'],width=445)

    s.title(20,385,525,T('(c) Compare within a fixed architecture','(c) 固定模型架构下的预处理对照'))
    s.title(585,385,395,T('(d) Model-dependent effects','(d) 进一步比较不同模型'))
    s.rect(20,431,525,52,C['pale'],C['rule'],1.2,3)
    s.text(282.5,436,T('Same data split and training protocol','相同数据划分与训练设置'),18,align='center',width=505)
    s.text(282.5,459,T('Same architecture; trained separately','相同模型架构；各自独立训练'),17.5,C['muted'],align='center',width=505)
    s.text(129,496,T('Processed input','预处理后输入'),17.5,align='center')
    s.text(323,496,T('Model M','模型 M'),17.5,align='center')
    s.text(499,496,T('Score','识别性能'),17.5,align='center')
    for i,y in enumerate([526,582,638],1):
        col=[C['blue'],C['orange'],C['purple']][i-1]
        s.badge(23,y,i,48,38)
        s.wave(84,y+3,104,28,i-1,col,[None,[7,4],[7,3,2,3]][i-1])
        s.line([(199,y+19),(247,y+19)],C['muted'],1.6,arrow=True)
        s.rect(255,y,136,38,'white',C['muted'],1.5,3)
        s.text(323,y+8,T('Train + test','训练与测试'),18,align='center')
        s.line([(399,y+19),(446,y+19)],C['muted'],1.6,arrow=True)
        s.rect(455,y,87,38,'white',col,1.5,3)
        s.score(498.5,y+5,str(i),col)
    s.text(20,694,T('Different inputs may yield different scores.','输入不同，识别性能可能不同。'),18,width=525)

    s.text(585,432,T('Feature extraction differs across models.','模型的特征提取能力存在差异。'),17.5,width=395)
    s.text(585,458,T('Compare every preprocessing-model pair:','比较各个“预处理—模型”组合：'),17.5,width=395)
    tx,ty,tw=585,493,395
    s.rect(tx,ty,tw,36,C['pale'])
    s.line([(tx,ty),(tx+tw,ty)],C['muted'],1.6)
    s.text(620,ty+7,T('Input','输入'),17.5,align='center')
    s.text(747,ty+7,T('Model A','模型 A'),17.5,align='center')
    s.text(903,ty+7,T('Model B','模型 B'),17.5,align='center')
    s.line([(tx,ty+36),(tx+tw,ty+36)],C['rule'],1.2)
    for i,y in enumerate([534,576,618],1):
        col=[C['blue'],C['orange'],C['purple']][i-1]
        s.text(620,y+4,f'P{i}',18.5,col,True,'center')
        s.score(747,y+2,f'{i},A')
        s.score(903,y+2,f'{i},B')
        if i<3:s.line([(tx,y+36),(tx+tw,y+36)],C['rule'],.9)
    s.line([(tx,658),(tx+tw,658)],C['muted'],1.6)
    s.rect(585,676,4,40,C['blue'])
    s.text(601,674,T('Preprocessing effects may depend','预处理对识别性能的影响'),18,width=379)
    s.text(601,698,T('on the model used.','可能随模型而变化。'),18,width=379)

    s.line([(20,729),(980,729)],C['rule'],1)
    s.text(500,740,T('Conceptual illustration: synthetic waveforms and symbolic scores.',
                       '概念示意图：波形为人工构造，性能以符号表示，不代表实验结果。'),16.6,C['muted'],align='center')
    return s


def to_pdf(s, path):
    c=canvas.Canvas(str(path),pagesize=(W*SCALE,H*SCALE),pageCompression=1,initialFontName='Arial')
    c.setTitle('CSI preprocessing and model-dependent recognition: conceptual illustration')
    c.setAuthor('')
    c.scale(SCALE,SCALE)
    c.setFillColor(HexColor('#FFFFFF'));c.rect(0,0,W,H,fill=1,stroke=0)
    for e in s.items:
        kind=e['kind']
        if kind=='text':
            c.setFillColor(HexColor(e['color']));c.setFont(e['font'],e['size'])
            # Shared top-of-em reference for SVG/mxGraph/PDF.
            x=e['x']-({'left':0,'center':.5,'right':1}[e['align']])*e['tw']
            c.drawString(x,H-e['y']-.82*e['size'],e['value'])
        elif kind=='rect':
            c.setFillColor(HexColor(e['fill']))
            if e['stroke']:c.setStrokeColor(HexColor(e['stroke']))
            c.setLineWidth(e['sw']);c.setDash()
            c.roundRect(e['x'],H-e['y']-e['h'],e['w'],e['h'],e['radius'],
                        fill=1,stroke=bool(e['stroke']))
        elif kind=='dot':
            c.setFillColor(HexColor(e['fill']));c.circle(e['x'],H-e['y'],e['radius'],fill=1,stroke=0)
        elif kind=='line':
            c.setStrokeColor(HexColor(e['color']));c.setLineWidth(e['sw'])
            c.setLineJoin(1);c.setLineCap(1);c.setDash(e['dash'] or [])
            p=c.beginPath();p.moveTo(e['pts'][0][0],H-e['pts'][0][1])
            for x,y in e['pts'][1:]:p.lineTo(x,H-y)
            c.drawPath(p)
            if e['arrow']:
                x,y=e['pts'][-1];a,b=e['pts'][-2];ang=math.atan2(y-b,x-a)
                tri=[(x,y),(x-8*math.cos(ang)+3.7*math.sin(ang),y-8*math.sin(ang)-3.7*math.cos(ang)),
                     (x-8*math.cos(ang)-3.7*math.sin(ang),y-8*math.sin(ang)+3.7*math.cos(ang))]
                p=c.beginPath();p.moveTo(tri[0][0],H-tri[0][1])
                for xx,yy in tri[1:]:p.lineTo(xx,H-yy)
                p.close();c.setFillColor(HexColor(e['color']));c.drawPath(p,stroke=0,fill=1)
    c.showPage();c.save()


def to_svg(s, path):
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="181.864mm" height="{H*SCALE/72*25.4:.3f}mm" viewBox="0 0 {W} {H}">',
           '<title>CSI preprocessing and model-dependent recognition</title>',
           '<desc>Conceptual illustration. All waveforms are synthetic; scores are symbolic.</desc>',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    family='Microsoft YaHei, Arial, sans-serif' if s.lang=='zh' else 'Arial, Helvetica, sans-serif'
    for e in s.items:
        if e['kind']=='text':
            anchor={'left':'start','center':'middle','right':'end'}[e['align']]
            parts.append(f'<text x="{e["x"]:.2f}" y="{e["y"]+.82*e["size"]:.2f}" text-anchor="{anchor}" font-family="{family}" font-size="{e["size"]}" font-weight="{700 if e["bold"] else 400}" fill="{e["color"]}">{html.escape(e["value"])}</text>')
        elif e['kind']=='rect':
            parts.append(f'<rect x="{e["x"]}" y="{e["y"]}" width="{e["w"]}" height="{e["h"]}" rx="{e["radius"]}" fill="{e["fill"]}" stroke="{e["stroke"] or "none"}" stroke-width="{e["sw"]}"/>')
        elif e['kind']=='dot':
            parts.append(f'<circle cx="{e["x"]}" cy="{e["y"]}" r="{e["radius"]}" fill="{e["fill"]}"/>')
        else:
            dash=f' stroke-dasharray="{" ".join(map(str,e["dash"]))}"' if e['dash'] else ''
            pts=' '.join(f'{x:.2f},{y:.2f}' for x,y in e['pts'])
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{e["color"]}" stroke-width="{e["sw"]}" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
            if e['arrow']:
                x,y=e['pts'][-1];a,b=e['pts'][-2];ang=math.atan2(y-b,x-a)
                tri=[(x,y),(x-8*math.cos(ang)+3.7*math.sin(ang),y-8*math.sin(ang)-3.7*math.cos(ang)),(x-8*math.cos(ang)-3.7*math.sin(ang),y-8*math.sin(ang)+3.7*math.cos(ang))]
                points=' '.join(f'{xx:.2f},{yy:.2f}' for xx,yy in tri)
                parts.append(f'<polygon points="{points}" fill="{e["color"]}"/>')
    parts.append('</svg>');path.write_text('\n'.join(parts),encoding='utf-8')


def graph_model(s):
    model=ET.Element('mxGraphModel',dict(dx='1100',dy='820',grid='0',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),math='0',shadow='0',background='#FFFFFF'))
    root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
    def geometry(cell,x,y,w,h):
        return ET.SubElement(cell,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
    for idx,e in enumerate(s.items,2):
        attrs={'id':str(idx),'parent':'1'}
        if e['kind']=='line':
            style=f'edgeStyle=none;rounded=0;html=0;endArrow={"block" if e["arrow"] else "none"};endFill=1;endSize=7;strokeColor={e["color"]};strokeWidth={e["sw"]};'
            if e['dash']:style+='dashed=1;fixDash=1;dashPattern='+' '.join(map(str,e['dash']))+';'
            cell=ET.SubElement(root,'mxCell',attrs|{'style':style,'edge':'1'})
            g=ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'})
            first,last=e['pts'][0],e['pts'][-1]
            ET.SubElement(g,'mxPoint',{'x':str(first[0]),'y':str(first[1]),'as':'sourcePoint'})
            ET.SubElement(g,'mxPoint',{'x':str(last[0]),'y':str(last[1]),'as':'targetPoint'})
            if len(e['pts'])>2:
                arr=ET.SubElement(g,'Array',{'as':'points'})
                for x,y in e['pts'][1:-1]:ET.SubElement(arr,'mxPoint',{'x':str(round(x,3)),'y':str(round(y,3))})
        elif e['kind']=='text':
            # Generous invisible text boxes; no HTML font fallback/export issues.
            width=max(e['tw']+8,20);height=e['size']*1.15
            x=e['x']-{'left':0,'center':.5,'right':1}[e['align']]*width
            family='Microsoft YaHei' if s.lang=='zh' else 'Arial'
            style=f'text;html=0;strokeColor=none;fillColor=none;align={e["align"]};verticalAlign=top;whiteSpace=nowrap;overflow=visible;spacing=0;fontFamily={family};fontSize={e["size"]};fontColor={e["color"]};fontStyle={1 if e["bold"] else 0};'
            cell=ET.SubElement(root,'mxCell',attrs|{'value':e['value'],'style':style,'vertex':'1'})
            geometry(cell,x,e['y']-1,width,height)
        elif e['kind']=='rect':
            style=f'rounded={1 if e["radius"] else 0};absoluteArcSize=1;arcSize={e["radius"]*2};whiteSpace=wrap;html=0;fillColor={e["fill"]};strokeColor={e["stroke"] or "none"};strokeWidth={e["sw"]};'
            cell=ET.SubElement(root,'mxCell',attrs|{'style':style,'vertex':'1'})
            geometry(cell,e['x'],e['y'],e['w'],e['h'])
        else:
            cell=ET.SubElement(root,'mxCell',attrs|{'style':f'ellipse;fillColor={e["fill"]};strokeColor=none;','vertex':'1'})
            geometry(cell,e['x']-e['radius'],e['y']-e['radius'],e['radius']*2,e['radius']*2)
    return model


def main():
    HERE.mkdir(parents=True,exist_ok=True);PDF_DIR.mkdir(parents=True,exist_ok=True)
    doc=ET.Element('mxfile',{'host':'app.diagrams.net','agent':'Editable scientific figure generator','version':'24.7.17','type':'device'})
    for lang,label in [('en','English - IEEE'),('zh','中文 - 论文')]:
        s=build(lang)
        if s.warnings: raise ValueError('\n'.join(s.warnings))
        to_pdf(s,PDF_DIR/f'csi_preprocessing_motivation_{lang}.pdf')
        to_svg(s,HERE/f'csi_preprocessing_motivation_{lang}.svg')
        diagram=ET.SubElement(doc,'diagram',{'id':f'csi-{lang}','name':label})
        diagram.append(graph_model(s))
    ET.indent(doc)
    xml=ET.tostring(doc,encoding='unicode')
    (HERE/'csi_preprocessing_motivation.drawio').write_text('<?xml version="1.0" encoding="UTF-8"?>\n'+xml,encoding='utf-8')
    # URL fragment is interpreted locally by draw.io; no image server is needed.
    compact=ET.tostring(doc,encoding='unicode')
    encoded=urllib.parse.quote(compact,safe="~()*!.'-").encode()
    compressor=zlib.compressobj(9,zlib.DEFLATED,-15)
    token=base64.b64encode(compressor.compress(encoded)+compressor.flush()).decode()
    (HERE/'drawio_editor_url.txt').write_text('https://app.diagrams.net/?splash=0&offline=1#R'+token,encoding='utf-8')
    editor_url='https://app.diagrams.net/?splash=0&offline=1#R'+token
    (HERE/'open_editor.html').write_text('<!doctype html><meta charset="utf-8"><title>CSI figure - draw.io</title><p><a href="'+html.escape(editor_url,quote=True)+'">Open the editable CSI figure in draw.io</a></p>',encoding='utf-8')
    trace={'artifact':'csi_preprocessing_motivation','figure_table_trace':[{
        'source_data':{'dataset_id':'user-provided-conceptual-description','file':'user attachment: codex-clipboard-46cd2569-5e43-41b2-8fcb-9e0268f98eec.png'},
        'transformation':{'script':str(Path(__file__).relative_to(ROOT)),'hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'caption_claim':'Conceptual motivation for controlled comparisons of CSI preprocessing and for evaluating dependence on downstream model architecture.',
        'supported_manuscript_claims':[],
        'limitations':['All waveforms are synthetic illustrations, not measurements.','Score symbols do not encode numeric values, ordering, or significance.','Panel (b) depicts possible changes to representative signal structure; it does not guarantee any operation produces a particular outcome.','No new empirical conclusions about recognition performance are established.']}],
        'pdf_width_inches':7.16,'pdf_height_inches':H*SCALE/72,
        'minimum_font_pt':16.6*SCALE,'main_font_pt':18*SCALE,
        'palette':C,'languages':['en','zh'],'editable_drawio_pages':2}
    (HERE/'figure_trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'created','shapes_per_page':len(s.items),'width_inches':7.16,'min_font_pt':16.6*SCALE,'folder':str(HERE)},ensure_ascii=False))


if __name__=='__main__':main()
