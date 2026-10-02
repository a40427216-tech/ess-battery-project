"""Render editable Korean Markdown to a PDF with embedded Korean fonts."""
from pathlib import Path
import re
import html
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = set()

def inline(text):
    text = ''.join(html.escape(c) if ord(c) in SUPPORTED or c.isspace() else
                   '<font name="SymbolFallback">'+html.escape(c)+'</font>' for c in text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#2463A6">\1</link>', text)
    return text

def render(source=None):
    source = Path(source or ROOT/'docs/DAY2_분석보고서.md')
    pdfmetrics.registerFont(TTFont('Nanum', str(ROOT/'references/NanumGothic-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('NanumBold', str(ROOT/'references/NanumGothic-Bold.ttf')))
    pdfmetrics.registerFontFamily('Nanum',normal='Nanum',bold='NanumBold',italic='Nanum',boldItalic='NanumBold')
    from fontTools.ttLib import TTFont as FontToolsFont
    import matplotlib
    global SUPPORTED
    SUPPORTED = set(FontToolsFont(str(ROOT/'references/NanumGothic-Regular.ttf')).getBestCmap())
    fallback=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    pdfmetrics.registerFont(TTFont('SymbolFallback',str(fallback)))
    navy = colors.HexColor('#15324F')
    styles = {
        'h1': ParagraphStyle('h1',fontName='NanumBold',fontSize=24,leading=34,textColor=navy,spaceAfter=18,wordWrap='CJK'),
        'h2': ParagraphStyle('h2',fontName='NanumBold',fontSize=16,leading=23,textColor=navy,spaceAfter=12,wordWrap='CJK'),
        'h3': ParagraphStyle('h3',fontName='NanumBold',fontSize=11.5,leading=17,textColor=navy,spaceBefore=9,spaceAfter=6,wordWrap='CJK'),
        'body': ParagraphStyle('body',fontName='Nanum',fontSize=10,leading=16,spaceAfter=8,wordWrap='CJK'),
        'small': ParagraphStyle('small',fontName='Nanum',fontSize=8.1,leading=12,spaceAfter=5,wordWrap='CJK'),
        'table': ParagraphStyle('table',fontName='Nanum',fontSize=8.2,leading=12,wordWrap='CJK'),
    }
    doc = SimpleDocTemplate(str(source.with_suffix('.pdf')),pagesize=A4,rightMargin=40,leftMargin=40,
        topMargin=48,bottomMargin=43,title='ESS 배터리 수명 예측: DAY 2 모델 개발 및 평가',author='울산캠퍼스 1반 김진형')
    available = A4[0]-80
    story=[]
    markdown = source.read_text(encoding='utf-8').splitlines()
    i=0
    while i<len(markdown):
        line=markdown[i].strip()
        if not line:
            i+=1;continue
        if line=='<!-- PAGEBREAK -->':
            story.append(PageBreak());i+=1;continue
        if line.startswith('!['):
            m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            if m:
                path=(source.parent/m.group(2)).resolve()
                img=Image(str(path)); ratio=img.imageHeight/img.imageWidth
                width=available; height=width*ratio
                if height>300:
                    height=300;width=height/ratio
                if path.name=='08_correlations.png' and height>200:
                    height=200;width=height/ratio
                img.drawWidth=width;img.drawHeight=height;img.hAlign='CENTER'
                story.extend([img,Spacer(1,7)])
            i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(markdown) and markdown[i].strip().startswith('|'):
                cells=[x.strip() for x in markdown[i].strip().strip('|').split('|')]
                if not all(re.match(r'^:?-+:?$',x) for x in cells):rows.append(cells)
                i+=1
            count=len(rows[0]); width_weights=[1]*count
            if count==2:width_weights=[1.15,3]
            if count==3:width_weights=[1.3,1.4,3]
            if count==4:width_weights=[1.2,1.2,1.2,2.4]
            if count==4 and rows[0][0].startswith('Pearson r:'):
                width_weights=[2.6,1,1,1]
            if count>=7:width_weights=[1.05]+[1]*(count-1)
            widths=[available*w/sum(width_weights) for w in width_weights]
            tab=Table([[Paragraph(inline(x),styles['table']) for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
            tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EFF5')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                ('LINEBELOW',(0,0),(-1,0),.6,navy),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D4DCE5')),
                ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
            story.extend([tab,Spacer(1,10)]);continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));text=line[n:].strip()
            story.append(Paragraph(inline(text),styles['h1' if n==1 else 'h2' if n==2 else 'h3']))
        elif line.startswith('> '):
            story.append(Paragraph(inline(line[2:]),styles['small']))
        elif line.startswith('- '):
            story.append(Paragraph('• '+inline(line[2:]),styles['body']))
        else:
            # Keep a Markdown paragraph together, including intentional hard wraps.
            paragraph=line
            while i+1<len(markdown) and markdown[i+1].strip() and not markdown[i+1].strip().startswith(('#','|','![','<!--','- ','> ')):
                i+=1;paragraph+=' '+markdown[i].strip()
            story.append(Paragraph(inline(paragraph),styles['body']))
        i+=1
    def footer(canvas, doc):
        canvas.saveState();canvas.setFont('Nanum',8);canvas.setFillColor(colors.HexColor('#526272'))
        canvas.drawString(40,A4[1]-29,'ESS 배터리 수명 예측 · DAY 2 모델 개발 및 평가')
        canvas.setStrokeColor(colors.HexColor('#CBD6E1'));canvas.line(40,A4[1]-35,A4[0]-40,A4[1]-35)
        canvas.drawString(40,24,'울산캠퍼스 1반 김진형 · 2026.10.02')
        canvas.drawRightString(A4[0]-40,24,str(doc.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('PDF:',source.with_suffix('.pdf'))

if __name__=='__main__':
    import sys
    render(sys.argv[1] if len(sys.argv)>1 else None)
