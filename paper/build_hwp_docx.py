"""50_draft_v4.md → 한글(HWP) 이관용 .docx 생성.

.hwp 5.x는 독점 바이너리라 macOS에서 직접 생성할 수 없으므로,
한글이 무손실로 여는 .docx를 HWP 양식(휴먼명조·함초롬돋움·2단)으로 만든다.
한글에서 열어 '다른 이름으로 저장 → .hwp' 하면 이관 완료.

파이프라인: pandoc(md→docx) 후 OOXML 직접 패치
  - 제목·저자 블록(서론 앞)은 1단 + 가운데 정렬, 본문은 2단(간격 4mm)
  - A4, 여백 위 33/아래 28/좌우 10mm (김홍근 양식 실측값)
  - 본문 휴먼명조 10pt + Times New Roman, 절 제목 함초롬돋움 굵게

사용: python paper/build_hwp_docx.py
"""
import os
import re
import shutil
import subprocess
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
PANDOC = "/Users/ijunsu/anaconda3/bin/pandoc"
SRC = os.path.join(BASE, "50_draft_v4.md")
OUT = os.path.join(BASE, "50_draft_v4_hwp.docx")
TMP = os.path.join(BASE, "_docx_tmp")

MM = 56.693  # 1mm → twip

PAGE = (f'<w:pgSz w:w="{int(210*MM)}" w:h="{int(297*MM)}"/>'
        f'<w:pgMar w:top="{int(33*MM)}" w:right="{int(10*MM)}" '
        f'w:bottom="{int(28*MM)}" w:left="{int(10*MM)}" '
        f'w:header="720" w:footer="720" w:gutter="0"/>')

SECT_2COL = (f'<w:sectPr><w:type w:val="continuous"/>{PAGE}'
             f'<w:cols w:num="2" w:space="{int(4*MM)}" w:equalWidth="1"/></w:sectPr>')

SECT_1COL_PARA = (f'<w:p><w:pPr><w:sectPr><w:type w:val="continuous"/>{PAGE}'
                  f'<w:cols w:num="1"/></w:sectPr></w:pPr></w:p>')


def run_pandoc():
    subprocess.run([PANDOC, os.path.basename(SRC), "-f", "markdown", "-t", "docx",
                    "-o", OUT, "--resource-path", BASE],
                   cwd=BASE, check=True)


def patch_fonts(styles):
    """기본 라틴은 Times New Roman, 한글은 휴먼명조. 제목류는 함초롬돋움."""
    body_fonts = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"'
                  ' w:eastAsia="휴먼명조" w:cs="Times New Roman"/>')
    head_fonts = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"'
                  ' w:eastAsia="함초롬돋움" w:cs="Times New Roman"/>')
    # 문서 기본 폰트
    styles = re.sub(r"<w:rFonts[^/]*/>", body_fonts, styles, count=1)
    # 스타일별 크기·폰트: (스타일ID, 폰트, 크기 half-pt, 굵게)
    rules = [("Title", head_fonts, 32, True),          # 국문 제목 16pt
             ("Heading1", head_fonts, 20, True),       # 절 제목 10pt 돋움 굵게
             ("Heading2", body_fonts, 20, True),       # 소절 10pt 명조 굵게
             ("Heading3", body_fonts, 20, True),
             ("Normal", body_fonts, 20, False)]        # 본문 10pt
    for sid, fonts, sz, bold in rules:
        m = re.search(rf'(<w:style [^>]*w:styleId="{sid}"[^>]*>.*?</w:style>)',
                      styles, re.S)
        if not m:
            continue
        block = m.group(1)
        rpr = (f"<w:rPr>{fonts}<w:color w:val=\"000000\"/>"
               f"<w:sz w:val=\"{sz}\"/><w:szCs w:val=\"{sz}\"/>")
        rpr += "<w:b/>" if bold else ""
        rpr += "</w:rPr>"
        if "<w:rPr>" in block:
            new_block = re.sub(r"<w:rPr>.*?</w:rPr>", rpr, block, count=1, flags=re.S)
        else:
            new_block = block.replace("</w:style>", rpr + "</w:style>")
        styles = styles.replace(block, new_block)
    return styles


def patch_document(doc):
    # 0) 표 폭을 단 폭 100%로 (pandoc 2.12는 pct 0.0 + 빈 tblGrid를 생성)
    doc = re.sub(r'<w:tblW w:type="pct" w:w="0(?:\.0)?"\s*/>',
                 '<w:tblW w:type="pct" w:w="5000"/>', doc)
    doc = re.sub(r'<w:tblW w:w="\d+" w:type="dxa"\s*/>',
                 '<w:tblW w:w="5000" w:type="pct"/>', doc)
    # 1) 본문 끝 sectPr(pandoc은 자기닫힘 태그) → 2단 연속 구역
    doc, n = re.subn(r"<w:sectPr\s*/>|<w:sectPr></w:sectPr>", SECT_2COL, doc)
    assert n == 1, f"최종 sectPr 치환 실패 (n={n})"
    # 2) '1. 서 론' 헤딩 앞에 1단 구역 종료 삽입 + 머리 블록 가운데 정렬
    m = re.search(r"<w:p [^>]*>(?:(?!</w:p>).)*?1\. 서\s*론.*?</w:p>", doc, re.S)
    if not m:
        m = re.search(r"<w:p>(?:(?!</w:p>).)*?1\. 서\s*론.*?</w:p>", doc, re.S)
    intro_start = m.start()
    head, body = doc[:intro_start], doc[intro_start:]
    # 머리 블록의 각 문단 가운데 정렬
    def center(pm):
        p = pm.group(0)
        if "<w:pPr>" in p:
            return p.replace("<w:pPr>", '<w:pPr><w:jc w:val="center"/>', 1)
        return p.replace(">", '><w:pPr><w:jc w:val="center"/></w:pPr>', 1)
    body_tag = head.index("<w:body>") + len("<w:body>")
    head = head[:body_tag] + re.sub(r"<w:p\b.*?</w:p>", center,
                                    head[body_tag:], flags=re.S)
    return head + SECT_1COL_PARA + body


def main():
    run_pandoc()
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    with zipfile.ZipFile(OUT) as z:
        z.extractall(TMP)

    doc_path = os.path.join(TMP, "word", "document.xml")
    sty_path = os.path.join(TMP, "word", "styles.xml")
    doc = open(doc_path, encoding="utf-8").read()
    sty = open(sty_path, encoding="utf-8").read()
    open(doc_path, "w", encoding="utf-8").write(patch_document(doc))
    open(sty_path, "w", encoding="utf-8").write(patch_fonts(sty))

    os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(TMP):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, TMP))
    shutil.rmtree(TMP)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
