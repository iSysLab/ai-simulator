"""50_draft_v4.md → 2단 조판 미리보기 PDF (김홍근 논문.pdf 양식 근사).

pandoc으로 HTML 변환 후 Chrome 헤드리스로 A4 PDF 렌더링.
참조 논문(HWP 출력) 실측 스펙:
  - 여백: 좌우 ≈10mm, 위 ≈33mm, 아래 ≈28mm / 2단 간격 ≈4mm
  - 국문 제목 16pt 휴먼명조 / 본문 10pt 휴먼명조, 줄간격 160%
  - 요약 9pt / 절 제목 10pt 함초롬돋움 굵게 / 영문 Times·Palatino
Mac에 휴먼명조·함초롬돋움이 없어 AppleMyungjo·Apple SD Gothic Neo로 대체.

사용: python paper/build_pdf.py
"""
import os
import re
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
PANDOC = "/Users/ijunsu/anaconda3/bin/pandoc"
CHROME = "/Applications/Google Chrome 2.app/Contents/MacOS/Google Chrome"

SRC = "50_draft_v4.md"
OUT_HTML = "50_draft_v4_preview.html"
OUT_PDF = "50_draft_v4.pdf"

CSS = """
@page { size: A4; margin: 33mm 10mm 28mm; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", "AppleMyungjo", serif;
       font-size: 10pt; line-height: 1.6; color: #000; margin: 0;
       word-break: keep-all; }
/* ---- 1단: 제목·저자·요약 ---- */
.head { text-align: center; }
.title-ko { font-family: "Times New Roman", "AppleMyungjo", serif; font-weight: bold;
            font-size: 16pt; line-height: 1.7; margin: 0 0 5mm; }
.authors-ko { font-size: 10pt; margin: 0 0 1mm; }
.affil-ko { font-size: 10pt; margin: 0 0 1mm; }
.email { font-size: 9pt; margin: 0 0 6mm; }
.title-en { font-family: "Times New Roman", serif; font-weight: bold;
            font-size: 14pt; line-height: 1.7; margin: 0 0 4mm; }
.authors-en { font-size: 10pt; margin: 0 0 0.5mm; }
.affil-en { font-size: 9.5pt; margin: 0 0 6mm; }
.abs-head { font-family: "Apple SD Gothic Neo", sans-serif; font-weight: 700;
            font-size: 10pt; text-align: center; margin: 0 0 2mm; }
.abstract { font-size: 9pt; line-height: 1.5; text-align: justify;
            margin: 0 7mm 7mm; }
.abstract p { margin: 0; text-indent: 2em; }
/* ---- 2단 본문 ---- */
.twocol { column-count: 2; column-gap: 4mm; text-align: justify; }
.twocol h2 { font-family: "Apple SD Gothic Neo", sans-serif; font-weight: 700;
             font-size: 10pt; margin: 4mm 0 2mm; break-after: avoid; }
.twocol h3 { font-family: "Times New Roman", "AppleMyungjo", serif; font-weight: bold;
             font-size: 10pt; margin: 3mm 0 1.5mm; break-after: avoid; }
.twocol p { margin: 0 0 2mm; }
.twocol ul { margin: 1mm 0 2mm; padding-left: 5mm; }
.twocol li { margin-bottom: 1mm; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.35;
        margin: 1.5mm 0 3mm; font-family: "Times New Roman", "Apple SD Gothic Neo", sans-serif;
        break-inside: avoid; }
/* [표 N] 캡션 문단이 표와 다른 단으로 갈라지지 않도록 */
p:has(+ table) { break-after: avoid; break-inside: avoid; font-size: 9pt; }
th, td { border-top: 0.8pt solid #000; border-bottom: 0.8pt solid #000;
         padding: 0.8mm 1.4mm; }
thead th { border-bottom: 0.4pt solid #000; }
tbody td { border-top: none; border-bottom: none; }
tbody tr:last-child td { border-bottom: 0.8pt solid #000; }
img { width: 100%; break-inside: avoid; margin: 1mm 0; }
figure { margin: 1mm 0; break-inside: avoid; }
figcaption { display: none; }  /* [그림 N] 캡션을 본문에 직접 쓰므로 alt 중복 숨김 */
p:has(> img) + p { font-size: 9pt; }  /* [그림 N] 캡션 */
code { font-family: Menlo, monospace; font-size: 8.5pt; }
strong { font-weight: bold; }
/* 참고문헌: 관례상 본문보다 작게 */
h2:last-of-type ~ p { font-size: 9pt; line-height: 1.4; margin-bottom: 1mm; }
"""


def pandoc(text):
    p = subprocess.run([PANDOC, "-f", "markdown", "-t", "html"],
                       input=text, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[:500])
    return p.stdout


def field(md, label):
    m = re.search(rf"^\*\*{label}\*\*:\s*(.+)$", md, re.M)
    return m.group(1).strip() if m else ""


def build_head(head_md):
    """제목·저자·요약 블록을 참조 양식(라벨 없는 센터 정렬)으로 조립."""
    m = re.search(r"^# (.+)$", head_md, re.M)
    title_ko = re.sub(r"^\(국문 제목\)\s*", "", m.group(1)) if m else ""
    # 둘째 줄에 '모델' 한 단어만 남지 않도록 균형 지점에서 수동 개행
    title_ko = title_ko.replace("아우르는 ", "아우르는<br>")
    m = re.search(r"## 요 약\s*\n+(.+?)(?:\n---|\Z)", head_md, re.S)
    abstract = m.group(1).strip() if m else ""
    return (
        f'<div class="head">'
        f'<p class="title-ko">{title_ko}</p>'
        f'<p class="authors-ko">{field(head_md, "저자")}</p>'
        f'<p class="affil-ko">{field(head_md, "소속")}</p>'
        f'<p class="email">{field(head_md, "이메일")}</p>'
        f'<p class="title-en">{field(head_md, "Title")}</p>'
        f'<p class="authors-en">{field(head_md, "Authors")}</p>'
        f'<p class="affil-en">{field(head_md, "Affiliation")}</p>'
        f'<p class="abs-head">요&nbsp;&nbsp;약</p>'
        f'</div>'
        f'<div class="abstract">{pandoc(abstract)}</div>'
    )


def main():
    md = open(os.path.join(BASE, SRC)).read()
    head_md, body_md = md.split("## 1. 서 론", 1)
    body_md = "## 1. 서 론" + body_md

    html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f"<title>KIISE draft v4</title><style>{CSS}</style></head><body>"
            f"{build_head(head_md)}"
            f'<div class="twocol">{pandoc(body_md)}</div></body></html>')
    html_path = os.path.join(BASE, OUT_HTML)
    open(html_path, "w").write(html)

    pdf_path = os.path.join(BASE, OUT_PDF)
    subprocess.run([CHROME, "--headless", "--disable-gpu",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", f"file://{html_path}"],
                   check=True, capture_output=True)
    print(f"저장: {pdf_path}")


if __name__ == "__main__":
    main()
