"""30_draft_v3.md → 2단 조판 미리보기 PDF.

pandoc으로 HTML 변환 후 Chrome 헤드리스로 A4 PDF 렌더링.
제목·요약은 1단 전체 폭, 본문은 2단 (KIISE 학술발표회 양식 근사).

사용: python paper/build_pdf.py
"""
import os
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
PANDOC = "/Users/ijunsu/anaconda3/bin/pandoc"
CHROME = "/Applications/Google Chrome 2.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 14mm 13mm; }
* { box-sizing: border-box; }
body { font-family: "AppleMyungjo", "Apple SD Gothic Neo", serif;
       font-size: 9pt; line-height: 1.4; color: #111; margin: 0; }
.head { text-align: center; margin-bottom: 4mm; }
.head h1 { font-family: "Apple SD Gothic Neo"; font-weight: 800; font-size: 15pt; margin: 0 0 3mm; }
.head p { margin: 1mm 0; font-size: 9.5pt; }
.head h2 { font-family: "Apple SD Gothic Neo"; font-size: 11pt; margin: 4mm 0 1mm; }
.abstract { text-align: justify; margin: 2mm 8mm 4mm; }
.abstract h2 { text-align: center; font-size: 10.5pt; font-family: "Apple SD Gothic Neo"; margin: 2mm 0; }
hr { border: none; border-top: 0.6pt solid #333; margin: 3mm 0; }
.twocol { column-count: 2; column-gap: 7mm; text-align: justify; }
.twocol h2 { font-family: "Apple SD Gothic Neo"; font-size: 11pt; font-weight: 800;
             margin: 3mm 0 1.2mm; break-after: avoid; }
.twocol h3 { font-family: "Apple SD Gothic Neo"; font-size: 10pt; font-weight: 700;
             margin: 2.5mm 0 1mm; break-after: avoid; }
.twocol p { margin: 0 0 1.8mm; }
.twocol ul { margin: 1mm 0 2mm; padding-left: 5mm; }
.twocol li { margin-bottom: 1mm; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 1.5mm 0 2.5mm;
        font-family: "Apple SD Gothic Neo"; break-inside: avoid; }
/* [표 N] 캡션 문단이 표와 다른 단으로 갈라지지 않도록 */
p:has(+ table) { break-after: avoid; break-inside: avoid; }
th, td { border-top: 0.7pt solid #333; border-bottom: 0.7pt solid #333;
         padding: 0.8mm 1.4mm; }
thead th { border-bottom: 0.4pt solid #333; background: #f4f4f4; }
tbody td { border-top: none; border-bottom: none; }
tbody tr:last-child td { border-bottom: 0.7pt solid #333; }
img { width: 100%; break-inside: avoid; margin: 1mm 0; }
figure { margin: 1mm 0; break-inside: avoid; }
figcaption { display: none; }  /* [그림 N] 캡션을 본문에 직접 쓰므로 alt 중복 숨김 */
code { font-family: Menlo, monospace; font-size: 8.3pt; }
strong { font-family: "Apple SD Gothic Neo"; }
/* 참고문헌은 관례상 본문보다 작게 */
h2:last-of-type ~ p { font-size: 7.5pt; line-height: 1.28; margin-bottom: 1mm; }
"""


def pandoc(text):
    p = subprocess.run([PANDOC, "-f", "markdown", "-t", "html"],
                       input=text, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[:500])
    return p.stdout


def main():
    md = open(os.path.join(BASE, "30_draft_v3.md")).read()
    head_md, body_md = md.split("## 1. 서 론", 1)
    body_md = "## 1. 서 론" + body_md

    html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f"<title>KIISE draft v3</title><style>{CSS}</style></head><body>"
            f'<div class="head abstract">{pandoc(head_md)}</div>'
            f'<div class="twocol">{pandoc(body_md)}</div></body></html>')
    html_path = os.path.join(BASE, "30_draft_v3_preview.html")
    open(html_path, "w").write(html)

    pdf_path = os.path.join(BASE, "30_draft_v3.pdf")
    subprocess.run([CHROME, "--headless", "--disable-gpu",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", f"file://{html_path}"],
                   check=True, capture_output=True)
    print(f"저장: {pdf_path}")


if __name__ == "__main__":
    main()
