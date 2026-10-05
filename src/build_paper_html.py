"""Render reports/paper.md to a self-contained, print-ready HTML paper
(academic styling; figures referenced in the markdown embedded inline as base64).
Open in a browser and Print -> Save as PDF. Run from the project dir.
"""
import base64, re
from pathlib import Path
import markdown

MD = Path("reports/paper.md")
OUT = Path("reports/paper.html")
FIGDIR = Path("reports/figures")


def b64(p): return base64.b64encode(Path(p).read_bytes()).decode()


def inline_figures(html: str) -> str:
    def repl(m):
        src, alt = m.group("src"), m.group("alt")
        p = FIGDIR / Path(src).name
        if not p.exists():
            return m.group(0)
        return (f'<figure><img src="data:image/png;base64,{b64(p)}" alt="{alt}"/>'
                f'<figcaption>{alt}</figcaption></figure>')
    pat = re.compile(r'<img\s+(?=[^>]*alt="(?P<alt>[^"]*)")(?=[^>]*src="(?P<src>[^"]*)")[^>]*/?>')
    return pat.sub(repl, html)


CSS = """
:root { --ink:#1a1a1a; --muted:#555; --rule:#d0d0d0; --accent:#1a4d8b; }
* { box-sizing: border-box; }
body { max-width: 820px; margin: 40px auto; padding: 0 24px;
       font-family: Georgia, 'Times New Roman', serif; color: var(--ink); line-height: 1.5; font-size: 16px; }
h1 { font-size: 1.7em; line-height: 1.25; text-align: center; margin: 0 0 .2em; }
h2 { font-size: 1.2em; border-bottom: 1px solid var(--rule); padding-bottom: 3px; margin-top: 1.8em; }
h3 { font-size: 1.05em; margin-top: 1.3em; }
p, li { text-align: justify; }
strong { color: #000; }
code { font-family: 'SF Mono', Menlo, monospace; font-size: .85em; background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fa; padding: 12px 14px; border-radius: 6px; overflow-x: auto; font-size: .82em; border: 1px solid #eaeaea; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .9em; }
th, td { border: 1px solid var(--rule); padding: 6px 10px; text-align: center; }
th { background: #f2f2f2; } td:first-child, th:first-child { text-align: left; }
figure { margin: 1.5em 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #eaeaea; border-radius: 4px; }
figcaption { font-size: .85em; color: var(--muted); margin-top: 6px; text-align: left; }
a { color: var(--accent); text-decoration: none; }
@media print { body { margin: 0; font-size: 11pt; max-width: 100%; }
               h2 { page-break-after: avoid; } figure, table, pre { page-break-inside: avoid; } }
"""


def main():
    body = markdown.markdown(MD.read_text(), extensions=["tables", "toc", "fenced_code"])
    body = inline_figures(body)
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>{MD.read_text().splitlines()[0].lstrip('# ').strip()}</title>"
           f"<style>{CSS}</style></head><body>{body}</body></html>")
    OUT.write_text(doc)
    print(f"wrote {OUT} ({len(doc)//1024} KB, {doc.count('data:image/png;base64')} figures inline)")


if __name__ == "__main__":
    main()
