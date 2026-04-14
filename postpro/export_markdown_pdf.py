from __future__ import annotations

import argparse
import base64
import mimetypes
import subprocess
from pathlib import Path

import markdown


REPO = Path(__file__).resolve().parent.parent


CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
}
body {
  font-family: Arial, Helvetica, sans-serif;
  color: #111;
  line-height: 1.45;
  font-size: 11pt;
}
h1, h2, h3 {
  color: #0f2744;
  page-break-after: avoid;
}
h1 {
  font-size: 24pt;
  border-bottom: 2px solid #d9e2ef;
  padding-bottom: 8px;
}
h2 {
  margin-top: 26px;
  font-size: 17pt;
}
h3 {
  margin-top: 18px;
  font-size: 13pt;
}
p, li {
  orphans: 3;
  widows: 3;
}
code {
  background: #f3f5f7;
  padding: 1px 4px;
  border-radius: 4px;
  font-size: 0.95em;
}
pre code {
  display: block;
  padding: 10px 12px;
  overflow-x: auto;
}
blockquote {
  border-left: 4px solid #8aa7c7;
  margin: 12px 0;
  padding: 6px 14px;
  color: #22384f;
  background: #f8fbff;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 14px 0;
  font-size: 10pt;
}
th, td {
  border: 1px solid #ccd6e0;
  padding: 8px;
  text-align: left;
}
th {
  background: #edf3f9;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 14px auto;
  break-inside: avoid;
}
hr {
  border: none;
  border-top: 1px solid #d9e2ef;
  margin: 24px 0;
}
"""


def _embed_local_images(html: str, base_dir: Path) -> str:
    import re

    def replace_src(match: re.Match[str]) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        img_path = (base_dir / src).resolve()
        if not img_path.exists():
            return match.group(0)
        mime, _ = mimetypes.guess_type(str(img_path))
        if not mime:
            mime = "application/octet-stream"
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return match.group(0).replace(src, f"data:{mime};base64,{data}")

    return re.sub(r'src="([^"]+)"', replace_src, html)


def build_html(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )
    html_body = _embed_local_images(html_body, md_path.parent)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{md_path.stem}</title>
  <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>
"""


def export_pdf(md_path: Path, pdf_path: Path, chrome_path: Path):
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(build_html(md_path), encoding="utf-8")
    subprocess.run(
        [
            str(chrome_path),
            "--headless=new",
            "--disable-gpu",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
    )
    return html_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--chrome",
        type=str,
        default=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    args = parser.parse_args()

    md_path = Path(args.input)
    if not md_path.is_absolute():
        md_path = (REPO / md_path).resolve()

    pdf_path = Path(args.output) if args.output else md_path.with_suffix(".pdf")
    if not pdf_path.is_absolute():
        pdf_path = (REPO / pdf_path).resolve()

    chrome_path = Path(args.chrome)
    export_pdf(md_path, pdf_path, chrome_path)
    print(f"Exported PDF to {pdf_path}")


if __name__ == "__main__":
    main()
