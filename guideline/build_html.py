"""GUIDELINE_v1.md -> GUIDELINE_v1.html (một file, ảnh nhúng base64, cây quyết định vẽ bằng mermaid)."""
import base64, re
from pathlib import Path
import markdown

ROOT = Path(__file__).parent
md = (ROOT / "GUIDELINE_v1.md").read_text(encoding="utf-8")
html = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])

# mermaid: <pre><code class="language-mermaid">...</code></pre> -> <pre class="mermaid">...</pre>
html = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
              lambda m: '<pre class="mermaid">' + m.group(1) + '</pre>', html, flags=re.S)
# checkbox
html = html.replace("[ ] ", '<input type="checkbox"> ')


def embed(m):
    src = m.group(1)
    p = ROOT / src
    if not p.exists():
        return m.group(0)
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'src="data:image/jpeg;base64,{b64}"'


html = re.sub(r'src="(img/[^"]+)"', embed, html)

page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guideline GreenSM v1</title>
<style>
:root{{--bg:#f6f8f7;--fg:#17201f;--muted:#55605e;--card:#ffffff;--line:#d8e0de;--accent:#0b7a6e;--warn:#b45309;--code:#eef3f2}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0f1514;--fg:#e6eceb;--muted:#9fb0ad;--card:#18211f;--line:#2b3735;--accent:#4fd1c0;--warn:#f0a646;--code:#1f2a28}}}}
:root[data-theme="dark"]{{--bg:#0f1514;--fg:#e6eceb;--muted:#9fb0ad;--card:#18211f;--line:#2b3735;--accent:#4fd1c0;--warn:#f0a646;--code:#1f2a28}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 80px}}
h1{{font-size:1.8rem;line-height:1.25;color:var(--accent)}} h2{{margin-top:2.4rem;padding-top:.8rem;border-top:2px solid var(--line)}}
h3{{margin-top:1.6rem}}
img{{max-width:100%;height:auto;border-radius:10px;border:1px solid var(--line);display:block;margin:12px 0;background:#fff}}
table{{border-collapse:collapse;width:100%;display:block;overflow-x:auto;margin:12px 0;background:var(--card)}}
th,td{{border:1px solid var(--line);padding:8px 10px;vertical-align:top;text-align:left}}
th{{background:var(--code)}}
code{{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.92em}}
pre{{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto}}
pre.mermaid{{background:var(--card);border:1px solid var(--line);text-align:center}}
blockquote{{margin:12px 0;padding:10px 14px;border-left:4px solid var(--warn);background:var(--card);border-radius:0 8px 8px 0}}
a{{color:var(--accent)}} li{{margin:3px 0}} input[type=checkbox]{{transform:scale(1.2);margin-right:6px}}
</style></head><body><main>
{html}
</main>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
mermaid.initialize({{startOnLoad:true, theme: dark ? "dark" : "default", flowchart:{{htmlLabels:true}}}});
</script>
</body></html>"""
out = ROOT / "GUIDELINE_v1.html"
out.write_text(page, encoding="utf-8")
print("->", out, round(out.stat().st_size / 1e6, 2), "MB")
