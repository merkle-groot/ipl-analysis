"""Render a .ipynb into self-contained HTML that matches the newspaper styling.

Deliberately not nbconvert's exporter: that ships its own stylesheet and, in some
templates, external script tags, both of which the artifact CSP would block.
"""
import html
import json
import re

import mistune
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from pygments.style import Style
from pygments.token import (Comment, Keyword, Name, Number, Operator, Punctuation,
                            String, Text)


class Newsprint(Style):
    background_color = "#f8f7f3"
    styles = {
        Text: "#15171b",
        Comment: "italic #8b8e94",
        Keyword: "bold #2f6f4a",
        Keyword.Constant: "bold #2f6f4a",
        Name: "#15171b",
        Name.Function: "bold #15171b",
        Name.Class: "bold #15171b",
        Name.Builtin: "#54565c",
        Name.Decorator: "#2f6f4a",
        String: "#7a5230",
        String.Doc: "italic #8b8e94",
        Number: "#54565c",
        Operator: "#54565c",
        Punctuation: "#54565c",
    }


FMT = HtmlFormatter(style=Newsprint, nowrap=True)


def style_css(scope=".nb-code"):
    """get_style_defs also emits unscoped `pre` and linenos rules, which would leak out
    and override the page's own type. Keep only the scoped token rules."""
    keep = [ln for ln in FMT.get_style_defs(scope).splitlines()
            if ln.startswith(scope)]
    return "\n".join(keep)
LEXER = PythonLexer()
MD = mistune.create_markdown(plugins=["table", "strikethrough"])


def clean_markdown(src):
    """Some cells were edited through a tool that left stray string-concatenation
    quotes at the start of continued lines. Strip them for display only."""
    lines = src.split("\n")
    if not any(ln.strip() == '"' for ln in lines):
        return src
    kept = []
    for ln in lines:
        if ln.strip() == '"':
            continue
        ln = re.sub(r'^\s*"', "", ln)
        ln = re.sub(r'"\s*$', "", ln)
        kept.append(ln)
    return "\n".join(kept)


def strip_inline_style(h):
    return re.sub(r"<style>.*?</style>", "", h, flags=re.S)


def render_output(o):
    t = o.get("output_type")
    if t == "stream":
        txt = "".join(o.get("text", []))
        return f'<pre class="nb-out">{html.escape(txt.rstrip())}</pre>'
    if t == "error":
        txt = "\n".join(o.get("traceback", []))
        txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
        return f'<pre class="nb-out nb-err">{html.escape(txt.rstrip())}</pre>'
    data = o.get("data", {})
    if "image/png" in data:
        b64 = data["image/png"]
        b64 = "".join(b64) if isinstance(b64, list) else b64
        return f'<div class="nb-fig"><img alt="notebook figure" src="data:image/png;base64,{b64.strip()}"></div>'
    if "text/html" in data:
        h = "".join(data["text/html"])
        return f'<div class="nb-table">{strip_inline_style(h)}</div>'
    if "text/plain" in data:
        txt = "".join(data["text/plain"])
        return f'<pre class="nb-out">{html.escape(txt.rstrip())}</pre>'
    return ""


def render(path, title, standfirst):
    nb = json.load(open(path))
    parts = [f'<div class="nb"><header class="nb-head"><p class="kick">{html.escape(title)}</p>'
             f'<p>{html.escape(standfirst)}</p></header>']
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            if not src.strip():
                continue
            parts.append(f'<div class="nb-md">{MD(clean_markdown(src))}</div>')
        elif c["cell_type"] == "code":
            if not src.strip():
                continue
            code = highlight(src, LEXER, FMT)
            n = c.get("execution_count")
            gutter = f"[{n}]" if n else "[ ]"
            parts.append(
                f'<div class="nb-cell"><div class="nb-gutter">{gutter}</div>'
                f'<pre class="nb-code"><code>{code.rstrip()}</code></pre></div>'
            )
            outs = "".join(render_output(o) for o in c.get("outputs", []))
            if outs.strip():
                parts.append(f'<div class="nb-outs">{outs}</div>')
    parts.append("</div>")
    return "\n".join(parts)


if __name__ == "__main__":
    import sys
    print(len(render(sys.argv[1], "x", "y")))
