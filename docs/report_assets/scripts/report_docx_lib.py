"""Helpers for editing the NeuroImaging project report (.docx) with python-docx + lxml.

Everything builds XML that mirrors the report's existing formatting (Times New Roman runs,
BodyJ / Heading / Caption / ListBullet / TableText styles, manually numbered captions).
"""
import copy
import os
import re

import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Emu
from docx.text.paragraph import Paragraph
from lxml import etree

DOCS = "/Users/aman/code/faltu/mish/NeuroImaging/docs"
TARGET = os.path.join(DOCS, "NeuroImaging_BCSE497J_Project_Report.docx")
BACKUP = os.path.join(DOCS, "NeuroImaging_BCSE497J_Project_Report_v1_backup.docx")
SCRATCH = "/private/tmp/claude-501/-Users-aman-code-faltu-mish-NeuroImaging/89124790-ece5-4a67-8d1e-bb538f8062d3/scratchpad"

NS = nsdecls("w")
TNR = 'w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman"'
CRN = 'w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New" w:eastAsia="Courier New"'


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load(path=TARGET):
    return docx.Document(path)


def save(doc, path=TARGET):
    doc.save(path)


# --------------------------------------------------------------------------- runs
def mk_run(text, b=False, i=False, code=False, sz=None, sub=False, sup=False):
    rpr = f'<w:rFonts {CRN if code else TNR}/>'
    if b:
        rpr += "<w:b/>"
    if i:
        rpr += "<w:i/>"
    if sz:
        rpr += f'<w:sz w:val="{sz}"/>'
    if sub:
        rpr += '<w:vertAlign w:val="subscript"/>'
    if sup:
        rpr += '<w:vertAlign w:val="superscript"/>'
    return parse_xml(f'<w:r {NS}><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>')


_TOK = re.compile(r"(`[^`]+`|\*\*.+?\*\*|\*[^*\s][^*]*?\*|_\{[^}]*\}|\^\{[^}]*\})")


def parse_inline(s, sz=None, base_b=False, base_i=False):
    """Mini markup -> list of w:r. `code`, **bold**, *italic*, _{sub}, ^{sup}."""
    runs = []
    for part in _TOK.split(s):
        if not part:
            continue
        if part.startswith("`") and part.endswith("`") and len(part) > 2:
            runs.append(mk_run(part[1:-1], code=True, sz=sz, b=base_b, i=base_i))
        elif part.startswith("**") and part.endswith("**") and len(part) > 4:
            inner = part[2:-2]
            runs.extend(parse_inline(inner, sz=sz, base_b=True, base_i=base_i))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            runs.extend(parse_inline(part[1:-1], sz=sz, base_b=base_b, base_i=True))
        elif part.startswith("_{") and part.endswith("}"):
            runs.append(mk_run(part[2:-1], sub=True, sz=sz, b=base_b, i=base_i))
        elif part.startswith("^{") and part.endswith("}"):
            runs.append(mk_run(part[2:-1], sup=True, sz=sz, b=base_b, i=base_i))
        else:
            runs.append(mk_run(part, sz=sz, b=base_b, i=base_i))
    return runs


# ---------------------------------------------------------------------- paragraphs
def _p(ppr, runs):
    p = parse_xml(f"<w:p {NS}><w:pPr>{ppr}</w:pPr></w:p>")
    for r in runs:
        p.append(r)
    return p


def P_body(s):
    return _p('<w:pStyle w:val="BodyJ"/>', parse_inline(s))


def P_h2(text):
    return _p('<w:pStyle w:val="Heading2"/>', [mk_run(text, b=True, sz=28)])


def P_h3(text):
    return _p('<w:pStyle w:val="Heading3"/>', [mk_run(text, b=True, sz=24)])


def P_bullet(s):
    return _p('<w:pStyle w:val="ListBullet"/>', parse_inline(s))


def P_num(n, s):
    runs = [mk_run(f"{n}.  ")] + parse_inline(s)
    return _p('<w:spacing w:after="60"/><w:ind w:left="567" w:hanging="454"/><w:jc w:val="both"/>', runs)


def P_spacer():
    return _p('<w:spacing w:after="80" w:line="240" w:lineRule="auto"/>', [])


def P_table_caption(label, text):
    runs = [mk_run(label + ": ", b=True, sz=22)] + parse_inline(text, sz=22)
    return _p('<w:pStyle w:val="Caption"/><w:keepNext/><w:spacing w:before="160" w:after="80"/>', runs)


def P_figure_caption(label, text):
    runs = [mk_run(label + ": ", b=True, sz=22)] + parse_inline(text, sz=22)
    return _p('<w:pStyle w:val="Caption"/>', runs)


def P_listing_caption(text):
    return _p('<w:keepNext/><w:spacing w:before="160" w:after="60" w:line="240" w:lineRule="auto"/><w:jc w:val="left"/>',
              [mk_run(text, b=True, sz=22)])


def P_code(text):
    p = parse_xml(f'<w:p {NS}><w:pPr><w:pStyle w:val="CodeListing"/></w:pPr></w:p>')
    p.append(mk_run(text if text else " ", code=True, sz=16))
    return p


def P_picture(doc, path, width_emu):
    """Centered picture paragraph (same pPr as the report's figures)."""
    p = parse_xml(f'<w:p {NS}><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="40" w:line="240" w:lineRule="auto"/>'
                  f'<w:jc w:val="center"/></w:pPr></w:p>')
    para = Paragraph(p, doc._body)
    from PIL import Image
    w, h = Image.open(path).size
    para.add_run().add_picture(path, width=Emu(width_emu), height=Emu(int(width_emu * h / w)))
    return p


def P_equation(doc, png, label):
    """Equation paragraph: centred picture + right-aligned number, as in the report."""
    p = parse_xml(f'<w:p {NS}><w:pPr><w:keepLines/><w:tabs><w:tab w:val="center" w:pos="4153"/><w:tab w:val="right" w:pos="8306"/></w:tabs>'
                  f'<w:spacing w:before="80" w:after="160" w:line="240" w:lineRule="auto"/><w:jc w:val="left"/></w:pPr></w:p>')
    p.append(parse_xml(f"<w:r {NS}><w:tab/></w:r>"))
    from PIL import Image
    w, h = Image.open(png).size
    para = Paragraph(p, doc._body)
    para.add_run().add_picture(png, width=Emu(int(w / 300 * 914400)), height=Emu(int(h / 300 * 914400)))
    r = parse_xml(f'<w:r {NS}><w:rPr><w:rFonts {TNR}/><w:sz w:val="24"/></w:rPr><w:tab/><w:t>({label})</w:t></w:r>')
    p.append(r)
    return p


# -------------------------------------------------------------------------- tables
def mk_table(widths, header, rows, sz=20, bold_first=False, long_ok=False):
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    t = parse_xml(
        f'<w:tbl {NS}><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:type="auto" w:w="0"/><w:jc w:val="center"/>'
        f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
        f'<w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
        f'<w:tblLook w:firstColumn="1" w:firstRow="1" w:lastColumn="0" w:lastRow="0" w:noHBand="0" w:noVBand="1" w:val="04A0"/>'
        f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid></w:tbl>'
    )
    keep = (len(rows) <= 16) and not long_ok

    def cell(w, text, hdr=False, b=False, last=False):
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="D9D9D9"/>' if hdr else ""
        tc = parse_xml(f'<w:tc {NS}><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/>{shd}</w:tcPr></w:tc>')
        for line in str(text).split("\n"):
            kn = "<w:keepNext/>" if (keep and not last) or hdr else ""
            p = parse_xml(f'<w:p {NS}><w:pPr><w:pStyle w:val="TableText"/>{kn}<w:jc w:val="left"/></w:pPr></w:p>')
            for r in parse_inline(line, sz=sz, base_b=(hdr or b)):
                p.append(r)
            tc.append(p)
        return tc

    tr = parse_xml(f"<w:tr {NS}><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr></w:tr>")
    for w, h in zip(widths, header):
        tr.append(cell(w, h, hdr=True))
    t.append(tr)
    for ri, row in enumerate(rows):
        tr = parse_xml(f"<w:tr {NS}><w:trPr><w:cantSplit/></w:trPr></w:tr>")
        for ci, (w, c) in enumerate(zip(widths, row)):
            tr.append(cell(w, c, b=(bold_first and ci == 0), last=(ri == len(rows) - 1)))
        t.append(tr)
    return t


# ---------------------------------------------------------------------- navigation
def ptext(p):
    out = []
    for n in p.iter():
        if n.tag == qn("w:t"):
            out.append(n.text or "")
        elif n.tag == qn("w:tab"):
            out.append("\t")
        elif n.tag == qn("w:br") and n.get(qn("w:type")) in (None, "textWrapping"):
            out.append("\n")
    return "".join(out)


def norm(s):
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def body_ps(doc):
    return [e for e in doc.element.body.iterchildren() if e.tag == qn("w:p")]


def pstyle(p):
    ps = p.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    return ps.get(qn("w:val")) if ps is not None else None


def find(doc, start, style=None, nth=0):
    """Find a top-level paragraph whose normalised text starts with `start`."""
    hits = []
    for p in body_ps(doc):
        if style and pstyle(p) != style:
            continue
        if norm(ptext(p)).startswith(norm(start)):
            hits.append(p)
    if len(hits) <= nth:
        raise KeyError(f"paragraph not found: {start!r} (style={style}) hits={len(hits)}")
    return hits[nth]


def find_unique(doc, start, style=None):
    hits = [p for p in body_ps(doc) if (not style or pstyle(p) == style) and norm(ptext(p)).startswith(norm(start))]
    if len(hits) != 1:
        raise KeyError(f"expected exactly 1 paragraph for {start!r}, got {len(hits)}")
    return hits[0]


def next_tbl(p):
    n = p.getnext()
    while n is not None and n.tag != qn("w:tbl"):
        n = n.getnext()
    return n


def insert_after(ref, *elems):
    cur = ref
    for e in elems:
        cur.addnext(e)
        cur = e
    return cur


def insert_before(ref, *elems):
    for e in elems:
        ref.addprevious(e)


def remove(e):
    e.getparent().remove(e)


# -------------------------------------------------------------------- text editing
def rewrite_par(p, markup, keep_ppr=True):
    """Replace all runs of paragraph p with runs parsed from markup (pPr kept)."""
    for ch in list(p):
        if ch.tag != qn("w:pPr"):
            p.remove(ch)
    for r in parse_inline(markup):
        p.append(r)


def rewrite_par_sz(p, markup, sz):
    for ch in list(p):
        if ch.tag != qn("w:pPr"):
            p.remove(ch)
    for r in parse_inline(markup, sz=sz):
        p.append(r)


def replace_in_par(p, old, new, count=1):
    """Replace `old` by `new` inside a single w:t (error if not found)."""
    n = 0
    for t in p.iter(qn("w:t")):
        if t.text and old in t.text:
            t.text = t.text.replace(old, new, 1 if count == 1 else -1)
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            n += 1
            if count == 1:
                return
    if n == 0:
        raise KeyError(f"{old!r} not found in paragraph {ptext(p)[:80]!r}")


def set_cell(cell_tc, markup, sz=None, b=None):
    """Replace the text of a table cell (w:tc), keeping its first paragraph's pPr and run size/bold."""
    ps = cell_tc.findall(qn("w:p"))
    first = ps[0]
    for extra in ps[1:]:
        cell_tc.remove(extra)
    r0 = first.find(qn("w:r"))
    base_sz = sz
    base_b = b
    if r0 is not None:
        sze = r0.find(qn("w:rPr") + "/" + qn("w:sz"))
        if base_sz is None and sze is not None:
            base_sz = int(sze.get(qn("w:val")))
        if base_b is None:
            base_b = r0.find(qn("w:rPr") + "/" + qn("w:b")) is not None
    for ch in list(first):
        if ch.tag != qn("w:pPr"):
            first.remove(ch)
    for r in parse_inline(markup, sz=base_sz, base_b=bool(base_b)):
        first.append(r)


def rows_of(tbl):
    return tbl.findall(qn("w:tr"))


def cells_of(tr):
    return tr.findall(qn("w:tc"))


def row_texts(tbl):
    return [[norm("".join(t.text or "" for t in c.iter(qn("w:t")))) for c in cells_of(tr)] for tr in rows_of(tbl)]


def clone_row(tr):
    return copy.deepcopy(tr)


def set_row(tr, texts):
    for c, t in zip(cells_of(tr), texts):
        set_cell(c, t)


def add_row_after(tr, texts):
    new = clone_row(tr)
    set_row(new, texts)
    tr.addnext(new)
    return new


def set_caption(p, label, text):
    """Rewrite a caption paragraph keeping its pPr: bold 'label: ' + normal text (sz 22)."""
    for ch in list(p):
        if ch.tag != qn("w:pPr"):
            p.remove(ch)
    p.append(mk_run(label + ": ", b=True, sz=22))
    for r in parse_inline(text, sz=22):
        p.append(r)


def replace_num(doc, start, n, markup):
    p = find(doc, start)
    for ch in list(p):
        if ch.tag != qn("w:pPr"):
            p.remove(ch)
    p.append(mk_run(f"{n}.  "))
    for r in parse_inline(markup):
        p.append(r)
    return p


def replace_table(old_tbl, new_tbl):
    old_tbl.addnext(new_tbl)
    old_tbl.getparent().remove(old_tbl)
    return new_tbl


def sync_toc_and_lists(doc):
    """Add missing TOC entries (new headings) and List-of-Tables/Figures entries (new captions), and refresh
    list entries whose caption text changed. New entries get an empty page number (Word: update the TOC field,
    and type the page numbers of the manual lists)."""
    import copy as _copy
    ps = [e for e in doc.element.body.iterchildren() if e.tag == qn("w:p")]
    toc = [p for p in ps if pstyle(p) in ("TOC1", "TOC2", "TOC3")]
    tocmap = {norm(next(p.iter(qn("w:t"))).text or "").upper(): p for p in toc}
    tmpl = {}
    for p in toc:
        tmpl.setdefault(pstyle(p), p)
    last, added = None, []
    for h in [p for p in ps if pstyle(p) in ("Heading1", "Heading2", "Heading3")]:
        key = norm(ptext(h).replace("\n", " ")).upper()
        lvl = {"Heading1": "TOC1", "Heading2": "TOC2", "Heading3": "TOC3"}[pstyle(h)]
        if key in tocmap:
            last = tocmap[key]
            continue
        new = _copy.deepcopy(tmpl[lvl])
        ts = list(new.iter(qn("w:t")))
        ts[0].text = key
        for t in ts[1:]:
            t.text = ""
        last.addnext(new)
        last = new
        tocmap[key] = new
        added.append(key)
    entries = [p for p in ps if pstyle(p) == "ListEntry"]
    entmap = {norm(next(p.iter(qn("w:t"))).text or ""): p for p in entries}
    prev = {"Table": None, "Figure": None}
    changed, new_entries = [], []

    def rebuild(entry, label, cap):
        ts = [t.text or "" for t in entry.iter(qn("w:t"))]
        page = ts[-1] if len(ts) >= 3 else ""
        for ch in list(entry):
            if ch.tag != qn("w:pPr"):
                entry.remove(ch)
        entry.append(mk_run(label, sz=24))
        entry.append(parse_xml(f"<w:r {NS}><w:tab/></w:r>"))
        for r in [_copy.deepcopy(r) for r in cap.findall(qn("w:r"))][1:]:
            sz = r.find(qn("w:rPr") + "/" + qn("w:sz"))
            if sz is not None:
                sz.set(qn("w:val"), "24")
            entry.append(r)
        entry.append(parse_xml(f"<w:r {NS}><w:tab/></w:r>"))
        entry.append(mk_run(page, sz=24))

    for cap in [p for p in ps if pstyle(p) == "Caption"]:
        txt = ptext(cap)
        m = re.match(r"(Table|Figure) ([A-Z]?\d+\.\d+): ", txt)
        if not m:
            continue
        kind, label = m.group(1), f"{m.group(1)} {m.group(2)}"
        cap_text = norm(txt[m.end():])
        if label in entmap:
            e = entmap[label]
            cur = norm("".join((t.text or "") for t in list(e.iter(qn("w:t")))[1:-1]))
            if cur != cap_text:
                rebuild(e, label, cap)
                changed.append(label)
        else:
            e = _copy.deepcopy(entries[0])
            prev[kind].addnext(e)
            rebuild(e, label, cap)
            entmap[label] = e
            new_entries.append(label)
        prev[kind] = entmap[label]
    return added, changed, new_entries
