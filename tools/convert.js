/* HLD.md -> HLD.docx + HLD.html (for PDF). Run: node convert.js <src.md> <outdir> */
const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');
const D = require('docx');

const [,, SRC, OUTDIR] = process.argv;
const BASEDIR = path.dirname(path.resolve(SRC));
let md = fs.readFileSync(SRC, 'utf8');

// strip collapsible <details>…</details> (mermaid source) for clean shareable output
md = md.replace(/\r\n/g, '\n').replace(/[ \t]*<details>[\s\S]*?<\/details>[ \t]*\n?/g, '');

const mdit = new MarkdownIt({ html: true, linkify: true, typographer: false });
const tokens = mdit.parse(md, {});

/* ---------- shared: PNG size ---------- */
function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

/* ============================ DOCX ============================ */
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, LevelFormat, ShadingType, ImageRun,
  ExternalHyperlink, TableLayoutType
} = D;

const CONTENT_W = 9020; // dxa, A4 minus ~1" margins
const HEADING = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3,
                  4: HeadingLevel.HEADING_4, 5: HeadingLevel.HEADING_5, 6: HeadingLevel.HEADING_6 };

function inlineRuns(tok, opts = {}) {
  const runs = [];
  let bold = !!opts.bold, ital = !!opts.italic;
  let link = null;
  const push = (t, extra = {}) => {
    const r = new TextRun({ text: t, bold, italics: ital, font: extra.code ? 'Consolas' : undefined,
      size: extra.code ? 20 : undefined,
      shading: extra.code ? { type: ShadingType.CLEAR, fill: 'EFEFEF' } : undefined,
      break: extra.break });
    if (link) runs.push(new ExternalHyperlink({ children: [r], link }));
    else runs.push(r);
  };
  for (const c of tok.children || []) {
    if (c.type === 'text') push(c.content);
    else if (c.type === 'strong_open') bold = true;
    else if (c.type === 'strong_close') bold = false;
    else if (c.type === 'em_open') ital = true;
    else if (c.type === 'em_close') ital = false;
    else if (c.type === 'code_inline') push(c.content, { code: true });
    else if (c.type === 'softbreak') push(' ');
    else if (c.type === 'hardbreak') runs.push(new TextRun({ text: '', break: 1 }));
    else if (c.type === 'link_open') link = c.attrGet('href');
    else if (c.type === 'link_close') link = null;
    else if (c.type === 'html_inline') { /* skip */ }
  }
  return runs;
}

function imageParagraph(src) {
  const p = src.replace(/^\.\//, '');
  const abs = path.resolve(BASEDIR, p);
  const buf = fs.readFileSync(abs);
  const { w, h } = pngSize(buf);
  const maxW = 620;
  const scale = Math.min(1, maxW / w);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 160 },
    children: [new ImageRun({ type: 'png', data: buf,
      transformation: { width: Math.round(w * scale), height: Math.round(h * scale) } })]
  });
}

function codeParagraphs(text) {
  const lines = text.replace(/\n$/, '').split('\n');
  return [new Paragraph({
    spacing: { before: 80, after: 120 },
    shading: { type: ShadingType.CLEAR, fill: 'F0F0F0' },
    children: lines.flatMap((l, i) => [new TextRun({ text: l || ' ', font: 'Consolas', size: 18, break: i ? 1 : undefined })])
  })];
}

function hr() {
  return new Paragraph({ spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: 'BBBBBB' } }, children: [] });
}

// walk tokens -> docx blocks
const blocks = [];
let i = 0;
const listStack = []; // {type:'ol'|'ul'}
while (i < tokens.length) {
  const t = tokens[i];
  if (t.type === 'heading_open') {
    const lvl = +t.tag[1];
    const inl = tokens[i + 1];
    blocks.push(new Paragraph({ heading: HEADING[lvl] || HeadingLevel.HEADING_6,
      spacing: { before: lvl <= 2 ? 240 : 160, after: 100 },
      children: inlineRuns(inl) }));
    i += 3; continue;
  }
  if (t.type === 'paragraph_open') {
    const inl = tokens[i + 1];
    const onlyImg = inl.children && inl.children.length === 1 && inl.children[0].type === 'image';
    if (onlyImg) {
      blocks.push(imageParagraph(inl.children[0].attrGet('src')));
    } else if (listStack.length) {
      const top = listStack[listStack.length - 1];
      blocks.push(new Paragraph({
        numbering: { reference: top.type, level: Math.min(listStack.length - 1, 2) },
        spacing: { before: 20, after: 20 },
        children: inlineRuns(inl)
      }));
    } else {
      blocks.push(new Paragraph({ spacing: { before: 60, after: 120 }, children: inlineRuns(inl) }));
    }
    i += 3; continue;
  }
  if (t.type === 'bullet_list_open') { listStack.push({ type: 'ul' }); i++; continue; }
  if (t.type === 'ordered_list_open') { listStack.push({ type: 'ol' }); i++; continue; }
  if (t.type === 'bullet_list_close' || t.type === 'ordered_list_close') { listStack.pop(); i++; continue; }
  if (t.type === 'list_item_open' || t.type === 'list_item_close') { i++; continue; }
  if (t.type === 'fence' || t.type === 'code_block') { blocks.push(...codeParagraphs(t.content)); i++; continue; }
  if (t.type === 'hr') { blocks.push(hr()); i++; continue; }
  if (t.type === 'blockquote_open') {
    i++;
    while (tokens[i] && tokens[i].type !== 'blockquote_close') {
      if (tokens[i].type === 'paragraph_open') {
        blocks.push(new Paragraph({ indent: { left: 360 }, spacing: { before: 40, after: 80 },
          border: { left: { style: BorderStyle.SINGLE, size: 18, color: 'CCCCCC', space: 8 } },
          children: inlineRuns(tokens[i + 1], { italic: true }) }));
        i += 3;
      } else i++;
    }
    i++; continue;
  }
  if (t.type === 'table_open') {
    const rows = [];
    let header = true;
    i++;
    while (tokens[i] && tokens[i].type !== 'table_close') {
      const tk = tokens[i];
      if (tk.type === 'tr_open') {
        const cells = [];
        i++;
        while (tokens[i] && tokens[i].type !== 'tr_close') {
          if (tokens[i].type === 'th_open' || tokens[i].type === 'td_open') {
            const isH = tokens[i].type === 'th_open';
            const inl = tokens[i + 1];
            cells.push({ isH, runs: inlineRuns(inl) });
            i += 3;
          } else i++;
        }
        rows.push({ header, cells });
        if (header) header = false;
      }
      i++;
    }
    i++; // table_close
    // drop an all-empty header row (front-matter style "| | |")
    const rows2 = rows.filter(r => !(r.header && r.cells.every(c => !c.runs.length)));
    if (rows2.length && rows.length !== rows2.length) rows2[0].header = false;
    rows.length = 0; rows.push(...rows2);
    const ncol = Math.max(...rows.map(r => r.cells.length));
    const colW = Math.floor(CONTENT_W / ncol);
    const widths = Array(ncol).fill(colW);
    widths[ncol - 1] += CONTENT_W - colW * ncol;
    const trs = rows.map(r => new TableRow({
      ...(r.header ? { tableHeader: true } : {}),
      children: r.cells.map((c, ci) => new TableCell({
        width: { size: widths[ci], type: WidthType.DXA },
        shading: r.header ? { type: ShadingType.CLEAR, fill: 'E7E7F2' } : undefined,
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        children: [new Paragraph({ spacing: { before: 0, after: 0 },
          children: (c.runs.length ? c.runs : [new TextRun('')]).map(x => x) })]
      }))
    }));
    blocks.push(new Table({
      columnWidths: widths,
      width: { size: CONTENT_W, type: WidthType.DXA },
      layout: TableLayoutType.FIXED,
      borders: {
        top:{style:BorderStyle.SINGLE,size:2,color:'AAAAAA'}, left:{style:BorderStyle.SINGLE,size:2,color:'AAAAAA'},
        bottom:{style:BorderStyle.SINGLE,size:2,color:'AAAAAA'}, right:{style:BorderStyle.SINGLE,size:2,color:'AAAAAA'},
        insideHorizontal:{style:BorderStyle.SINGLE,size:2,color:'CCCCCC'}, insideVertical:{style:BorderStyle.SINGLE,size:2,color:'CCCCCC'}
      },
      rows: trs
    }));
    blocks.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
    continue;
  }
  if (t.type === 'html_block') { i++; continue; }
  i++;
}

const numLevels = (fmt, texts) => [0,1,2].map(l => ({
  level: l, format: fmt, text: texts(l), alignment: AlignmentType.START,
  style: { paragraph: { indent: { left: 460 * (l + 1) + 260, hanging: 320 } } }
}));

const doc = new Document({
  numbering: { config: [
    { reference: 'ol', levels: numLevels(LevelFormat.DECIMAL, l => `%${l + 1}.`) },
    { reference: 'ul', levels: numLevels(LevelFormat.BULLET, l => ['●','○','▪'][l]) },
  ] },
  styles: { default: {
    document: { run: { font: 'Calibri', size: 21 }, paragraph: { spacing: { line: 288 } } },
    heading1: { run: { font: 'Calibri', size: 34, bold: true, color: '1A1A1A' } },
    heading2: { run: { font: 'Calibri', size: 28, bold: true, color: '20304A' } },
    heading3: { run: { font: 'Calibri', size: 24, bold: true, color: '2A2A2A' } },
    heading4: { run: { font: 'Calibri', size: 22, bold: true, color: '444444' } },
  } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    children: blocks
  }]
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(path.join(OUTDIR, 'HLD.docx'), b);
  console.log('wrote HLD.docx', b.length, 'bytes,', blocks.length, 'blocks');
});

/* ============================ HTML (for PDF) ============================ */
mdit.renderer.rules.image = (toks, idx) => {
  const tk = toks[idx];
  const src = tk.attrGet('src').replace(/^\.\//, '');
  const abs = path.resolve(BASEDIR, src);
  const b64 = fs.readFileSync(abs).toString('base64');
  const alt = tk.content || '';
  return `<figure><img alt="${alt.replace(/"/g,'&quot;')}" src="data:image/png;base64,${b64}"><figcaption>${alt}</figcaption></figure>`;
};
const body = mdit.render(md);
const css = `
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font: 11pt/1.5 "Segoe UI", Calibri, Arial, sans-serif; color: #1c1c1c; max-width: 100%; }
h1 { font-size: 22pt; margin: 0 0 .3em; color:#141414; }
h2 { font-size: 16pt; margin: 1.4em 0 .4em; color:#20304a; border-bottom:1px solid #d5d5e0; padding-bottom:.15em; }
h3 { font-size: 13pt; margin: 1.1em 0 .3em; color:#2a2a2a; }
h4 { font-size: 11.5pt; margin: 1em 0 .3em; color:#444; }
p { margin: .5em 0; }
a { color: #1a4f8a; text-decoration: none; }
code { font-family: Consolas, "Courier New", monospace; font-size: .9em; background:#eef0f2; padding:.05em .3em; border-radius:3px; }
pre { background:#f5f5f7; border:1px solid #d8d8dd; border-radius:4px; padding:.7em .9em; overflow-x:auto; font-family:Consolas,monospace; font-size:9pt; line-height:1.45; page-break-inside:avoid; }
pre code { background:none; padding:0; }
table { border-collapse: collapse; width: 100%; margin: .8em 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #b8b8c0; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #e7e7f2; }
tr:nth-child(even) td { background: #fafafb; }
blockquote { margin: .6em 0; padding: .2em .9em; border-left: 3px solid #ccc; color: #444; background:#fafafa; }
figure { margin: 1em 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #e2e2e2; }
figcaption { font-size: 8.5pt; color: #666; margin-top: .3em; font-style: italic; }
hr { border: none; border-top: 1px solid #c8c8c8; margin: 1.2em 0; }
h1,h2,h3,h4 { page-break-after: avoid; }
ul, ol { margin: .4em 0 .6em; padding-left: 1.4em; }
li { margin: .15em 0; }
`;
fs.writeFileSync(path.join(OUTDIR, 'HLD.html'),
  `<!doctype html><html><head><meta charset="utf-8"><title>HLD — Jamming Survival: On-Device Mesh Agent</title><style>${css}</style></head><body>${body}</body></html>`);
console.log('wrote HLD.html');
