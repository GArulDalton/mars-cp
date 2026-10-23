const fs = require("fs");
const { Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, ShadingType, PageBreak } = require("docx");
const T = JSON.parse(fs.readFileSync(__dirname + "/../results/tables.json"));
T.synth = T.synth.map(r => r.map(c => (c === "nan" ? "—" : c)));
const FONT = "Times New Roman";

// inline markup: **bold**, *italic*, _{sub}, ^{sup}
function runs(text, base = {}) {
  const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (tok.startsWith("*")) out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (tok.startsWith("_{")) out.push(new TextRun({ text: tok.slice(2, -1), subScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: tok.slice(2, -1), superScript: true, font: FONT, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 300 }, ...(o.p || {}) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], spacing: { before: 280, after: 140 } });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })], spacing: { before: 200, after: 100 } });
const EQ = (t, n) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  children: [...runs(t), new TextRun({ text: n ? `\t\t(${n})` : "", font: FONT })] });
const BUL = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 290 } });
const NUM = (t) => new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 290 } });
const CAP = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 200 }, children: runs(t, { size: 19 }) });
const BREAK = () => new Paragraph({ children: [new PageBreak()] });

function FIG(file, w, h, cap, width = 600) {
  cap = cap.replace(/[.]$/, "");              // MLJ: no punctuation at end of caption
  const W = width, H = Math.round(W * h / w);
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: W, height: H },
      altText: { title: cap.replace(/\*/g, "").slice(0, 60), description: cap.replace(/\*/g, "").slice(0, 200), name: file.split("/").pop() } })] }), CAP(cap)];
}
const bd = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function TABLE(header, rows, widths, cap, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, i, isHead, shade) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: isHead ? bd : nb, bottom: isHead ? bd : nb, left: nb, right: nb },
    shading: shade ? { fill: "EDEDED", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(txt), { size: opts.size || 16, bold: isHead }) })] });
  const trs = [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, i, true)) })];
  rows.forEach(r => {
    const shade = opts.shadeFn ? opts.shadeFn(r) : false;
    trs.push(new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, shade)) }));
  });
  return [new Paragraph({ keepNext: true, spacing: { before: 160, after: 60 }, children: runs(cap, { size: 19 }) }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: trs,
      borders: { top: bd, bottom: bd, left: nb, right: nb, insideHorizontal: nb, insideVertical: nb } }),
    new Paragraph({ spacing: { after: 160 }, children: [] })];
}
function ALGO(title, lines) {
  const cellParas = [new Paragraph({ spacing: { after: 60 }, children: runs(title, { bold: true, size: 19 }) })]
    .concat(lines.map(l => new Paragraph({ spacing: { after: 0, line: 260 }, indent: { left: 240 * (l[0] || 0) },
      children: runs(l[1], { size: 18 }) })));
  return [new Table({ width: { size: 9026, type: WidthType.DXA }, columnWidths: [9026], rows: [new TableRow({ cantSplit: true, children: [
    new TableCell({ width: { size: 9026, type: WidthType.DXA }, borders: { top: bd, bottom: bd, left: nb, right: nb },
      margins: { top: 100, bottom: 100, left: 120, right: 120 }, children: cellParas })] })] }),
    new Paragraph({ spacing: { after: 160 }, children: [] })];
}
module.exports = { runs, P, H1, H2, EQ, BUL, NUM, CAP, FIG, TABLE, ALGO, BREAK, T, FONT };
