const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

const ROOT = __dirname;
const SRC = path.join(ROOT, 'src');
const DOCS = path.join(ROOT, 'docs');
const DIAGRAMS_SRC = path.join(ROOT, 'diagrams');
const DIAGRAMS_DEST = path.join(DOCS, 'diagrams');

const IFRAME_HEIGHTS = {
  'diagram_01_caa_heatmap.html': 580,
  'diagram_07_somatic_ci.html': 480,
  'diagram_08_ising_chain.html': 560,
};

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function copyRecursive(src, dest) {
  ensureDir(dest);
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

/** Collapse excessive blank lines but keep single blank lines (required to end markdown tables) */
function normalizeWhitespace(markdown) {
  return markdown.replace(/(\n[ \t\u00a0]*){3,}/g, '\n\n');
}

/** Ensure prose after a table isn't parsed as part of the table */
function terminateTables(markdown) {
  const lines = markdown.split('\n');
  const out = [];
  const isTableLine = (line) => /^\|.+\|/.test(line.trim());

  for (let i = 0; i < lines.length; i++) {
    out.push(lines[i]);
    const next = lines[i + 1];
    if (
      isTableLine(lines[i]) &&
      next !== undefined &&
      next.trim() &&
      !isTableLine(next)
    ) {
      out.push('');
    }
  }
  return out.join('\n');
}

/** Convert section titles and subheadings to markdown headings */
function preprocessHeadings(markdown) {
  const lines = markdown.split('\n');
  const out = [];

  const isMainSection = (t) =>
    /^(\d+[a-z]?\. (Phase |The Question|Limitations|What's Next))/i.test(t) ||
    /^Phase \d+:/i.test(t) ||
    /^10a\./i.test(t) ||
    /^References$/i.test(t);

  const isListItem = (t) =>
    /^\d+\.\s+\w/.test(t) &&
    (t.length > 85 || /^(\d+\.\s+\w+:\s+).{20,}/.test(t));

  let inReferences = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    const prevBlank = i === 0 || !lines[i - 1].trim();
    const nextLine = lines[i + 1]?.trim() || '';
    const nextBlank = !nextLine;
    const nextIsBody = nextLine.length > 50 && /^[A-Z"'(]/.test(nextLine);

    if (isMainSection(trimmed)) {
      out.push(`## ${trimmed}`);
      if (/^References$/i.test(trimmed)) inReferences = true;
      continue;
    }

    if (inReferences) {
      out.push(line);
      continue;
    }

    if (/^\*\*(.+)\*\*\s*$/.test(trimmed)) {
      out.push(`### ${trimmed.replace(/^\*\*|\*\*$/g, '')}`);
      continue;
    }

    if (
      prevBlank &&
      (nextBlank || nextIsBody) &&
      trimmed.length >= 8 &&
      trimmed.length <= 78 &&
      /^[A-Z{*]/.test(trimmed) &&
      !trimmed.startsWith('|') &&
      !trimmed.startsWith('!') &&
      !trimmed.startsWith('-') &&
      !trimmed.startsWith('$$') &&
      !isListItem(trimmed) &&
      !/^[a-z_]+\s*=/.test(trimmed) &&
      !/^(\d+\.\d+\s|joy ↔|Every pair|For example|Hope:|Dread:|The full corpus)/.test(trimmed) &&
      !/^"[A-Z]/.test(trimmed) &&
      trimmed.split(/\s+/).length <= 12
    ) {
      out.push(`### ${trimmed}`);
      continue;
    }

    out.push(line);
  }

  return out.join('\n');
}

/** Fix inline equations and remove diagram placeholders */
function preprocessContent(markdown) {
  let result = markdown;

  result = result.replace(
    /logitsl[\s\u00a0]*=[\s\u00a0]*rl\(pos\)[\s\u00a0]*[⋅·\u22c5][\s\u00a0]*WU[\s\u00a0]*,[\s\u00a0]*r[\s\u00a0]*∈[\s\u00a0]*R[\s\u00a0]*2304[\s\u00a0]*/g,
    () => '$$\\text{logits}_l = r_l(\\text{pos}) \\cdot W_U, \\quad r \\in \\mathbb{R}^{2304}$$\n\n'
  );

  result = result.replace(/\{\[DIAGRAM[\s\u00a0]*3[^\}]*\}[\s\u00a0]*/g, '');

  return result;
}

/** Fix tables that start with an empty header row */
function fixEmptyTableHeaders(markdown) {
  return markdown.replace(
    /\|[\s|]+\|\s*\n\|[\s\-:|]+\|\s*\n(\|[^|\n]+\|[^\n]*\n)/g,
    (_, headerRow) => {
      const colCount = (headerRow.match(/\|/g) || []).length - 1;
      const sep = '|' + Array(colCount).fill('---').join('|') + '|';
      return `${headerRow.trimEnd()}\n${sep}\n`;
    }
  );
}

/** Convert Obsidian-style ![[diagram_name]] embeds to HTML */
function preprocessDiagrams(markdown) {
  return markdown.replace(
    /!\[\[([^\]]+)\]\]/g,
    (_, rawName) => {
      const name = rawName.trim().replace(/\s+\d+(?=\.\w)/, '');
      const src = `diagrams/${name}`;

      if (name.endsWith('.html')) {
        const height = IFRAME_HEIGHTS[name] || 450;
        return `\n<div class="diagram-embed"><iframe src="${src}" width="100%" height="${height}px" frameborder="0" loading="lazy"></iframe></div>\n`;
      }

      if (/\.(svg|png|jpe?g)$/i.test(name)) {
        const alt = name.replace(/\.[^.]+$/, '').replace(/_/g, ' ');
        return `\n<img src="${src}" alt="${alt}">\n`;
      }

      return `\n<p><em>[Missing diagram: ${name}]</em></p>\n`;
    }
  );
}

/** Fix LaTeX issues that break MathJax */
function preprocessMath(markdown) {
  let result = markdown;

  // Put display math on its own lines (use concat — $$ in templates collapses to $)
  result = result.replace(/\$\$\s*([\s\S]*?)\s*\$\$/g, (_, expr) => '\n$$' + expr.trim() + '$$\n');

  // \in {4, 8} → \in \{4, 8\}  (bare braces break MathJax)
  result = result.replace(/\\in\s*\{([\d,\s\u00a0]+)\}/g, (_, nums) => '\\in \\{' + nums + '\\}');

  // Normalize unicode spaces inside math to regular spaces
  result = result.replace(/\$\$([\s\S]*?)\$\$/g, (_, inner) => '$$' + inner.replace(/[\u00a0\u2009]/g, ' ') + '$$');
  result = result.replace(/\$([^$\n]+)\$/g, (_, inner) => '$' + inner.replace(/[\u00a0\u2009]/g, ' ') + '$');

  return result;
}

/** Protect math from marked eating backslashes, restore after parse */
function protectMath(markdown) {
  const blocks = [];
  let result = markdown;

  result = result.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
    blocks.push(match);
    return `\n\n<!--MATH${blocks.length - 1}-->\n\n`;
  });

  result = result.replace(/\$([^$\n]+)\$/g, (match) => {
    blocks.push(match);
    return `<!--MATH${blocks.length - 1}-->`;
  });

  return { markdown: result, blocks };
}

/** Restore math placeholders; wrap display math for styling */
function restoreMath(html, blocks) {
  return html.replace(/<!--MATH(\d+)-->/g, (_, i) => {
    const block = blocks[Number(i)];
    if (block.startsWith('$$') && block.endsWith('$$')) {
      return `<div class="math-display">${block}</div>`;
    }
    return block;
  });
}

/** Normalize references section to uniform small paragraphs */
function formatReferencesHtml(html) {
  return html.replace(
    /(<h2>References<\/h2>)([\s\S]*?)(?=\s*$)/,
    (_, heading, body) => {
      const normalized = body
        .replace(/<h3>([^<]+)<\/h3>/g, '<p class="reference">$1</p>')
        .replace(/<p>(?![^<]*class="reference")/g, '<p class="reference">');
      return heading + '\n<section class="references">' + normalized + '</section>';
    }
  );
}

function build() {
  const writeupPath = path.join(ROOT, 'writeup.md');
  const templatePath = path.join(SRC, 'template.html');
  const stylePath = path.join(SRC, 'style.css');

  if (!fs.existsSync(writeupPath)) {
    console.error('Error: writeup.md not found.');
    process.exit(1);
  }

  let markdown = fs.readFileSync(writeupPath, 'utf8');

  if (!markdown.trim()) {
    console.warn('');
    console.warn('Warning: writeup.md is empty. Save your content to disk, then re-run:');
    console.warn('  node build.js');
    console.warn('');
  }

  markdown = preprocessHeadings(markdown);
  markdown = normalizeWhitespace(markdown);
  markdown = terminateTables(markdown);
  markdown = fixEmptyTableHeaders(markdown);
  markdown = preprocessDiagrams(markdown);
  markdown = preprocessMath(markdown);
  markdown = preprocessContent(markdown);

  marked.setOptions({
    gfm: true,
    breaks: false,
  });

  const { markdown: safeMd, blocks: mathBlocks } = protectMath(markdown);
  let contentHtml = marked.parse(safeMd);
  contentHtml = restoreMath(contentHtml, mathBlocks);
  contentHtml = formatReferencesHtml(contentHtml);
  const template = fs.readFileSync(templatePath, 'utf8');
  const finalHtml = template.replace('{{CONTENT}}', () => contentHtml);

  ensureDir(DOCS);
  fs.writeFileSync(path.join(DOCS, 'index.html'), finalHtml, 'utf8');
  fs.copyFileSync(stylePath, path.join(DOCS, 'style.css'));

  if (fs.existsSync(DIAGRAMS_SRC)) {
    copyRecursive(DIAGRAMS_SRC, DIAGRAMS_DEST);
  }

  console.log('');
  console.log('✓ Build complete.');
  console.log('  Output: docs/index.html');
  console.log('  Assets: docs/style.css, docs/diagrams/');
  console.log('');
  console.log('Preview locally:  npx serve docs');
  console.log('');
}

build();
