// Build only known document elements. Lecture text and raw HTML always stay text.
function renderMarkdown(source) {
  function element(tag, text) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function inline(node, text) {
    const tokens = /(`[^`\n]+`|\*\*[^\n]+?\*\*|__[^\n]+?__|\*[^*\n]+?\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/gi;
    let end = 0;
    for (const match of text.matchAll(tokens)) {
      node.append(text.slice(end, match.index));
      const token = match[0];
      if (token.startsWith('[')) {
        const split = token.indexOf('](');
        const anchor = element('a', token.slice(1, split));
        anchor.href = token.slice(split + 2, -1);
        node.append(anchor);
      } else if (token.startsWith('`')) {
        node.append(element('code', token.slice(1, -1)));
      } else {
        const bold = token.startsWith('**') || token.startsWith('__');
        const size = bold ? 2 : 1;
        node.append(element(bold ? 'strong' : 'em', token.slice(size, -size)));
      }
      end = match.index + token.length;
    }
    node.append(text.slice(end));
  }

  const root = document.createDocumentFragment();
  const listLine = /^(\s*)([-+*]|\d+[.)])\s+(.+)$/;
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const lists = [];
  let paragraph = null;
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    const heading = line.match(/^ {0,3}(#{1,6})\s+(.+)$/);
    const item = line.match(listLine);
    if (!line.trim()) { paragraph = null; continue; }
    if (item) {
      paragraph = null;
      const depth = item[1].length;
      const type = /^\d/.test(item[2]) ? 'ol' : 'ul';
      while (lists.length && lists.at(-1).depth > depth) lists.pop();
      if (lists.at(-1)?.depth === depth && lists.at(-1).type !== type) lists.pop();
      if (lists.at(-1)?.depth !== depth) {
        const list = element(type);
        if (type === 'ol') list.start = parseInt(item[2], 10);
        (lists.at(-1)?.last || root).append(list);
        lists.push({type, depth, list});
      }
      const active = lists.at(-1);
      active.last = element('li');
      inline(active.last, item[3]);
      active.list.append(active.last);
      continue;
    }
    if (lists.length && /^\s+\S/.test(line)) {
      lists.at(-1).last.append(' ');
      inline(lists.at(-1).last, line.trim());
      continue;
    }
    lists.length = 0;
    if (/^\s*(```|~~~)/.test(line)) {
      const fence = line.trimStart().slice(0, 3);
      const body = [];
      while (++index < lines.length && !lines[index].trimStart().startsWith(fence)) body.push(lines[index]);
      const pre = element('pre');
      pre.append(element('code', body.join('\n')));
      root.append(pre);
    } else if (heading || /^>\s?/.test(line)) {
      const block = element(heading ? 'h' + heading[1].length : 'blockquote');
      inline(block, heading ? heading[2] : line.replace(/^>\s?/, ''));
      root.append(block);
    } else if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      root.append(element('hr'));
    } else {
      if (!paragraph) { paragraph = element('p'); root.append(paragraph); }
      else paragraph.append(' ');
      inline(paragraph, line);
      continue;
    }
    paragraph = null;
  }
  return root;
}
