'use strict';
const el = id => document.getElementById(id);
const stages = {
  queued: 'Waiting for transcription.',
  transcribing: 'Transcribing your recording…',
  generating: 'Writing your study notes…',
  ready: 'Your study notes are ready.'
};
let job, cursor = 0, timer, version = 0, markdown = '', paused = false;
let modelsReady = false;

const presets = new Map([
  ['tiny.en', ['Fastest', 'For a quick draft of a clearly recorded lecture. May miss more words.']],
  ['small', ['Balanced', 'A good mix of speed and accuracy for everyday lectures.']],
  ['medium', ['More accurate', 'Takes longer, but usually handles difficult words more accurately.']],
  ['large-v3', ['Most accurate', 'Choose this when detail matters most. Allow extra time and memory.']]
]);

function explainModel() {
  const preset = presets.get(el('model').value);
  el('modelHelp').textContent = preset ? preset[1] : 'Uses the transcription model configured on this server.';
}
el('model').onchange = explainModel;

async function loadModels() {
  el('modelRetry').hidden = true;
  try {
    const response = await fetch('/models', {signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw Error();
    const data = await response.json();
    const choices = [...presets.keys()].filter(model => data.options.includes(model));
    if (!choices.includes(data.default)) choices.push(data.default);
    el('model').replaceChildren(...choices.map(model => {
      const label = presets.get(model)?.[0] || 'Server setting';
      const selected = model === data.default;
      return new Option(label + (selected ? ' (default)' : ''), model, selected, selected);
    }));
    modelsReady = true;
    explainModel();
  } catch {
    el('modelHelp').textContent = 'Could not load the available models. Check the server and try again.';
    el('modelRetry').hidden = false;
  }
  el('submit').disabled = !modelsReady;
}
el('modelRetry').onclick = loadModels;

function showFile() {
  const file = el('file').files[0];
  el('fileTitle').textContent = file ? file.name : 'Choose a recording';
  el('fileInfo').textContent = file ? 'Replace recording · ' + (file.size / 1048576).toFixed(1) + ' MB' : 'or drag and drop your audio file';
}
el('file').onchange = showFile;
el('drop').ondragover = event => { event.preventDefault(); if (!el('fields').disabled) el('drop').classList.add('drag'); };
el('drop').ondragleave = () => el('drop').classList.remove('drag');
el('drop').ondrop = event => {
  event.preventDefault();
  el('drop').classList.remove('drag');
  if (!el('fields').disabled && event.dataTransfer.files.length === 1) {
    el('file').files = event.dataTransfer.files;
    showFile();
  }
};

function status(message, error = false) {
  if (el('status').textContent !== message) el('status').textContent = message;
  el('status').classList.toggle('error', error);
}
function showWork(name) {
  clearTimeout(timer);
  version++;
  cursor = 0;
  markdown = '';
  paused = false;
  el('empty').hidden = true;
  el('work').hidden = false;
  el('fields').disabled = true;
  for (const id of ['result','retry','live','wait','meter','stop']) el(id).hidden = true;
  el('stop').disabled = false;
  for (const id of ['elapsed','copyState','transcript','preview']) el(id).replaceChildren();
  el('recordName').textContent = name;
  el('live').open = true;
  el('wait').textContent = 'Pause live updates';
}
function fail(message) {
  clearTimeout(timer);
  el('fields').disabled = false;
  for (const id of ['meter','wait','retry','stop']) el(id).hidden = true;
  status(message, true);
}
function resume() {
  clearTimeout(timer);
  paused = false;
  el('wait').hidden = false;
  el('stop').hidden = false;
  el('wait').textContent = 'Pause live updates';
  el('retry').hidden = true;
  status('Checking recording progress…');
  poll(++version);
}
async function poll(current) {
  const url = '/jobs/' + job;
  const options = () => ({cache: 'no-store', signal: AbortSignal.timeout(15000)});
  try {
    const response = await fetch(url + '?after=' + cursor, options());
    if (current !== version) return;
    if (response.status === 404) { fail('This job expired or the server restarted. Generate notes again to retry.'); return; }
    if (!response.ok) throw Error();
    const data = await response.json();
    if (current !== version) return;
    el('retry').hidden = true;
    el('recordName').textContent = data.filename || 'Lecture recording';
    const time = data.elapsed_seconds;
    el('elapsed').textContent = data.model + ' · ' + Math.floor(time / 60) + 'm ' + (time % 60) + 's elapsed';
    if (data.segments.length) {
      const live = el('live');
      const transcript = el('transcript');
      live.hidden = false;
      // scrollTop can be fractional even though the height measurements are rounded.
      const atBottom = transcript.scrollHeight - transcript.clientHeight - transcript.scrollTop <= 1;
      transcript.append((cursor ? ' ' : '') + data.segments.join(' '));
      if (live.open && atBottom) transcript.scrollTop = transcript.scrollHeight;
    }
    cursor = data.cursor;
    if (data.stage === 'stopped') {
      el('fields').disabled = false;
      for (const id of ['meter','wait','retry','stop']) el(id).hidden = true;
      status('Generation stopped.');
      return;
    }
    if (data.stage === 'stopping') {
      el('stop').disabled = true;
      el('wait').hidden = true;
      status('Stopping generation…');
      timer = setTimeout(() => poll(current), 500);
      return;
    }
    if (data.stage === 'failed') { fail(data.error || 'The recording could not be processed.'); return; }
    el('meter').hidden = false;
    const step = ['queued','transcribing','generating','ready'].indexOf(data.stage);
    Array.from(el('meter').children).forEach((bar, i) => bar.classList.toggle('on', i <= step));
    status(stages[data.stage]);
    if (data.stage === 'ready') {
      el('stop').hidden = true;
      const response = await fetch(url + '/notes', options());
      if (!response.ok) throw Error();
      const raw = await response.text();
      if (current !== version) return;
      el('preview').replaceChildren(renderMarkdown(raw));
      markdown = raw;
      el('download').href = url + '/notes';
      el('result').hidden = false;
      el('wait').hidden = true;
      el('live').open = false;
      return;
    }
  } catch {
    if (current !== version) return;
    status('Could not retrieve the latest update. Reconnecting to this job…', true);
    el('retry').hidden = false;
    timer = setTimeout(() => poll(current), 5000);
    return;
  }
  timer = setTimeout(() => poll(current), 1200);
}
el('retry').onclick = resume;
el('stop').onclick = async () => {
  clearTimeout(timer);
  const current = ++version;
  paused = false;
  el('stop').disabled = true;
  el('wait').hidden = true;
  el('retry').hidden = true;
  status('Stopping generation…');
  try {
    const response = await fetch('/jobs/' + job + '/stop', {
      method: 'POST', signal: AbortSignal.timeout(15000)
    });
    if (!response.ok) throw Error();
    await poll(current);
  } catch {
    el('stop').disabled = false;
    status('Could not confirm the stop. Check the job status or try stopping again.', true);
    el('retry').hidden = false;
    timer = setTimeout(() => poll(current), 5000);
  }
};
el('wait').onclick = () => {
  if (paused) { resume(); return; }
  clearTimeout(timer);
  version++;
  paused = true;
  el('retry').hidden = true;
  el('wait').textContent = 'Resume waiting';
  status('Updates paused. Your recording is still processing.');
};
el('upload').onsubmit = async event => {
  event.preventDefault();
  if (!modelsReady || el('fields').disabled) return;
  const body = new FormData(el('upload'));
  showWork(el('file').files[0].name);
  history.replaceState(null, '', location.pathname);
  status('Uploading your recording…');
  el('status').focus();
  try {
    const response = await fetch('/jobs', {method: 'POST', body});
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      fail(typeof data.detail === 'string' ? data.detail : 'Upload failed (HTTP ' + response.status + ').');
      return;
    }
    job = (await response.json()).id;
    history.replaceState(null, '', '#' + job);
    resume();
  } catch {
    fail('Could not confirm the upload. Your selections are saved here. Check your connection and try again.');
  }
};
el('copy').onclick = async () => {
  try {
    await navigator.clipboard.writeText(markdown);
    el('copyState').textContent = 'Markdown copied.';
  } catch {
    el('copyState').textContent = 'Clipboard access is unavailable. Download notes to save the original Markdown.';
  }
};
el('another').onclick = () => {
  clearTimeout(timer);
  version++;
  job = null;
  history.replaceState(null, '', location.pathname);
  el('upload').reset();
  if (modelsReady) explainModel();
  el('fields').disabled = false;
  showFile();
  el('work').hidden = true;
  el('empty').hidden = false;
  el('file').focus();
};
loadModels();
const savedJob = location.hash.slice(1);
if (/^[a-f0-9]{32}$/.test(savedJob)) {
  job = savedJob;
  showWork('Lecture recording');
  resume();
}
