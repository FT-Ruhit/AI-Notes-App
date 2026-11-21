'use strict';
// Lecture studio: upload a recording to POST /jobs, poll GET /jobs/{id} for progress
// and live transcript, then show the notes from GET /jobs/{id}/notes.
// The job id is kept in the URL hash so a page refresh reconnects to the same job.

const el = id => document.getElementById(id);

const POLL_MS = 1200;            // Normal progress polling.
const STOPPING_POLL_MS = 500;    // Faster polling while a stop is being confirmed.
const RECONNECT_MS = 5000;       // Wait after a failed request.
const REQUEST_TIMEOUT_MS = 15000;

const STAGE_MESSAGES = {
  queued: 'Waiting for transcription.',
  transcribing: 'Transcribing your recording…',
  generating: 'Writing your study notes…',
  ready: 'Your study notes are ready.'
};
const PROGRESS_STAGES = ['queued', 'transcribing', 'generating', 'ready'];

// Friendly names for known faster-whisper models: [label, description].
const MODEL_PRESETS = new Map([
  ['tiny.en', ['Fastest', 'For a quick draft of a clearly recorded lecture. May miss more words.']],
  ['small', ['Balanced', 'A good mix of speed and accuracy for everyday lectures.']],
  ['medium', ['More accurate', 'Takes longer, but usually handles difficult words more accurately.']],
  ['large-v3-turbo', ['Extra hard effort (Turbo)', 'A faster version of the large model for long lectures.']],
  ['large-v3', ['Most accurate', 'Choose this when detail matters most. Allow extra time and memory.']]
]);

let job = null;          // Current job id.
let cursor = 0;          // Number of transcript segments already shown.
let markdown = '';       // Raw notes, for the copy button.
let paused = false;
let modelsReady = false;
let timer;
// Incremented whenever polling restarts or stops. A poll that finds a newer
// version has been superseded and drops its result.
let version = 0;

const requestOptions = () => ({cache: 'no-store', signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)});

// ---- Model picker -------------------------------------------------------------------

function explainModel() {
  const preset = MODEL_PRESETS.get(el('model').value);
  el('modelHelp').textContent = preset ? preset[1] : 'Uses the transcription model configured on this server.';
}

async function loadModels() {
  el('modelRetry').hidden = true;
  try {
    const response = await fetch('/models', requestOptions());
    if (!response.ok) throw Error();
    const data = await response.json();
    const choices = [...MODEL_PRESETS.keys()].filter(model => data.options.includes(model));
    if (!choices.includes(data.default)) choices.push(data.default);
    el('model').replaceChildren(...choices.map(model => {
      const label = MODEL_PRESETS.get(model)?.[0] || 'Server setting';
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

// ---- File picker and drag and drop --------------------------------------------------

function showFile() {
  const file = el('file').files[0];
  el('fileTitle').textContent = file ? file.name : 'Choose a recording';
  el('fileInfo').textContent = file
    ? 'Replace recording · ' + (file.size / 1048576).toFixed(1) + ' MB'
    : 'or drag and drop your audio file';
}

function onDrop(event) {
  event.preventDefault();
  el('drop').classList.remove('drag');
  if (!el('fields').disabled && event.dataTransfer.files.length === 1) {
    el('file').files = event.dataTransfer.files;
    showFile();
  }
}

// ---- Work panel ---------------------------------------------------------------------

function status(message, error = false) {
  if (el('status').textContent !== message) el('status').textContent = message;
  el('status').classList.toggle('error', error);
}

// Reset the panel for a new or reconnected job.
function showWork(name) {
  clearTimeout(timer);
  version++;
  cursor = 0;
  markdown = '';
  paused = false;
  el('empty').hidden = true;
  el('work').hidden = false;
  el('fields').disabled = true;
  for (const id of ['result', 'retry', 'live', 'wait', 'meter', 'stop']) el(id).hidden = true;
  el('stop').disabled = false;
  for (const id of ['elapsed', 'copyState', 'transcript', 'preview']) el(id).replaceChildren();
  el('recordName').textContent = name;
  el('live').open = true;
  el('wait').textContent = 'Pause live updates';
}

// The job is over without notes (failed, stopped or lost). Let the user try again.
function endWork(message, error = true) {
  clearTimeout(timer);
  el('fields').disabled = false;
  for (const id of ['meter', 'wait', 'retry', 'stop']) el(id).hidden = true;
  status(message, error);
}

function appendTranscript(segments) {
  const live = el('live');
  const transcript = el('transcript');
  live.hidden = false;
  // Follow new text only if the reader is already at the bottom.
  // scrollTop can be fractional even though the height measurements are rounded.
  const atBottom = transcript.scrollHeight - transcript.clientHeight - transcript.scrollTop <= 1;
  transcript.append((cursor ? ' ' : '') + segments.join(' '));
  if (live.open && atBottom) transcript.scrollTop = transcript.scrollHeight;
}

async function showNotes(url, current) {
  const response = await fetch(url + '/notes', requestOptions());
  if (!response.ok) throw Error();
  const raw = await response.text();
  if (current !== version) return;
  el('preview').replaceChildren(renderMarkdown(raw));
  markdown = raw;
  el('download').href = url + '/notes';
  el('result').hidden = false;
  el('wait').hidden = true;
  el('live').open = false;
}

// ---- Polling ------------------------------------------------------------------------

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
  try {
    const response = await fetch(url + '?after=' + cursor, requestOptions());
    if (current !== version) return;
    if (response.status === 404) {
      endWork('This job expired or the server restarted. Generate notes again to retry.');
      return;
    }
    if (!response.ok) throw Error();
    const data = await response.json();
    if (current !== version) return;

    el('retry').hidden = true;
    el('recordName').textContent = data.filename || 'Lecture recording';
    const time = data.elapsed_seconds;
    el('elapsed').textContent = data.model + ' · ' + Math.floor(time / 60) + 'm ' + (time % 60) + 's elapsed';
    if (data.segments.length) appendTranscript(data.segments);
    cursor = data.cursor;

    switch (data.stage) {
      case 'stopped':
        endWork('Generation stopped.', false);
        return;
      case 'stopping':
        el('stop').disabled = true;
        el('wait').hidden = true;
        status('Stopping generation…');
        timer = setTimeout(() => poll(current), STOPPING_POLL_MS);
        return;
      case 'failed':
        endWork(data.error || 'The recording could not be processed.');
        return;
    }

    el('meter').hidden = false;
    const step = PROGRESS_STAGES.indexOf(data.stage);
    Array.from(el('meter').children).forEach((bar, i) => bar.classList.toggle('on', i <= step));
    status(STAGE_MESSAGES[data.stage]);
    if (data.stage === 'ready') {
      el('stop').hidden = true;
      await showNotes(url, current);
      return;
    }
  } catch {
    if (current !== version) return;
    status('Could not retrieve the latest update. Reconnecting to this job…', true);
    el('retry').hidden = false;
    timer = setTimeout(() => poll(current), RECONNECT_MS);
    return;
  }
  timer = setTimeout(() => poll(current), POLL_MS);
}

// ---- Button handlers ----------------------------------------------------------------

async function upload(event) {
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
      endWork(typeof data.detail === 'string' ? data.detail : 'Upload failed (HTTP ' + response.status + ').');
      return;
    }
    job = (await response.json()).id;
    history.replaceState(null, '', '#' + job);
    resume();
  } catch {
    endWork('Could not confirm the upload. Your selections are saved here. Check your connection and try again.');
  }
}

async function stopJob() {
  clearTimeout(timer);
  const current = ++version;
  paused = false;
  el('stop').disabled = true;
  el('wait').hidden = true;
  el('retry').hidden = true;
  status('Stopping generation…');
  try {
    const response = await fetch('/jobs/' + job + '/stop', {method: 'POST', signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)});
    if (!response.ok) throw Error();
    await poll(current);
  } catch {
    el('stop').disabled = false;
    status('Could not confirm the stop. Check the job status or try stopping again.', true);
    el('retry').hidden = false;
    timer = setTimeout(() => poll(current), RECONNECT_MS);
  }
}

function togglePause() {
  if (paused) { resume(); return; }
  clearTimeout(timer);
  version++;
  paused = true;
  el('retry').hidden = true;
  el('wait').textContent = 'Resume waiting';
  status('Updates paused. Your recording is still processing.');
}

async function copyMarkdown() {
  try {
    await navigator.clipboard.writeText(markdown);
    el('copyState').textContent = 'Markdown copied.';
  } catch {
    el('copyState').textContent = 'Clipboard access is unavailable. Download notes to save the original Markdown.';
  }
}

function startOver() {
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
}

// ---- Wiring -------------------------------------------------------------------------

el('model').onchange = explainModel;
el('modelRetry').onclick = loadModels;
el('file').onchange = showFile;
el('drop').ondragover = event => { event.preventDefault(); if (!el('fields').disabled) el('drop').classList.add('drag'); };
el('drop').ondragleave = () => el('drop').classList.remove('drag');
el('drop').ondrop = onDrop;
el('upload').onsubmit = upload;
el('retry').onclick = resume;
el('stop').onclick = stopJob;
el('wait').onclick = togglePause;
el('copy').onclick = copyMarkdown;
el('another').onclick = startOver;

loadModels();
const savedJob = location.hash.slice(1);
if (/^[a-f0-9]{32}$/.test(savedJob)) {  // Only accept ids in the format the server issues.
  job = savedJob;
  showWork('Lecture recording');
  resume();
}
