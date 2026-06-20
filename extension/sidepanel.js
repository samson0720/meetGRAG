// sidepanel.js
'use strict';

const API_BASE = 'http://localhost:9000/api/v1';

const TYPE_COLORS = {
    PERSON:        '#f472b6',
    ORGANIZATION:  '#fb923c',
    WORKING_GROUP: '#facc15',
    PROTOCOL:      '#34d399',
    RFC:           '#38bdf8',
    TECHNOLOGY:    '#a78bfa',
    CONCEPT:       '#94a3b8',
    UNKNOWN:       '#475569',
};

let SLIDES_DATA       = [];
let fullGraph         = null;
let graphSimId        = null;
let isQuerying        = false;
let msgCount          = 0;
const historyMap      = new Map();   // msgId → resp (stores full response per message)
let graphSearch         = null;      // search closure exposed by runForceGraph
let graphHighlightNodes = null;      // highlight-by-name-set closure exposed by runForceGraph
let currentMeeting    = null;        // { name, url } of the YouTube tab that opened this panel
let searchAllMeetings = false;       // when true, skip meeting scope filter

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dom = {
    connectBtn:    document.getElementById('connectBtn'),
    btnText:       document.getElementById('btnText'),
    btnIcon:       document.getElementById('btnIcon'),
    btnSpinner:    document.getElementById('btnSpinner'),
    analyzeBtn:    document.getElementById('analyzeBtn'),
    statusDot:     document.getElementById('status-dot'),
    statusText:    document.getElementById('status-text'),

    messages:      document.getElementById('messages'),
    queryInput:    document.getElementById('query-input'),
    sendBtn:       document.getElementById('send-btn'),

    answerBody:    document.getElementById('answer-body'),
    citeCount:     document.getElementById('cite-count'),

    graphCanvas:   document.getElementById('graph-canvas'),
    graphSvg:      document.getElementById('graphSvg'),
    graphLegend:   document.getElementById('graph-legend-row'),
    graphPlaceholder: document.getElementById('graph-placeholder'),
    nodeCount:     document.getElementById('node-count'),
    graphTooltip:  document.getElementById('graphTooltip'),

    videoSection:  document.getElementById('video-section'),
    v2Resizer:     document.getElementById('v2-resizer'),
    slideTitle:    document.getElementById('slideTitle'),
    slideTime:     document.getElementById('slideTime'),
    slideContent:  document.getElementById('slideContent'),
    transcriptContent: document.getElementById('transcriptContent'),
    tocList:       document.getElementById('tocList'),
    analysisRow:   document.getElementById('analysis-row'),
    analysisMessage: document.getElementById('analysis-message'),
    analysisProgressText: document.getElementById('analysis-progress-text'),
    analysisProgressBar: document.getElementById('analysis-progress-bar'),
};

// ── Status helpers ────────────────────────────────────────────────────────────
function setStatus(type, text) {
    dom.statusDot.className  = type;   // 'ok' | 'err' | 'loading'
    dom.statusText.textContent = text;
}

function showToast(msg, ms = 2800) {
    const el = document.getElementById('statusToast');
    el.textContent = msg;
    el.style.display = 'block';
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.style.display = 'none'; }, ms);
}

function now() {
    return new Date().toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' });
}

// ── YouTube video-ID extraction ──────────────────────────────────────────────
// Handles both /watch?v=ID and /live/ID URL formats
function _youtubeVideoId(url) {
    try {
        const u = new URL(url);
        if (u.hostname === 'youtu.be') return u.pathname.replace('/', '') || null;
        if (u.pathname.startsWith('/live/')) {
            return u.pathname.split('/')[2] || null;
        }
        return u.searchParams.get('v');
    } catch (_) { return null; }
}

function _isYouTubeWatchUrl(url) {
    try {
        const u = new URL(url);
        return /(^|\.)youtube\.com$/.test(u.hostname) || u.hostname === 'youtu.be';
    } catch (_) {
        return false;
    }
}

async function getActiveTabUrl() {
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    return activeTab?.url || '';
}

// ── Meeting scope UI ──────────────────────────────────────────────────────────
function updateScopeUI() {
    const btnCurrent  = document.getElementById('scope-btn-current');
    const btnAll      = document.getElementById('scope-btn-all');
    const meetLabel   = document.getElementById('scope-meeting-label');

    if (!currentMeeting) {
        // No meeting detected — lock to "All", disable "Current"
        btnCurrent.disabled = true;
        btnCurrent.classList.remove('active');
        btnAll.classList.add('active');
        meetLabel.textContent = '';
        return;
    }

    btnCurrent.disabled = false;
    if (searchAllMeetings) {
        btnAll.classList.add('active');
        btnCurrent.classList.remove('active');
        meetLabel.textContent = '';
    } else {
        btnCurrent.classList.add('active');
        btnAll.classList.remove('active');
        meetLabel.textContent = currentMeeting.name;
    }
}

document.getElementById('scope-btn-current').addEventListener('click', () => {
    if (!currentMeeting) return;
    searchAllMeetings = false;
    updateScopeUI();
});
document.getElementById('scope-btn-all').addEventListener('click', () => {
    searchAllMeetings = true;
    updateScopeUI();
});

// ── Chat messages ─────────────────────────────────────────────────────────────
function scrollMsgs() { dom.messages.scrollTop = dom.messages.scrollHeight; }

function appendUserMsg(text) {
    const row = document.createElement('div');
    row.className = 'msg-row user fade-in';
    row.innerHTML = `
        <div>
            <div class="bubble user">${escHtml(text)}</div>
            <div class="msg-meta">${now()}</div>
        </div>
        <div class="avatar user">Me</div>`;
    dom.messages.appendChild(row);
    scrollMsgs();
}

function appendTyping() {
    const row = document.createElement('div');
    row.className = 'msg-row bot fade-in';
    row.id = 'typing-row';
    row.innerHTML = `
        <div class="avatar bot">G</div>
        <div>
            <div class="bubble bot">
                <div class="typing-dots"><span></span><span></span><span></span></div>
            </div>
        </div>`;
    dom.messages.appendChild(row);
    scrollMsgs();
}

function replaceTypingWithAnswer(resp) {
    const row = document.getElementById('typing-row');
    if (!row) return;

    const idxMap = {};
    (resp.citations || []).forEach((c, i) => { idxMap[c.chunk_id] = i + 1; });

    const formatted = (resp.answer || '').replace(
        /\[REF:([^\]]+)\]/g,
        (_, id) => {
            const n = idxMap[id.trim()];
            return n ? `<sup class="ref-tag" data-idx="${n}" title="See [${n}]">[${n}]</sup>` : '';
        }
    );
    const fallback = resp.is_fallback
        ? '<div class="fallback">(No relevant data found — fallback response)</div>' : '';

    const msgId = `msg-${++msgCount}`;
    row.id = msgId;

    // Store resp for later retrieval when bubble is clicked
    historyMap.set(msgId, resp);

    row.innerHTML = `
        <div class="avatar bot">G</div>
        <div>
            <div class="bubble bot">${fallback}${formatted}</div>
            <div class="msg-meta">${now()} · ${resp.query_type || 'local'} search · click to view citations</div>
        </div>`;

    // Click bubble → show this response's citations in the right panel
    const bubble = row.querySelector('.bubble.bot');
    bubble.addEventListener('click', () => {
        // Deselect all other bot bubbles
        dom.messages.querySelectorAll('.bubble.bot').forEach(b => b.classList.remove('selected'));
        bubble.classList.add('selected');
        renderCitations(resp);
    });

    row.querySelectorAll('sup.ref-tag').forEach(el => {
        el.addEventListener('click', e => {
            e.stopPropagation();   // don't trigger bubble click
            highlightCitation(parseInt(el.dataset.idx));
        });
    });
    scrollMsgs();
}

function appendErrorMsg(message) {
    const row = document.getElementById('typing-row');
    if (!row) return;
    row.id = `msg-${++msgCount}`;
    row.innerHTML = `
        <div class="avatar bot">G</div>
        <div class="bubble bot" style="color:var(--danger)">${escHtml(friendlyError(message))}</div>`;
    scrollMsgs();
}

function friendlyError(raw) {
    if (!raw) return 'An unknown error occurred. Please try again.';
    const s = String(raw);
    if (/rate.limit|429|tokens per day/i.test(s)) return 'API usage limit reached. Please try again later.';
    if (/failed to fetch|networkerror|econnrefused/i.test(s)) return 'Cannot connect to server. Please check if the backend is running.';
    if (/timeout|timed out/i.test(s)) return 'Request timed out. Please try again.';
    if (/502|503|500/i.test(s)) return 'Server error. Please try again later.';
    return 'Failed to generate a response. Please try again.';
}

// ── Right panel: citations only ───────────────────────────────────────────────
function renderCitations(resp) {
    dom.citeCount.textContent = resp.citations?.length
        ? `${resp.citations.length} citation${resp.citations.length !== 1 ? 's' : ''}` : 'No citations';

    if (!resp.citations?.length) {
        dom.answerBody.innerHTML = '<div class="answer-placeholder">No direct source citations for this response — the answer was generated from high-level community summaries.</div>';
        return;
    }

    const VISIBLE_REFS = 3;

    function buildRefHtml(ref, citNum) {
        const slideTag = ref.slide_image
            ? `<span style="color:var(--accent);font-size:9px;margin-left:2px">▪ slide</span>` : '';
        const snippetHtml = ref.text_snippet
            ? `<button class="snippet-expand-btn" title="Show original text">&#9662; text</button>
               <div class="snippet-content">${escHtml(ref.text_snippet)}</div>` : '';
        return `
            <div class="source-ref">
                <span class="citation-num">${citNum}</span>
                <span class="video-name">${escHtml(ref.source_video || '—')}</span>
                <button class="ts-btn"
                    data-time="${ref.start_time}"
                    data-video="${escHtml(ref.source_video || '')}"
                    data-url="${escHtml(ref.video_url || '')}"
                    title="Jump to this timestamp">
                    ${fmtTs(ref.start_time)} – ${fmtTs(ref.end_time)}${slideTag}
                </button>
                ${snippetHtml}
            </div>`;
    }

    let citeHtml = '';
    resp.citations.forEach((c, i) => {
        const allRefs = c.source_refs || [];
        const visibleRefs  = allRefs.slice(0, VISIBLE_REFS);
        const hiddenRefs   = allRefs.slice(VISIBLE_REFS);

        const visibleHtml = visibleRefs.map(r => buildRefHtml(r, i+1)).join('');
        const hiddenHtml  = hiddenRefs.length
            ? `<div class="refs-overflow" style="display:none">${hiddenRefs.map(r => buildRefHtml(r, i+1)).join('')}</div>
               <button class="refs-more-btn" data-hidden-count="${hiddenRefs.length}">&#9662; ${hiddenRefs.length} more source${hiddenRefs.length > 1 ? 's' : ''}</button>`
            : '';

        const entityNamesJson = escHtml(JSON.stringify(c.entity_names || []));
        citeHtml += `
            <div class="citation-card" id="cit-${i+1}" data-idx="${i+1}" data-entities="${entityNamesJson}">
                ${visibleHtml || `<div class="source-ref"><span class="citation-num">${i+1}</span><span style="color:var(--dim);font-size:10px;font-style:italic">Community summary — no direct timestamp</span></div>`}
                ${hiddenHtml}
            </div>`;
    });

    const usageHtml = resp.usage && (resp.usage.prompt || resp.usage.completion)
        ? `<div class="usage-row">Tokens: ${(resp.usage.prompt||0) + (resp.usage.completion||0)}</div>` : '';

    dom.answerBody.innerHTML = `<div class="citations-list">${citeHtml}</div>${usageHtml}`;

    dom.answerBody.querySelectorAll('.ts-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const targetTime = parseFloat(btn.dataset.time);
            const videoName  = btn.dataset.video;
            const videoUrl   = btn.dataset.url;
            if (videoName && videoUrl) {
                // Check if the active tab is already playing this video —
                // if so, seek in place instead of opening a new tab.
                const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
                const activeVideoId = _youtubeVideoId(activeTab?.url || '');
                const citationVideoId = _youtubeVideoId(videoUrl);
                if (activeTab && activeVideoId && activeVideoId === citationVideoId) {
                    chrome.runtime.sendMessage({
                        type: 'SEEK_VIDEO_TAB',
                        tabId: activeTab.id,
                        targetTime,
                    });
                } else {
                    chrome.runtime.sendMessage({
                        type: 'OPEN_AND_SEEK',
                        videoName,
                        videoUrl,
                        targetTime,
                    });
                }
            } else {
                seekVideo(targetTime);   // fallback: seek current tab
            }
        });
    });

    dom.answerBody.querySelectorAll('.snippet-expand-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const content = btn.nextElementSibling;
            const isOpen = content.classList.toggle('open');
            btn.innerHTML = isOpen ? '&#9652; hide' : '&#9662; text';
        });
    });

    dom.answerBody.querySelectorAll('.refs-more-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            const overflow = btn.previousElementSibling;
            const isOpen = overflow.style.display !== 'none';
            overflow.style.display = isOpen ? 'none' : 'block';
            const n = btn.dataset.hiddenCount;
            btn.innerHTML = isOpen
                ? `&#9662; ${n} more source${parseInt(n) > 1 ? 's' : ''}`
                : `&#9652; hide`;
        });
    });

    // Citation card click → highlight related graph nodes
    let activeGraphCitation = null;
    dom.answerBody.querySelectorAll('.citation-card').forEach(card => {
        card.addEventListener('click', e => {
            if (e.target.closest('button')) return;  // ignore button clicks
            const idx = parseInt(card.dataset.idx);
            const names = JSON.parse(card.dataset.entities || '[]');

            if (activeGraphCitation === idx) {
                // Toggle off
                activeGraphCitation = null;
                card.classList.remove('graph-active');
                if (graphHighlightNodes) graphHighlightNodes(null);
            } else {
                // Deactivate previous card
                if (activeGraphCitation !== null) {
                    const prev = document.getElementById(`cit-${activeGraphCitation}`);
                    if (prev) prev.classList.remove('graph-active');
                }
                activeGraphCitation = idx;
                card.classList.add('graph-active');
                if (graphHighlightNodes) graphHighlightNodes(names.length ? new Set(names) : null);
            }
        });
    });

    highlightCitation(1);
}

function highlightCitation(idx) {
    dom.answerBody.querySelectorAll('.citation-card').forEach(el => el.classList.remove('active'));
    const card = document.getElementById(`cit-${idx}`);
    if (card) { card.classList.add('active'); card.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
}

// ── Send query ────────────────────────────────────────────────────────────────
dom.sendBtn.addEventListener('click', sendQuery);
dom.queryInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
});
dom.queryInput.addEventListener('input', () => {
    dom.queryInput.style.height = 'auto';
    dom.queryInput.style.height = Math.min(dom.queryInput.scrollHeight, 80) + 'px';
});

async function sendQuery() {
    const q = dom.queryInput.value.trim();
    if (!q || isQuerying) return;

    isQuerying = true;
    dom.sendBtn.disabled    = true;
    dom.queryInput.disabled = true;
    dom.queryInput.value    = '';
    dom.queryInput.style.height = 'auto';

    appendUserMsg(q);
    appendTyping();

    try {
        const body = { query: q, top_k: 5, score_cutoff: -0.5, max_chunks: 5 };
        if (!searchAllMeetings && currentMeeting) {
            body.current_meeting = currentMeeting.name;
        }
        const r = await fetch(`${API_BASE}/query`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(body),
        });
        if (!r.ok) {
            const err = await r.json().catch(() => ({ detail: r.statusText }));
            throw new Error(err.detail || `HTTP ${r.status}`);
        }
        const resp = await r.json();
        replaceTypingWithAnswer(resp);
        renderCitations(resp);
        buildSubgraph(resp);
    } catch (e) {
        appendErrorMsg(e.message);
    } finally {
        isQuerying              = false;
        dom.sendBtn.disabled    = false;
        dom.queryInput.disabled = false;
        dom.queryInput.focus();
    }
}

// ── Connect button ────────────────────────────────────────────────────────────
dom.connectBtn.addEventListener('click', connectToBackend);
dom.analyzeBtn.addEventListener('click', analyzeCurrentVideo);

async function connectToBackend() {
    dom.connectBtn.disabled        = true;
    dom.btnText.textContent        = 'Connecting...';
    dom.btnIcon.style.display      = 'none';
    dom.btnSpinner.style.display   = 'block';
    setStatus('loading', 'Connecting…');

    try {
        const healthRes = await fetch(`${API_BASE}/health`);
        if (!healthRes.ok) throw new Error(`Backend ${healthRes.status}`);
        const health = await healthRes.json();
        dom.analyzeBtn.disabled = false;

        setStatus('loading', 'Loading data…');
        const [slidesRes, graphRes, meetingsRes] = await Promise.all([
            fetch(`${API_BASE}/slides`),
            fetch(`${API_BASE}/graph`),
            fetch(`${API_BASE}/meetings`),
        ]);

        if (slidesRes.ok)   { const sd = await slidesRes.json(); SLIDES_DATA = sd.slides || []; }
        if (graphRes.ok)    { fullGraph = await graphRes.json(); }

        // Detect which meeting the current YouTube tab is playing
        if (meetingsRes.ok) {
            const meetings = await meetingsRes.json();   // [{name, url}, ...]
            // Query the currently active tab directly (requires "tabs" permission)
            const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const tabUrl = activeTab?.url || null;
            if (tabUrl) {
                const tabVideoId = _youtubeVideoId(tabUrl);
                for (const m of meetings) {
                    if (tabVideoId && tabVideoId === _youtubeVideoId(m.url)) {
                        currentMeeting = m;
                        break;
                    }
                }
            }
            updateScopeUI();
        }

        // Show video section when slides are available
        if (SLIDES_DATA.length > 0) {
            dom.videoSection.style.display = 'flex';
            dom.v2Resizer.style.display    = 'block';
            renderTOC();
        }

        const graphInfo = fullGraph ? ` · ${fullGraph.nodes.length} entities` : '';
        setStatus('ok', `Ready · ${health.model}`);
        dom.sendBtn.disabled = false;

        // Welcome bubble (only once)
        if (dom.messages.children.length === 0) {
            const row = document.createElement('div');
            row.className = 'msg-row bot fade-in';
            row.innerHTML = `
                <div class="avatar bot">G</div>
                <div>
                    <div class="bubble bot">
                        Hello! I'm GRASP, your meeting knowledge assistant.<br>
                        Loaded <strong>${SLIDES_DATA.length}</strong> slide segments${graphInfo}.<br>
                        Ask me anything about the meetings in this knowledge base!<br>
                        <span style="color:var(--dim);font-size:11px">e.g. "Who chairs the ADD working group?"</span>
                    </div>
                    <div class="msg-meta">${now()}</div>
                </div>`;
            dom.messages.appendChild(row);
            scrollMsgs();
        }

    } catch (err) {
        setStatus('err', 'Connection failed');
        showToast(`Connection failed: ${err.message}`);
        dom.analyzeBtn.disabled = true;
    } finally {
        dom.connectBtn.disabled      = false;
        dom.btnText.textContent      = 'Re-connect';
        dom.btnIcon.style.display    = 'block';
        dom.btnSpinner.style.display = 'none';
    }
}

function setAnalysisProgress(status) {
    const pct = Math.max(0, Math.min(100, parseInt(status.progress || 0)));
    dom.analysisRow.style.display = 'flex';
    dom.analysisProgressText.textContent = `${pct}%`;
    dom.analysisProgressBar.style.width = `${pct}%`;
    dom.analysisMessage.textContent = status.message || status.stage || status.status || 'Working...';
}

async function analyzeCurrentVideo() {
    if (dom.analyzeBtn.disabled) return;

    try {
        const tabUrl = await getActiveTabUrl();
        if (!_isYouTubeWatchUrl(tabUrl) || !_youtubeVideoId(tabUrl)) {
            showToast('Open a YouTube video tab first.');
            return;
        }

        dom.analyzeBtn.disabled = true;
        dom.connectBtn.disabled = true;
        dom.sendBtn.disabled = true;
        setStatus('loading', 'Analyzing video...');
        setAnalysisProgress({ progress: 0, message: 'Submitting analysis task...' });

        const res = await fetch(`${API_BASE}/analyze-youtube`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: tabUrl, auto_index: true }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const initial = await res.json();
        await pollAnalysisTask(initial.task_id);

        currentMeeting = null;
        searchAllMeetings = false;
        await connectToBackend();
        showToast('Video is indexed. You can ask questions now.', 3800);
    } catch (err) {
        setStatus('err', 'Analysis failed');
        showToast(`Analysis failed: ${err.message}`, 4800);
    } finally {
        dom.analyzeBtn.disabled = false;
        dom.connectBtn.disabled = false;
        dom.sendBtn.disabled = false;
    }
}

async function pollAnalysisTask(taskId) {
    while (true) {
        await new Promise(resolve => setTimeout(resolve, 2500));
        const res = await fetch(`${API_BASE}/analyze-youtube/${encodeURIComponent(taskId)}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const status = await res.json();
        setAnalysisProgress(status);
        if (status.status === 'done') {
            setStatus('ok', 'Indexed');
            return status;
        }
        if (status.status === 'failed') {
            throw new Error(status.error || status.message || 'Analysis failed');
        }
    }
}

// ── Subgraph ──────────────────────────────────────────────────────────────────
function buildSubgraph(resp) {
    if (!fullGraph) return;

    const primaryNames = new Set(resp.cited_entity_names || []);
    if (primaryNames.size === 0) {
        const commSet = new Set((resp.citations||[]).filter(c=>c.chunk_type==='community').map(c=>c.chunk_id));
        fullGraph.nodes.forEach(n => { if (commSet.has(n.community_id)) primaryNames.add(n.id); });
    }

    const relevantNames = new Set(primaryNames);
    fullGraph.links.forEach(l => {
        if (relevantNames.size >= 50) return;
        if (primaryNames.has(l.source)) relevantNames.add(l.target);
        if (primaryNames.has(l.target)) relevantNames.add(l.source);
    });

    if (relevantNames.size === 0) {
        const deg = {};
        fullGraph.links.forEach(l => { deg[l.source]=(deg[l.source]||0)+1; deg[l.target]=(deg[l.target]||0)+1; });
        [...fullGraph.nodes].sort((a,b)=>(deg[b.id]||0)-(deg[a.id]||0)).slice(0,40).forEach(n=>relevantNames.add(n.id));
    }

    const nodes = fullGraph.nodes
        .filter(n => relevantNames.has(n.id))
        .map(n => ({ ...n, isPrimary: primaryNames.has(n.id) }));
    const nameSet = new Set(nodes.map(n=>n.id));
    const links = fullGraph.links
        .filter(l => nameSet.has(l.source) && nameSet.has(l.target))
        .map(l => ({ ...l }));

    if (nodes.length === 0) return;

    dom.nodeCount.textContent = `${nodes.length} nodes · ${links.length} edges`;
    if (dom.graphPlaceholder) dom.graphPlaceholder.style.display = 'none';

    const applyTypeFilter = runForceGraph(nodes, links);
    renderGraphLegend(nodes, applyTypeFilter);
}

function renderGraphLegend(nodes, applyTypeFilter) {
    const types = [...new Set(nodes.map(n => n.type))];
    const hiddenTypes = new Set();

    dom.graphLegend.innerHTML = types.map(t => `
        <div class="legend-item legend-toggle" data-type="${escHtml(t)}" title="Click to show/hide ${escHtml(t)} nodes">
            <div class="legend-dot" style="background:${TYPE_COLORS[t]||TYPE_COLORS.UNKNOWN}"></div>
            ${escHtml(t)}
        </div>`).join('');

    dom.graphLegend.querySelectorAll('.legend-toggle').forEach(el => {
        el.addEventListener('click', () => {
            const type = el.dataset.type;
            if (hiddenTypes.has(type)) {
                hiddenTypes.delete(type);
                el.classList.remove('legend-off');
            } else {
                hiddenTypes.add(type);
                el.classList.add('legend-off');
            }
            applyTypeFilter(hiddenTypes);
        });
    });
}

// ── Force-directed graph ──────────────────────────────────────────────────────
function runForceGraph(nodes, links) {
    if (graphSimId) { cancelAnimationFrame(graphSimId); graphSimId = null; }

    const svgEl = dom.graphSvg;
    const NS = 'http://www.w3.org/2000/svg';
    const W  = svgEl.parentElement.clientWidth  || 320;
    const H  = svgEl.parentElement.clientHeight || 240;

    svgEl.setAttribute('width',   W);
    svgEl.setAttribute('height',  H);
    svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
    while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);

    const linkLayer = document.createElementNS(NS, 'g');
    const nodeLayer = document.createElementNS(NS, 'g');

    // Zoom/pan group wraps both layers
    let zk = 1, tx = 0, ty = 0;
    const zoomG = document.createElementNS(NS, 'g');
    zoomG.appendChild(linkLayer);
    zoomG.appendChild(nodeLayer);
    svgEl.appendChild(zoomG);

    function applyZoom() {
        zoomG.setAttribute('transform', `translate(${tx},${ty}) scale(${zk})`);
    }

    // Convert client event to SVG-unit coordinates (before zoom/pan)
    function clientToSvgRaw(e) {
        const rect = svgEl.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) * (W / rect.width),
            y: (e.clientY - rect.top)  * (H / rect.height),
        };
    }

    // Init node positions on circle (larger radius = fewer initial overlaps)
    const nodeMap = {};
    nodes.forEach((n, i) => {
        const angle = (2 * Math.PI * i) / nodes.length;
        const r = Math.min(W, H) * 0.40;
        n.x = W/2 + r*Math.cos(angle);
        n.y = H/2 + r*Math.sin(angle);
        n.vx = 0; n.vy = 0; n.dragged = false;
        n.color = TYPE_COLORS[n.type] || TYPE_COLORS.UNKNOWN;
        nodeMap[n.id] = n;
    });

    // Degree map for radius sizing
    const degMap = {};
    links.forEach(l => { degMap[l.source]=(degMap[l.source]||0)+1; degMap[l.target]=(degMap[l.target]||0)+1; });
    const nodeR = n => 6 + Math.sqrt(degMap[n.id]||1)*2.5;

    // Create link elements
    const linkEls = links.map(l => {
        const line = document.createElementNS(NS, 'line');
        line.setAttribute('stroke', 'rgba(255,255,255,0.55)');
        line.setAttribute('stroke-width', String(Math.max(0.5, Math.sqrt(l.weight||1)*0.8)));
        linkLayer.appendChild(line);
        return line;
    });

    // Create node group elements
    const nodeEls   = {};  // id → { g, inner, outer, txt }
    nodes.forEach(n => {
        const r = nodeR(n);
        const g = document.createElementNS(NS, 'g');
        g.setAttribute('cursor', 'pointer');
        g.dataset.nodeId = n.id;

        // Outer ring (visible for primary nodes)
        const outer = document.createElementNS(NS, 'circle');
        outer.setAttribute('r', String(r + 3));
        outer.setAttribute('fill', 'none');
        outer.setAttribute('stroke', n.isPrimary ? 'rgba(255,255,255,0.25)' : 'none');
        outer.setAttribute('stroke-width', '1.5');
        outer.dataset.nodeId = n.id;
        g.appendChild(outer);

        // Inner fill
        const inner = document.createElementNS(NS, 'circle');
        inner.setAttribute('r', String(r));
        inner.setAttribute('fill', n.color);
        inner.setAttribute('fill-opacity', n.isPrimary ? '1' : '0.65');
        inner.setAttribute('stroke', n.isPrimary ? '#fff' : 'none');
        inner.setAttribute('stroke-width', '1.5');
        inner.dataset.nodeId = n.id;
        g.appendChild(inner);

        // Label: only primary nodes show label by default; others rely on hover tooltip
        const txt = document.createElementNS(NS, 'text');
        txt.setAttribute('text-anchor', 'middle');
        txt.setAttribute('fill', '#e2e8f0');
        txt.setAttribute('font-size', '10');
        txt.setAttribute('font-family', 'Segoe UI,system-ui,sans-serif');
        txt.setAttribute('font-weight', n.isPrimary ? '600' : '400');
        txt.setAttribute('pointer-events', 'none');
        txt.setAttribute('dy', String(r + 11));
        txt.setAttribute('fill-opacity', '0');  // hidden by default; shown on hover or highlight
        txt.textContent = n.id.length > 14 ? n.id.slice(0, 13) + '…' : n.id;
        g.appendChild(txt);

        nodeLayer.appendChild(g);
        nodeEls[n.id] = { g, outer, inner, txt, r };
    });

    function updatePositions() {
        links.forEach((l, i) => {
            const a = nodeMap[l.source], b = nodeMap[l.target];
            if (!a || !b) return;
            linkEls[i].setAttribute('x1', String(a.x)); linkEls[i].setAttribute('y1', String(a.y));
            linkEls[i].setAttribute('x2', String(b.x)); linkEls[i].setAttribute('y2', String(b.y));
        });
        nodes.forEach(n => {
            const el = nodeEls[n.id];
            if (el) el.g.setAttribute('transform', `translate(${n.x},${n.y})`);
        });
    }

    // Physics constants
    const REPULSION = 2800, SPRING_K = 0.04, REST_LEN = 150, DAMPING = 0.84, GRAVITY = 0.010;
    const MAX_TICKS = 400;
    const SETTLE_V   = 0.08;   // max velocity considered "settled"
    const SETTLE_MIN = 60;     // don't stop before this tick even if velocities are low
    let tick = 0;

    function step() {
        // ── Force accumulation ────────────────────────────────────────────────
        nodes.forEach(n => { n.fx = 0; n.fy = 0; });

        // Coulomb repulsion between every pair
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i], b = nodes[j];
                const dx = b.x - a.x, dy = b.y - a.y;
                const d  = Math.sqrt(dx*dx + dy*dy) || 0.1;
                const f  = REPULSION / (d * d);
                const nx = dx / d, ny = dy / d;
                a.fx -= f * nx; a.fy -= f * ny;
                b.fx += f * nx; b.fy += f * ny;
            }
        }

        // Hooke spring along edges
        links.forEach(l => {
            const a = nodeMap[l.source], b = nodeMap[l.target];
            if (!a || !b) return;
            const dx = b.x - a.x, dy = b.y - a.y;
            const d  = Math.sqrt(dx*dx + dy*dy) || 0.1;
            const f  = (d - REST_LEN) * SPRING_K;
            const nx = dx / d, ny = dy / d;
            a.fx += f * nx; a.fy += f * ny;
            b.fx -= f * nx; b.fy -= f * ny;
        });

        // Weak center gravity
        nodes.forEach(n => {
            n.fx += (W / 2 - n.x) * GRAVITY;
            n.fy += (H / 2 - n.y) * GRAVITY;
        });

        // ── Velocity + position update ────────────────────────────────────────
        nodes.forEach(n => {
            if (n.dragged) return;
            n.vx = (n.vx + n.fx) * DAMPING;
            n.vy = (n.vy + n.fy) * DAMPING;
            n.x  = Math.max(20, Math.min(W - 20, n.x + n.vx));
            n.y  = Math.max(20, Math.min(H - 20, n.y + n.vy));
        });

        // ── Position-based collision resolution (prevents overlap) ────────────
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i], b = nodes[j];
                if (a.dragged && b.dragged) continue;
                const dx = b.x - a.x, dy = b.y - a.y;
                const d  = Math.sqrt(dx*dx + dy*dy) || 0.1;
                const minDist = nodeR(a) + nodeR(b) + 32;  // gap includes label clearance
                if (d < minDist) {
                    const push = (minDist - d) / 2;
                    const nx = dx / d, ny = dy / d;
                    if (!a.dragged) { a.x -= nx * push; a.y -= ny * push; a.vx *= 0.4; a.vy *= 0.4; }
                    if (!b.dragged) { b.x += nx * push; b.y += ny * push; b.vx *= 0.4; b.vy *= 0.4; }
                }
            }
        }
    }

    function animate() {
        step(); updatePositions(); tick++;
        // Stop early when kinetic energy drops below threshold (avoids endless micro-oscillation)
        const maxV = nodes.reduce((m, n) => Math.max(m, Math.abs(n.vx) + Math.abs(n.vy)), 0);
        const settled = tick >= SETTLE_MIN && maxV < SETTLE_V && !nodes.some(n => n.dragged);
        if (!settled && tick < MAX_TICKS) {
            graphSimId = requestAnimationFrame(animate);
        } else {
            graphSimId = null;
        }
    }
    animate();

    // ── Wheel zoom ────────────────────────────────────────────────────────────
    svgEl.addEventListener('wheel', e => {
        e.preventDefault();
        const raw = clientToSvgRaw(e);
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const newZk = Math.max(0.15, Math.min(6, zk * factor));
        // Keep the point under the cursor fixed
        tx = raw.x - (raw.x - tx) * (newZk / zk);
        ty = raw.y - (raw.y - ty) * (newZk / zk);
        zk = newZk;
        applyZoom();
    }, { passive: false });

    // ── Drag (node) + Pan (background) ───────────────────────────────────────
    let dragNode = null;
    let isPanning = false, panRaw0 = null, panTx0 = 0, panTy0 = 0;

    // Convert client event to content-space coords (inverse of zoom/pan)
    function svgCoords(e) {
        const raw = clientToSvgRaw(e);
        return { x: (raw.x - tx) / zk, y: (raw.y - ty) / zk };
    }

    svgEl.addEventListener('mousedown', e => {
        const id = e.target.dataset?.nodeId;
        if (id) {
            e.preventDefault();
            dragNode = nodeMap[id];
            if (!dragNode) return;
            dragNode.dragged = true; dragNode.vx = 0; dragNode.vy = 0;
            if (!graphSimId) { tick = 0; graphSimId = requestAnimationFrame(animate); }
        } else {
            isPanning = true;
            panRaw0 = clientToSvgRaw(e);
            panTx0 = tx; panTy0 = ty;
            svgEl.style.cursor = 'grabbing';
        }
    });
    svgEl.addEventListener('mousemove', e => {
        if (dragNode) {
            const pos = svgCoords(e);
            dragNode.x = Math.max(18, Math.min(W - 18, pos.x));
            dragNode.y = Math.max(18, Math.min(H - 18, pos.y));
            updatePositions();
        } else if (isPanning) {
            const raw = clientToSvgRaw(e);
            tx = panTx0 + (raw.x - panRaw0.x);
            ty = panTy0 + (raw.y - panRaw0.y);
            applyZoom();
        }
    });
    function endDrag() {
        if (dragNode) {
            dragNode.dragged = false; dragNode.vx = 0; dragNode.vy = 0;
            dragNode = null;
        }
        if (isPanning) {
            isPanning = false;
            svgEl.style.cursor = '';
        }
    }
    svgEl.addEventListener('mouseup',    endDrag);
    svgEl.addEventListener('mouseleave', endDrag);

    // Tracks which node IDs have labels pinned visible after a click-highlight
    let highlightedSet = new Set();

    // ── Click: highlight neighbors + node panel ────────────────────────────────
    nodeLayer.addEventListener('click', e => {
        const id = e.target.dataset?.nodeId;
        if (!id) return;
        e.stopPropagation();

        const n = nodeMap[id];
        if (n) showNodePanel(n);

        highlightedSet = new Set([id]);
        links.forEach(l => {
            if (l.source === id) highlightedSet.add(l.target);
            if (l.target === id) highlightedSet.add(l.source);
        });

        nodes.forEach(nd => {
            const el = nodeEls[nd.id];
            if (!el) return;
            const faded = !highlightedSet.has(nd.id);
            el.inner.setAttribute('fill-opacity', faded ? '0.12' : nd.isPrimary ? '1' : '0.65');
            el.txt.setAttribute('fill-opacity', faded ? '0' : '1');
        });
        linkEls.forEach((line, i) => {
            const s = links[i].source, t = links[i].target;
            line.setAttribute('stroke-opacity', (s===id||t===id) ? '0.8' : '0.04');
        });
    });

    // Click background → deselect (restore default visibility)
    svgEl.addEventListener('click', e => {
        if (e.target === svgEl || e.target === zoomG) {
            highlightedSet = new Set();
            nodes.forEach(nd => {
                const el = nodeEls[nd.id];
                if (!el) return;
                el.inner.setAttribute('fill-opacity', nd.isPrimary ? '1' : '0.65');
                el.txt.setAttribute('fill-opacity', '0');
            });
            linkEls.forEach(line => line.setAttribute('stroke-opacity', '0.25'));
            document.getElementById('node-panel').classList.remove('show');
        }
    });

    // ── Hover tooltip + inline label ──────────────────────────────────────────
    const tooltip = dom.graphTooltip;
    nodeLayer.addEventListener('mouseover', e => {
        const id = e.target.dataset?.nodeId;
        if (!id) return;
        const n = nodeMap[id];
        if (!n) return;

        // Show inline label temporarily if not already pinned visible
        if (!highlightedSet.has(id)) {
            const el = nodeEls[id];
            if (el) el.txt.setAttribute('fill-opacity', '1');
        }

        const desc = n.description
            ? `<div class="tt-desc">${escHtml(n.description.slice(0,130))}${n.description.length>130?'…':''}</div>` : '';
        const meetHtml = (n.meetings && n.meetings.length)
            ? `<div class="tt-meetings">${n.meetings.map(m => escHtml(m)).join(' · ')}</div>` : '';
        tooltip.innerHTML = `<div class="tt-name">${escHtml(n.id)}</div><div class="tt-type">${n.type}</div>${meetHtml}${desc}`;
        tooltip.style.display = 'block';
        positionTooltip(e);
    });
    nodeLayer.addEventListener('mousemove', e => { if (tooltip.style.display!=='none') positionTooltip(e); });
    nodeLayer.addEventListener('mouseout',  e => {
        const id = e.target.dataset?.nodeId;
        // Hide inline label on mouseout unless pinned by highlight
        if (id && !highlightedSet.has(id)) {
            const el = nodeEls[id];
            if (el) el.txt.setAttribute('fill-opacity', '0');
        }
        if (!e.relatedTarget?.dataset?.nodeId) tooltip.style.display = 'none';
    });

    function positionTooltip(e) {
        const wr = svgEl.parentElement.getBoundingClientRect();
        let left = e.clientX - wr.left + 10;
        let top  = e.clientY - wr.top  + 10;
        if (left + 200 > wr.width)  left = Math.max(0, wr.width - 205);
        if (top  + 100 > wr.height) top  = Math.max(0, wr.height - 105);
        tooltip.style.left = left + 'px';
        tooltip.style.top  = top  + 'px';
    }

    // ── Graph search (exposed as closure) ─────────────────────────────────────
    function resetNodeVisibility() {
        nodes.forEach(nd => {
            const el = nodeEls[nd.id];
            if (!el) return;
            el.inner.setAttribute('fill-opacity', nd.isPrimary ? '1' : '0.65');
            el.outer.setAttribute('stroke', nd.isPrimary ? 'rgba(255,255,255,0.25)' : 'none');
            el.txt.setAttribute('fill-opacity', '0');
        });
        linkEls.forEach(line => line.setAttribute('stroke-opacity', '0.25'));
    }

    graphSearch = (query) => {
        const q = query.trim().toLowerCase();
        if (!q) { resetNodeVisibility(); return -1; }

        const matched = new Set();
        nodes.forEach(nd => {
            if (nd.id.toLowerCase().includes(q) || (nd.type || '').toLowerCase().includes(q)) {
                matched.add(nd.id);
            }
        });

        nodes.forEach(nd => {
            const el = nodeEls[nd.id];
            if (!el) return;
            const hit = matched.has(nd.id);
            el.inner.setAttribute('fill-opacity', hit ? '1' : '0.07');
            el.outer.setAttribute('stroke', hit ? 'rgba(255,255,255,0.5)' : 'none');
            el.txt.setAttribute('fill-opacity', hit ? '1' : '0');
        });
        linkEls.forEach((line, i) => {
            const s = links[i].source, t = links[i].target;
            line.setAttribute('stroke-opacity', (matched.has(s) && matched.has(t)) ? '0.55' : '0.04');
        });

        return matched.size;
    };

    // Clear search when a new graph is loaded
    const searchInput = document.getElementById('graph-search-input');
    if (searchInput) { searchInput.value = ''; }
    const countEl = document.getElementById('graph-search-count');
    if (countEl) { countEl.textContent = ''; }
    const clearBtn = document.getElementById('graph-search-clear');
    if (clearBtn) { clearBtn.style.display = 'none'; }

    // ── Type filter (called by legend toggles) ────────────────────────────────
    function applyTypeFilter(hiddenTypes) {
        nodes.forEach(n => {
            const el = nodeEls[n.id];
            if (!el) return;
            el.g.setAttribute('visibility', hiddenTypes.has(n.type) ? 'hidden' : 'visible');
        });
        links.forEach((l, i) => {
            const srcHidden = hiddenTypes.has(nodeMap[l.source]?.type);
            const tgtHidden = hiddenTypes.has(nodeMap[l.target]?.type);
            linkEls[i].setAttribute('visibility', (srcHidden || tgtHidden) ? 'hidden' : 'visible');
        });
    }

    // ── Citation-driven node highlight ────────────────────────────────────────
    graphHighlightNodes = (nameSet) => {
        if (!nameSet || nameSet.size === 0) {
            // Reset to default visibility
            highlightedSet = new Set();
            nodes.forEach(nd => {
                const el = nodeEls[nd.id];
                if (!el) return;
                el.inner.setAttribute('fill-opacity', nd.isPrimary ? '1' : '0.65');
                el.txt.setAttribute('fill-opacity', '0');
            });
            linkEls.forEach(line => {
                line.setAttribute('stroke-opacity', '0.25');
            });
            return;
        }
        // Expand to 1-hop neighbors so isolated primary entities also show context
        const expanded = new Set(nameSet);
        links.forEach(l => {
            if (nameSet.has(l.source)) expanded.add(l.target);
            if (nameSet.has(l.target)) expanded.add(l.source);
        });
        highlightedSet = new Set(expanded);

        nodes.forEach(nd => {
            const el = nodeEls[nd.id];
            if (!el) return;
            const inSet = nameSet.has(nd.id);
            const neighbor = expanded.has(nd.id) && !inSet;
            el.inner.setAttribute('fill-opacity', inSet ? '1' : neighbor ? '0.45' : '0.08');
            el.txt.setAttribute('fill-opacity', inSet ? '1' : neighbor ? '0.6' : '0');
        });
        linkEls.forEach((line, i) => {
            const s = links[i].source, t = links[i].target;
            const connected = expanded.has(s) && expanded.has(t);
            line.setAttribute('stroke-opacity', connected ? '0.7' : '0.04');
        });
    };

    return applyTypeFilter;
}

// ── Graph search input ────────────────────────────────────────────────────────
(function () {
    const input    = document.getElementById('graph-search-input');
    const clearBtn = document.getElementById('graph-search-clear');
    const countEl  = document.getElementById('graph-search-count');

    function doSearch() {
        if (!graphSearch) return;
        const count = graphSearch(input.value);
        clearBtn.style.display = input.value ? 'block' : 'none';
        if (count < 0) {
            countEl.textContent = '';
        } else {
            countEl.textContent = count > 0 ? `${count} found` : 'No results';
            countEl.style.color = count > 0 ? 'var(--dim)' : 'var(--danger)';
        }
    }

    input.addEventListener('input', doSearch);
    input.addEventListener('keydown', e => { if (e.key === 'Escape') { input.value = ''; doSearch(); input.blur(); } });
    clearBtn.addEventListener('click', () => { input.value = ''; doSearch(); input.focus(); });
})();

// ── Node panel popup ──────────────────────────────────────────────────────────
function showNodePanel(n) {
    document.getElementById('np-name').textContent = n.id;
    document.getElementById('np-type').textContent = n.type + (n.community_id ? ` · Community ${String(n.community_id).slice(0,8)}…` : '');
    const meetEl = document.getElementById('np-meetings');
    if (meetEl) {
        if (n.meetings && n.meetings.length) {
            meetEl.textContent = n.meetings.join(' · ');
            meetEl.style.display = 'block';
        } else {
            meetEl.style.display = 'none';
        }
    }
    document.getElementById('np-desc').textContent = n.description || '(No description)';
    document.getElementById('node-panel').classList.add('show');
}
document.getElementById('np-close').addEventListener('click', e => {
    e.stopPropagation();
    document.getElementById('node-panel').classList.remove('show');
});

// ── Resizers ──────────────────────────────────────────────────────────────────
// Horizontal: left chat ↔ right detail
(function() {
    const resizer = document.getElementById('h-resizer');
    const chat    = document.getElementById('chat-panel');
    let drag=false, sx=0, sw=0;
    resizer.addEventListener('mousedown', e => {
        drag=true; sx=e.clientX; sw=chat.offsetWidth;
        resizer.classList.add('drag');
        document.body.style.cursor='col-resize'; document.body.style.userSelect='none';
    });
    document.addEventListener('mousemove', e => {
        if (!drag) return;
        chat.style.width = Math.max(160, Math.min(500, sw+e.clientX-sx)) + 'px';
    });
    document.addEventListener('mouseup', () => {
        if (!drag) return;
        drag=false; resizer.classList.remove('drag');
        document.body.style.cursor=''; document.body.style.userSelect='';
    });
})();

// Vertical: answer ↔ graph
(function() {
    const resizer = document.getElementById('v1-resizer');
    const ans     = document.getElementById('answer-section');
    let drag=false, sy=0, sh=0;
    resizer.addEventListener('mousedown', e => {
        drag=true; sy=e.clientY; sh=ans.offsetHeight;
        resizer.classList.add('drag');
        document.body.style.cursor='row-resize'; document.body.style.userSelect='none';
    });
    document.addEventListener('mousemove', e => {
        if (!drag) return;
        ans.style.height = Math.max(60, Math.min(450, sh+e.clientY-sy)) + 'px';
    });
    document.addEventListener('mouseup', () => {
        if (!drag) return;
        drag=false; resizer.classList.remove('drag');
        document.body.style.cursor=''; document.body.style.userSelect='';
    });
})();

// Vertical: graph ↔ video (drag resizer upward increases video height)
(function() {
    const resizer = document.getElementById('v2-resizer');
    const video   = document.getElementById('video-section');
    let drag=false, sy=0, sh=0;
    resizer.addEventListener('mousedown', e => {
        drag=true; sy=e.clientY; sh=video.offsetHeight;
        resizer.classList.add('drag');
        document.body.style.cursor='row-resize'; document.body.style.userSelect='none';
    });
    document.addEventListener('mousemove', e => {
        if (!drag) return;
        // dragging up = increasing video height
        video.style.height = Math.max(90, Math.min(520, sh-(e.clientY-sy))) + 'px';
    });
    document.addEventListener('mouseup', () => {
        if (!drag) return;
        drag=false; resizer.classList.remove('drag');
        document.body.style.cursor=''; document.body.style.userSelect='';
    });
})();

// ── Tab switching (video section) ─────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b===btn));
        document.getElementById('slide-tab').style.display = tab==='slide' ? 'block' : 'none';
        document.getElementById('toc-tab').style.display   = tab==='toc'   ? 'block' : 'none';
    });
});

// ── Video time sync (from content.js) ────────────────────────────────────────
chrome.runtime.onMessage.addListener(msg => {
    if (msg.type === 'VIDEO_TIME_UPDATE') updateVideoSync(msg.currentTime);
});

function updateVideoSync(currentTime) {
    if (!SLIDES_DATA.length) return;
    let current = null;
    for (const slide of SLIDES_DATA) {
        if (currentTime >= slide.time_range.start_sec) current = slide;
        else break;
    }
    if (!current) return;

    const min = Math.floor(currentTime/60).toString().padStart(2,'0');
    const sec = Math.floor(currentTime%60).toString().padStart(2,'0');
    dom.slideTime.textContent = `${min}:${sec}`;

    const title      = current.multimodal_content.visual_info.title;
    const visualData = current.multimodal_content.visual_info.content;
    const transcript = current.multimodal_content.audio_transcript;

    if (dom.slideTitle.textContent !== title) {
        dom.slideTitle.style.opacity = '0';
        setTimeout(() => { dom.slideTitle.textContent = title; dom.slideTitle.style.opacity = '1'; }, 200);
        updateTOCActiveState(current.slide_index);
    }

    let htmlContent = '';
    if (Array.isArray(visualData) && visualData.length > 0) {
        htmlContent = '<ul>' + visualData.map(it => `<li>${escHtml(it)}</li>`).join('') + '</ul>';
    } else if (typeof visualData === 'string' && visualData.trim()) {
        htmlContent = `<p>${escHtml(visualData)}</p>`;
    } else {
        htmlContent = "<p style='color:var(--dim);font-style:italic;font-size:11px'>No visual content</p>";
    }
    if (dom.slideContent.innerHTML !== htmlContent) {
        dom.slideContent.style.opacity = '0';
        setTimeout(() => { dom.slideContent.innerHTML = htmlContent; dom.slideContent.style.opacity = '1'; }, 200);
    }

    if (transcript?.trim()) {
        if (dom.transcriptContent.textContent !== transcript) {
            dom.transcriptContent.style.opacity = '0';
            setTimeout(() => { dom.transcriptContent.textContent = transcript; dom.transcriptContent.style.opacity = '1'; }, 200);
        }
    } else {
        dom.transcriptContent.textContent = 'No transcript for this segment';
        dom.transcriptContent.style.color = 'var(--dim)';
    }
}

// ── TOC ───────────────────────────────────────────────────────────────────────
function renderTOC() {
    if (!SLIDES_DATA.length) return;
    dom.tocList.innerHTML = '';
    SLIDES_DATA.forEach(slide => {
        const ts    = slide.time_range.display_timestamp;
        const title = slide.multimodal_content.visual_info.title;
        const item  = document.createElement('div');
        item.className = 'toc-item';
        item.dataset.slideIndex = slide.slide_index;
        item.setAttribute('role', 'button');
        item.setAttribute('tabindex', '0');
        item.innerHTML = `
            <span class="toc-time">${ts.substring(3)}</span>
            <span class="toc-title" title="${escHtml(title)}">${escHtml(title)}</span>`;
        item.addEventListener('click', () => seekVideo(slide.time_range.start_sec));
        item.addEventListener('keydown', e => {
            if (e.key==='Enter'||e.key===' ') { e.preventDefault(); seekVideo(slide.time_range.start_sec); }
        });
        dom.tocList.appendChild(item);
    });
}

function updateTOCActiveState(currentIndex) {
    dom.tocList.querySelectorAll('.toc-item').forEach(item => {
        const active = parseInt(item.dataset.slideIndex) === currentIndex;
        item.classList.toggle('active', active);
        if (active) item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
}

// ── Utilities ─────────────────────────────────────────────────────────────────
async function seekVideo(targetSeconds) {
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (activeTab) {
        chrome.runtime.sendMessage({ type: 'SEEK_VIDEO_TAB', tabId: activeTab.id, targetTime: targetSeconds });
    }
}

function fmtTs(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

function escHtml(s) {
    return String(s)
        .replace(/&/g,'&amp;').replace(/</g,'&lt;')
        .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function appendWelcomeMsg() {
    const row = document.createElement('div');
    row.className = 'msg-row bot fade-in';
    row.innerHTML = `
        <div class="avatar bot">G</div>
        <div>
            <div class="bubble bot" style="cursor:default">
                Ask any question about the meeting — GRASP will retrieve relevant segments from the knowledge graph and show you the exact video timestamps as sources.
            </div>
        </div>`;
    dom.messages.appendChild(row);
}

appendWelcomeMsg();
