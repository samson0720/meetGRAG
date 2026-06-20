let currentTabId = null;

// Maps source_video (e.g. "IETF 125_ IAB Open.mp4") → tabId
// Tracks which browser tab is playing each meeting video
const videoTabMap = new Map();

chrome.action.onClicked.addListener((tab) => {
    currentTabId = tab.id;
    chrome.sidePanel.open({ tabId: tab.id });
});

// Clean up videoTabMap when a tracked tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
    for (const [name, id] of videoTabMap) {
        if (id === tabId) { videoTabMap.delete(name); break; }
    }
    if (currentTabId === tabId) currentTabId = null;
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {

    // ── Sidepanel requests current tab URL to detect active meeting ──
    if (message.type === 'GET_CURRENT_TAB_INFO') {
        if (currentTabId === null) {
            sendResponse({ tabId: null, url: null });
            return;
        }
        chrome.tabs.get(currentTabId).then(tab => {
            sendResponse({ tabId: currentTabId, url: tab.url || null });
        }).catch(() => {
            sendResponse({ tabId: null, url: null });
        });
        return true;   // keep channel open for async response
    }

    // ── Legacy seek: TOC timeline clicks → seek the currently tracked tab ──
    if (message.type === 'SEEK_VIDEO' && currentTabId !== null) {
        _seekTab(currentTabId, message.targetTime);
    }

    // ── Direct tab seek: sidepanel already resolved the correct tabId ──
    if (message.type === 'SEEK_VIDEO_TAB') {
        _seekTab(message.tabId, message.targetTime);
    }

    // ── Citation timestamp clicks → find or open the correct meeting video tab ──
    if (message.type === 'OPEN_AND_SEEK') {
        const { videoName, videoUrl, targetTime } = message;

        if (!videoUrl) {
            // No URL mapping found; fall back to current tab
            if (currentTabId !== null) _seekTab(currentTabId, targetTime);
            return;
        }

        const existingTabId = videoTabMap.get(videoName);
        if (existingTabId !== undefined) {
            // Tab already tracked — activate it and seek
            chrome.tabs.update(existingTabId, { active: true }).then(() => {
                _seekTab(existingTabId, targetTime);
            }).catch(() => {
                // Tab was closed without us noticing; clear and retry
                videoTabMap.delete(videoName);
                _resolveAndSeek(videoName, videoUrl, targetTime);
            });
        } else {
            _resolveAndSeek(videoName, videoUrl, targetTime);
        }
    }
});

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Check whether currentTabId is already playing the requested video.
 * If yes → seek it in place. If no → open a new tab.
 */
function _resolveAndSeek(videoName, videoUrl, targetTime) {
    if (currentTabId === null) {
        _openVideoTab(videoName, videoUrl, targetTime);
        return;
    }
    chrome.tabs.get(currentTabId).then(tab => {
        if (tab && _sameYouTubeVideo(tab.url, videoUrl)) {
            // Current tab is already the right video — seek it directly
            videoTabMap.set(videoName, currentTabId);
            _seekTab(currentTabId, targetTime);
        } else {
            _openVideoTab(videoName, videoUrl, targetTime);
        }
    }).catch(() => {
        _openVideoTab(videoName, videoUrl, targetTime);
    });
}

/**
 * Returns true when both URLs refer to the same YouTube video.
 * Handles /watch?v=ID, /live/ID, and mixed combinations.
 */
function _sameYouTubeVideo(tabUrl, videoUrl) {
    try {
        const idA = _youtubeVideoId(tabUrl);
        const idB = _youtubeVideoId(videoUrl);
        return !!idA && idA === idB;
    } catch {
        return false;
    }
}

function _youtubeVideoId(url) {
    const u = new URL(url);
    if (u.pathname.startsWith('/live/')) return u.pathname.split('/')[2] || null;
    return u.searchParams.get('v');
}

function _seekTab(tabId, targetTime) {
    chrome.tabs.sendMessage(tabId, { type: 'SEEK_VIDEO', targetTime }).catch(async () => {
        try {
            await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
            await chrome.tabs.sendMessage(tabId, { type: 'SEEK_VIDEO', targetTime }).catch(() => {});
        } catch (e) {
            console.warn('[GRASP] Could not seek tab', tabId, e);
        }
    });
}

function _openVideoTab(videoName, videoUrl, targetTime) {
    // Append YouTube-style ?t= so the page opens near the right timestamp
    const urlWithTime = _appendTime(videoUrl, targetTime);

    chrome.tabs.create({ url: urlWithTime }).then(tab => {
        videoTabMap.set(videoName, tab.id);
        currentTabId = tab.id;   // also make it the current tab for TOC seeks

        // After the page finishes loading, also try a programmatic seek
        // (YouTube honours ?t= for VODs; this is a belt-and-suspenders fallback)
        function onUpdated(tabId, info) {
            if (tabId !== tab.id || info.status !== 'complete') return;
            chrome.tabs.onUpdated.removeListener(onUpdated);
            // Give the player a moment to initialise before seeking
            setTimeout(() => _seekTab(tab.id, targetTime), 1500);
        }
        chrome.tabs.onUpdated.addListener(onUpdated);
    });
}

// Appends &t=N (or ?t=N) to a URL for YouTube timestamp deep-linking
function _appendTime(url, seconds) {
    try {
        const u = new URL(url);
        u.searchParams.set('t', Math.floor(seconds) + 's');
        return u.toString();
    } catch {
        return url;
    }
}
