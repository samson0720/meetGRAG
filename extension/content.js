// content.js
console.log("IETF Extension: Content script loaded");

function findVideo() {
    const video = document.querySelector('video');
    if (video) {
        console.log("✅ 偵測到影片元素！");
        video.ontimeupdate = function () {
            try {
                chrome.runtime.sendMessage({
                    type: "VIDEO_TIME_UPDATE",
                    currentTime: video.currentTime
                });
            } catch (e) {
                // Extension was reloaded — stop sending time updates
                video.ontimeupdate = null;
            }
        };
    } else {
        setTimeout(findVideo, 1000);
    }
}
findVideo();

chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "SEEK_VIDEO") {
        const video = document.querySelector('video');
        if (video) {
            video.currentTime = message.targetTime;
            console.log(`跳轉至影片時間: ${message.targetTime} 秒`);
        }
    }
});
