// Service worker: route the action click into the side panel when the
// browser has one, otherwise fall back to a standalone popup window.

const HAS_SIDE_PANEL = typeof chrome.sidePanel !== "undefined";

if (HAS_SIDE_PANEL) {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((e) => console.error("sidePanel behavior", e));
} else {
  chrome.action.onClicked.addListener(async (tab) => {
    await chrome.windows.create({
      url: chrome.runtime.getURL(`panel/panel.html?tabId=${tab.id}`),
      type: "popup",
      width: 1000,
      height: 720,
    });
  });
}
