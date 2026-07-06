// jsdom lacks a handful of browser APIs the design-system / rendered components touch at
// mount time. Stub the ones story rendering needs so `play` functions can run headlessly.
if (typeof document !== "undefined") {
    const d = document as unknown as Record<string, unknown>;
    if (typeof d.queryCommandSupported !== "function") d.queryCommandSupported = () => false;
    if (typeof d.queryCommandState !== "function") d.queryCommandState = () => false;
    if (typeof d.execCommand !== "function") d.execCommand = () => false;
}

if (typeof window !== "undefined") {
    if (!window.matchMedia) {
        window.matchMedia = ((query: string) => ({
            matches: false,
            media: query,
            onchange: null,
            addListener: () => {},
            removeListener: () => {},
            addEventListener: () => {},
            removeEventListener: () => {},
            dispatchEvent: () => false,
        })) as never;
    }
    for (const name of ["ResizeObserver", "IntersectionObserver"]) {
        if (!(name in window)) {
            (window as unknown as Record<string, unknown>)[name] = class {
                observe() {}
                unobserve() {}
                disconnect() {}
                takeRecords() {
                    return [];
                }
            };
        }
    }
}

if (typeof Element !== "undefined" && !Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = () => {};
}
