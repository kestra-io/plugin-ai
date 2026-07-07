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

// A real browser always exposes Web Storage, but this env's `window.localStorage` lacks a working
// `getItem` (Node's experimental `--localstorage-file` shim), so composables that read persisted
// host state (e.g. the active tenant under `selectedTenant`) would throw at mount. Provide a
// minimal in-memory Storage so those reads return null instead of blowing up story rendering.
if (typeof window !== "undefined" && typeof window.localStorage?.getItem !== "function") {
    const store = new Map<string, string>();
    const storage: Storage = {
        getItem: (key) => (store.has(key) ? store.get(key)! : null),
        setItem: (key, value) => void store.set(key, String(value)),
        removeItem: (key) => void store.delete(key),
        clear: () => store.clear(),
        key: (index) => Array.from(store.keys())[index] ?? null,
        get length() {
            return store.size;
        },
    };
    Object.defineProperty(window, "localStorage", { configurable: true, value: storage });
}