## 2026-04-17 - [External Link Decorative Icon Accessibility]
**Learning:** For accessibility in HTML/JS components, add `aria-hidden="true"` to decorative external link icons. Instead of appending a visually hidden `.sr-only` span, which can cause visual regressions if the CSS is missing or misapplied, prefer setting an `aria-label` directly on the link element (e.g., `link.setAttribute('aria-label', originalText + ' (opens in a new tab)');`).
**Action:** Update external link icon scripts to append the `aria-label` to the anchor tag itself, making sure to capture the original textContent before any new child nodes (like icons) are appended.

## 2026-04-18 - [Keyboard Navigation Focus State in Search Results]
**Learning:** Keyboard navigation (Up/Down arrows) added a dynamic `.active` class to search results via JavaScript, but there was no corresponding CSS styling. Consequently, screen reader or keyboard-only users couldn't see which item was focused. By appending `.search-result-item.active` to share the `.search-result-item:hover` styles, this was cleanly addressed without structural changes.
**Action:** When inspecting custom Javascript-driven keyboard navigation elements, always verify that the Javascript-toggled focus/active class is mirrored in the CSS to visually match the `:hover` state.

## 2026-04-21 - [Screen Reader Swallowing ARIA Labels]
**Learning:** Screen readers completely ignore an element's inner text if an `aria-label` is present. Therefore, if a link has existing text and you want to add contextual information for screen readers (like "(opens in a new tab)"), you must include the original text inside the `aria-label` instead of just appending context via a `.sr-only` span or using `aria-label` on a child element if that behavior could lead to confusing nesting. For `aria-label` appended dynamically, ensure you preserve the original `textContent`.
**Action:** Consistently set or append contextual string like "(opens in a new tab)" directly to the main element's `aria-label` attribute (incorporating its original text) rather than appending visually hidden child text nodes.

## 2026-04-23 - [Screen Reader Announcements for Dynamic Content]
**Learning:** For screen readers to announce dynamic content like search results when focus remains in an input field, you must use an `aria-live` region. Additionally, when using arrow keys to navigate a custom dropdown, you must manually update the `aria-live` region with the active item's text.
**Action:** Inject a visually hidden `aria-live="polite"` element and update its `textContent` when dynamic UI regions change state or list navigation occurs.

## 2024-04-24 - Accessibility for Custom Interactive Elements
**Learning:** Collapsible sections in `docs/assets/js/navigation.js` were built using standard `<div>` elements with only `click` event listeners. This entirely broke keyboard navigation and screen reader support, as non-semantic tags lack focusability (`tabindex="0"`) and default keyboard activation (`Enter`/`Space`).
**Action:** When building custom interactive components like collapsibles or dropdowns with non-interactive HTML elements (div/span), always explicitly add `role="button"`, `tabindex="0"`, `aria-expanded` state, and listen to `keydown` events for `Enter` and `Space` keys to restore native behavior.

## 2026-04-25 - [Keyboard Accessibility for JavaScript-Toggled Visibility]
**Learning:** Elements (like heading anchors) that are only made visible on `mouseenter` become invisible traps for keyboard navigators, who cannot see what element has currently received focus when tabbing. Additionally, symbols like "#" used as links are announced poorly (e.g., "number") by screen readers unless given contextual `aria-label`s.
**Action:** Always provide corresponding `focus` and `blur` event handlers on focusable elements if their visibility is dynamically toggled via `mouseenter`/`mouseleave`. Furthermore, ensure symbol-only links are given descriptive `aria-label`s capturing their functional context (e.g., "Link to section: [heading text]").

## 2026-04-29 - [Keyboard Focus Management with Programmatic Scrolling]
**Learning:** Intercepting anchor clicks (like "Skip to main content") with `e.preventDefault()` to apply smooth scrolling breaks native focus movement. If focus remains on the clicked link, subsequent `Tab` presses will not start from the target element, rendering the skip link ineffective for keyboard users.
**Action:** Always manually move focus to the target element (`target.focus({ preventScroll: true })`) and ensure it's focusable by temporarily setting `tabindex="-1"` when implementing programmatic smooth scrolling for in-page anchors.

## 2026-04-30 - [Keyboard Accessibility for Custom Dropdowns]
**Learning:** Custom dropdowns (like search results) that rely on `click` outside to close will remain open when a keyboard user presses `Tab` to navigate away from the input or container. This can leave floating content visible and obscure other parts of the page.
**Action:** When implementing custom dropdowns, always include a `document.addEventListener("focusin", ...)` listener to check if the newly focused element is outside the dropdown container, and if so, close the dropdown to maintain a clean UI for keyboard users.

## 2026-05-01 - [Semantic Context for Dynamic Navigation Links]
**Learning:** Dynamically applying visual `.active` classes to navigation items via client-side JavaScript leaves screen readers oblivious to the active state. Users relying on assistive technology will not know which link represents their current page if only visual classes are modified.
**Action:** When conditionally applying an `.active` class to indicate the current page or step, consistently apply `link.setAttribute("aria-current", "page")` simultaneously, and ensure you `removeAttribute("aria-current")` when the active state is removed.

## 2026-05-02 - [Keyboard Shortcuts Accessibility & OS Awareness]
**Learning:** Displaying hardcoded OS-specific keyboard shortcuts (like "Ctrl+K") in placeholders confuses Mac users, who expect "⌘K". Additionally, mapping common single-key shortcuts like "/" for search focus requires careful event handling to avoid intercepting the key when the user is already typing in an input field.
**Action:** When adding keyboard shortcuts for quick focus or actions, dynamically update visual hints (like placeholders) based on `navigator.userAgent.includes("Mac")`. Furthermore, always check `e.target.tagName` against `"INPUT"`, `"TEXTAREA"`, and `e.target.isContentEditable` before intercepting single-key shortcuts like "/".

## 2026-05-08 - [Keyboard Accessibility for Modal Drawer Menus]
**Learning:** Mobile "hamburger" menus that slide in (like `.sidebar`) act as modal drawers. If they only close on `click` outside, keyboard users are left with no standard way to dismiss them (`Escape` key) and can accidentally tab out of the menu into the obscured main content, breaking the logical focus flow.
**Action:** When implementing modal drawer menus or sidebars, always add a `keydown` listener for the `Escape` key (and return focus to the toggle button), as well as a `focusin` listener to auto-close the menu if the user tabs out of its container.

## 2026-05-09 - [Search Empty States as Dead Ends]
**Learning:** A generic "No results found" message is a UX dead end. Users experiencing an empty state often abandon the action because they feel stuck and receive no actionable guidance. Adding visual comfort (like an icon) and explicit suggestions ("Try adjusting your search terms") significantly improves the recovery rate from zero-result states.
**Action:** When designing or implementing search components or filtered lists, never use a plain text string for an empty state. Always provide a structured empty state with visual feedback (icon/illustration) and actionable guidance for the user's next step.

## 2026-05-10 - [Dynamic ARIA Labels for Feedback States]
**Learning:** When dynamically updating a button's visual text to provide feedback (e.g., changing 'Copy' to 'Copied!' or 'Failed'), if the button has a static `aria-label`, the screen reader will completely ignore the new visual text. Users relying on assistive technology will not receive any confirmation that their action succeeded or failed.
**Action:** Always simultaneously update the `aria-label` attribute (using `setAttribute`) whenever visual text is modified to indicate a state change or feedback, ensuring assistive technologies announce the updated state correctly.

## 2026-05-12 - [Keyboard Accessibility for Scrollable Code Blocks]
**Learning:** Code blocks (`<pre>`) often have `overflow-x: auto` applied via CSS to handle long lines of code without breaking the page layout. However, if these `<pre>` elements are not focusable, keyboard-only users cannot scroll them horizontally to read the truncated content.
**Action:** When styling elements with `overflow: auto` or `overflow: scroll` (especially code blocks or tables), always ensure they are keyboard-focusable by adding `tabindex="0"`. Additionally, provide a `role="region"` and an `aria-label` (e.g., "Code snippet") so screen readers announce the scrollable container properly.

## 2026-05-13 - [Keyboard Accessibility for Markdown Tables]
**Learning:** Markdown generators (like Kramdown) output raw `<table>` elements without wrapping them in containers. Consequently, adding `overflow-x: auto` directly to the `<table>` element often fails to contain horizontal overflow consistently or securely across different browsers, and without a wrapper, keyboard users cannot scroll wide tables horizontally.
**Action:** Always use a client-side JavaScript initialization script to locate raw `<table>` elements and wrap them in a `<div class="table-wrapper">` with `tabindex="0"`, `role="region"`, and a descriptive `aria-label` (e.g., "Data table"), while applying `overflow-x: auto` to the wrapper.

## 2026-05-15 - [Semantic Grouping for Navigation Submenus]
**Learning:** When building complex sidebar navigation with categorized sections (e.g., using `<div>` for titles above `<ul>` submenus), screen readers announce the submenus as isolated lists without context. Users cannot easily discern which category the list belongs to.
**Action:** Always associate navigation section titles with their respective submenus by giving the title an `id` and adding `aria-labelledby="[id]"` to the `<ul>` element. This ensures screen readers announce the category name when entering the list.

## 2026-05-18 - [Visual Hierarchy in Search Inputs]
**Learning:** When designing search inputs, the text placeholder alone isn't always enough to establish visual hierarchy. Adding a persistent search icon ensures immediate recognizability, while applying `pointer-events: none` prevents the icon from blocking the input's clickable area.
**Action:** Always add a visual search icon (like a magnifying glass) inside search inputs to improve immediate recognizability, ensuring it has `pointer-events: none` and appropriate padding on the input text.

## 2026-05-19 - [ARIA Combobox for Search Inputs]
**Learning:** For accessibility in custom search dropdowns (input + results list), implement the ARIA combobox pattern. Screen readers need these explicit roles and attributes to understand the relationship between the input field and the dynamic list of results, and to announce the currently active item during keyboard navigation.
**Action:** Add `role="combobox"`, `aria-expanded`, `aria-autocomplete`, and `aria-controls` to the input, `role="listbox"` to the dropdown container, and use `aria-activedescendant` on the input combined with `role="option"` and `aria-selected` on the results to announce active items.
## 2025-05-18 - Semantic Anchor IDs vs TOC Execution Order
**Learning:** Generating TOC elements with fallback IDs (`heading-0`) *before* semantic IDs are calculated breaks anchor functionality and produces inaccessible URLs. When multiple features depend on element IDs (like TOC generation and hover anchors), ID generation must be centralized and executed first.
**Action:** Always centralize DOM node ID generation in a dedicated initialization function that runs before any features that depend on those IDs. Track seen IDs to prevent duplicates and append counters if necessary.

## 2026-05-22 - [Keyboard Accessibility for Combobox Options]
**Learning:** In an ARIA combobox where `aria-activedescendant` is used to manage focus via arrow keys, leaving interactive elements (like `<a>` with `href`) in the dropdown list in the normal document tab order forces keyboard users to awkwardly tab through every single search result to reach the rest of the page.
**Action:** When implementing an ARIA combobox pattern, always ensure the child options (like search results) have `tabindex="-1"` applied, removing them from the tab sequence so users can efficiently bypass the dropdown while still navigating via arrow keys.

## 2024-05-23 - [Keyboard Focus Management on Combobox Escape]
**Learning:** When users dismiss a combobox dropdown (like search results) using the `Escape` key, automatically blurring the input causes them to lose their position in the tab order, forcing them to start navigating from the top of the document again.
**Action:** When handling the `Escape` key in a combobox, always call `e.preventDefault()` to stop the event from bubbling, and leave focus on the input field so users can seamlessly continue keyboard navigation.

## 2026-05-28 - [Respecting prefers-reduced-motion for Accessibility]
**Learning:** Animations, transitions, and smooth scrolling can cause discomfort or nausea for users with vestibular disorders. If these features are added without respecting the user's OS-level motion preferences, it creates a severe accessibility barrier.
**Action:** Always include a `@media (prefers-reduced-motion: reduce)` block in the global CSS to forcefully disable animations, transitions, and smooth scrolling site-wide when the user has requested reduced motion.

## 2026-05-29 - [Event Binding Order for Dynamic Content]
**Learning:** Attaching event listeners (like smooth scrolling behavior) to DOM elements using a generic selector early in the initialization sequence misses elements that are generated later dynamically (like Table of Contents or Heading anchors).
**Action:** Always ensure that behavioral initialization functions (like `initSmoothScroll`) run *after* all functions that dynamically generate the targeted DOM nodes.

## 2026-05-31 - [Search Input UX Dead Ends]
**Learning:** When users focus back into a search input containing a previous query, failing to select the text or re-display results creates a dead end requiring manual deletion or typing to trigger a search. Additionally, pressing Escape when results are closed should clear the input to easily reset the state.
**Action:** Add a `focus` event listener to select existing text and re-trigger searches, and handle the Escape key to clear the input if the dropdown is already hidden.

## 2026-05-32 - [Skip to Content Focus Targets]
**Learning:** Adding a "Skip to main content" link at the top of the document is a best practice for accessibility, but it only partially works if the target element (usually `<main id="main-content">`) is not programmatically focusable. When clicked or activated via keyboard, the browser scrolls to the target, but keyboard focus stays on the skip link itself, causing the user to start tabbing through the very navigation they were trying to bypass.
**Action:** When creating skip links, always ensure the target element has `tabindex="-1"`. This allows the browser to shift programmatic focus to the content area so subsequent `Tab` keystrokes continue from the main content.

## 2024-06-17 - Accessible Sidebar Navigation
**Learning:** When building complex sidebar navigation with categorized sections (e.g., using `<div>` for titles above `<ul>` submenus), always associate navigation section titles with their respective submenus by assigning an `id` to the title and adding `aria-labelledby="[id]"` to the `<ul>` element to ensure screen readers properly announce the category context.
**Action:** Apply this pattern using `<nav>` tags, structured `<ul>` / `<li>` lists, and `aria-labelledby` attributes for complex sectioned navigations.

## 2026-06-01 - [ARIA Tablist for Interactive Switchers]
**Learning:** When building interactive components that switch between content sections (like a journey map), standard buttons alone leave screen readers without context about the relationship between the controls and the content.
**Action:** Always implement the ARIA tablist pattern (`role="tablist"`, `role="tab"`, `aria-selected`, `role="tabpanel"`) and ensure the tabpanel has `tabIndex={0}` so keyboard users can shift focus to the newly revealed content.

## 2026-10-25 - [Semantic Grouping for Pill Badges]
**Learning:** When displaying groups of inline visual badges or pills (like tags or categories), flat `<span>` or `<div>` elements cause screen readers to read them as an unstructured text block. By wrapping them in a semantic `<ul role="list">` and `<li>` structure, screen readers will properly announce the item count and boundaries.
**Action:** Always wrap visual pill or tag groups in a `<ul role="list">` and apply CSS flexbox for wrapping to maintain visual layout while providing semantic meaning.


## 2026-07-21 - [Global Focus Outline Overriding Native Radii]
**Learning:** Applying `border-radius` directly within a global `:focus-visible` rule overrides the element's native border radius on focus, turning rounded pill buttons into squares. Modern browsers automatically curve the `outline` property to match the element's `border-radius` anyway.
**Action:** Never apply `border-radius` inside global focus-visible rules. Only apply `outline` and `outline-offset` to ensure the visual indicator conforms to the element's existing geometry.

## 2026-08-20 - [Spatial Visual Context for Active Links]
**Learning:** When sidebar navigation links indicate their active state only by changing text color and font weight, the change may be too subtle for users with low vision or cognitive impairments to quickly locate their current page within a dense menu. Sighted users benefit from a clear, spatial visual indicator (like a background block) corresponding to the `aria-current` state.
**Action:** When implementing navigation links with `aria-current="page"`, always provide a strong spatial visual state (e.g., adding `background: var(--panel2)`) alongside text changes to ensure clear spatial context for sighted users.

## 2026-08-22 - [Keyboard Navigation for ARIA Tablists]
**Learning:** Adding ARIA roles (`role="tablist"`, `role="tab"`) provides structural context to screen readers, but fails to deliver the expected interaction model if keyboard users cannot navigate between tabs using the arrow keys. Without manual focus management and roving `tabIndex` (`0` for active, `-1` for inactive), keyboard users are forced to awkwardly tab through every single inactive tab.
**Action:** When implementing an ARIA tablist, always ensure complete keyboard interaction by managing focus with `ArrowRight`/`ArrowLeft` handlers, setting `tabIndex={0}` on the active tab, and `tabIndex={-1}` on inactive tabs so the entire tablist acts as a single stop in the document's tab sequence.
