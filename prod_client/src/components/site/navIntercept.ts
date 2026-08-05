/**
 * IS THIS CLICK ONE WE MAY TAKE OVER?
 *
 * Both the About page's moon and the masthead's type hold a navigation back for a
 * moment so something can finish moving before the route changes. Deciding WHICH
 * clicks may be held is the fiddly half of that, and it is the same decision in
 * both places — so it is made once, here.
 *
 * EVERY ESCAPE HATCH STAYS OPEN. A middle click, a modifier, a download, a new tab,
 * another host, a fragment on this page — all of those are somebody asking for
 * something an animation has no business delaying, and they fall straight through
 * to the browser. Getting this list wrong is how a site ends up unable to
 * open a link in a new tab.
 *
 * Returns the route to go to once the animation is done, or null to leave the click
 * entirely alone.
 */
export function heldNavigation(
	event: MouseEvent | React.MouseEvent,
	here: string,
): string | null {
	if (event.defaultPrevented) return null;
	// Anything but a plain primary click belongs to the browser.
	if (event.button !== 0) return null;
	if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
		return null;
	}

	const link = (event.target as HTMLElement | null)?.closest?.("a");
	const href = link?.getAttribute("href");
	if (!link || !href) return null;
	if (link.target === "_blank" || link.hasAttribute("download")) return null;

	// Internal routes only — an absolute URL or a fragment is not ours to hold.
	if (!href.startsWith("/")) return null;
	// And never the page already open: there is nothing to transition to.
	if (href === here) return null;

	return href;
}
