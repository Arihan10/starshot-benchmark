export function heldNavigation(
	event: MouseEvent | React.MouseEvent,
	here: string,
): string | null {
	if (event.defaultPrevented) return null;
	if (event.button !== 0) return null;
	if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
		return null;
	}

	const link = (event.target as HTMLElement | null)?.closest?.("a");
	const href = link?.getAttribute("href");
	if (!link || !href) return null;
	if (link.target === "_blank" || link.hasAttribute("download")) return null;

	if (!href.startsWith("/")) return null;
	if (href === here) return null;

	return href;
}
