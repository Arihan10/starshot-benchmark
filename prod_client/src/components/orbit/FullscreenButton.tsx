"use client";

// Corner brackets rather than arrows: the shape reads as "the frame changes
// size", which is what happens, and it stays legible at 16px where an arrow
// glyph turns to mush. Drawn inline so the icon inherits `currentColor` and
// follows the button's hover state without a second rule.
const EXPAND = ["M8 3H5a2 2 0 0 0-2 2v3", "M21 8V5a2 2 0 0 0-2-2h-3", "M3 16v3a2 2 0 0 0 2 2h3", "M16 21h3a2 2 0 0 0 2-2v-3"];
const COLLAPSE = ["M8 3v3a2 2 0 0 1-2 2H3", "M21 8h-3a2 2 0 0 1-2-2V3", "M3 16h3a2 2 0 0 1 2 2v3", "M16 21v-3a2 2 0 0 1 2-2h3"];

/**
 * The viewer's only chrome.
 *
 * It rests at low opacity and comes up to full on hover of the panel (the parent
 * carries `group`), so the scene is never competing with a control — but the
 * control is still THERE at rest rather than hidden, because a button that only
 * exists once you happen to hover the right region is a button most people never
 * find.
 */
export default function FullscreenButton({
	isFullscreen,
	onToggle,
}: {
	isFullscreen: boolean;
	onToggle: () => void;
}) {
	const label = isFullscreen ? "Exit full screen" : "Full screen";
	return (
		<button
			type='button'
			onClick={onToggle}
			title={label}
			aria-label={label}
			aria-pressed={isFullscreen}
			className='absolute right-4 top-4 z-30 flex h-9 w-9 items-center justify-center rounded-lg border border-mark-8 bg-ground/40 text-ink-64 opacity-40 backdrop-blur transition duration-200 hover:border-mark-16 hover:bg-ground/60 hover:text-ink group-hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-mark-40'
		>
			<svg
				viewBox='0 0 24 24'
				aria-hidden='true'
				className='h-4 w-4'
				fill='none'
				stroke='currentColor'
				strokeWidth={1.8}
				strokeLinecap='round'
				strokeLinejoin='round'
			>
				{(isFullscreen ? COLLAPSE : EXPAND).map((d) => (
					<path key={d} d={d} />
				))}
			</svg>
		</button>
	);
}
