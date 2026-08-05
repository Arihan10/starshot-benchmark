import type { CSSProperties, Ref } from "react";

/**
 * WHERE THE MOON SHOULD BE, on this page, right now.
 *
 * An empty box that draws nothing and marks a rectangle. The moon itself lives once
 * in the root layout and never unmounts (see MoonStage); a page does not render a
 * moon, it says where one belongs, and the moon goes there.
 *
 * THAT INDIRECTION IS THE WHOLE POINT. When each page rendered its own moon, a
 * navigation destroyed one object and created another, and making that look
 * continuous took a held navigation, a flying clone, a baton of state passed
 * through a module, and reconstruction of the animation's progress from a clock on
 * the far side. Every bug in it was at that seam. With one moon and a moving
 * target, arriving on a new page changes a rectangle — and the thing that has to
 * travel was never destroyed, so there is nothing to hand over.
 *
 * `visibility: hidden` rather than `display: none`: it still has a box, and a box is
 * the entire contribution.
 */
export default function MoonAnchor({
	ref,
	className = "",
	style,
}: {
	ref?: Ref<HTMLDivElement>;
	className?: string;
	style?: CSSProperties;
}) {
	return (
		<div
			ref={ref}
			data-moon-anchor
			aria-hidden
			className={`pointer-events-none invisible ${className}`}
			style={style}
		/>
	);
}
