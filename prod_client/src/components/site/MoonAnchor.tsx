import type { CSSProperties, Ref } from "react";

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
