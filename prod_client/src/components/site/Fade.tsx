"use client";

import { type ReactNode, useEffect, useState } from "react";
import { useLeaving } from "./PageTransition";

/**
 * Anything that should arrive rather than appear, and leave rather than vanish.
 *
 * THREE PHASES, AND THE MIDDLE ONE IS NOT DECORATION. The entrance is a KEYFRAME on
 * purpose — a page that never hydrates still shows its content, where a transition
 * driven from state would leave it at zero for ever. But animations sit above
 * inline styles in the cascade, so an element cannot simply be faded out from under
 * one: the animation has to be called off first, and a transition needs a frame at
 * the old value before it will run to the new one. `holding` is that frame.
 *
 * INLINE STYLES RATHER THAN CLASSES, because the durations are props. Tailwind
 * generates rules by scanning source text for literal class names, so
 * `duration-[${leave}ms]` is a class nobody ever wrote and a rule that never
 * exists — the kind of bug that looks like the animation is broken.
 */
export default function Fade({
	/** Milliseconds to arrive in, or null for content that stages its own entrance
	 *  and only needs seeing out. */
	enter = 520,
	/** Held back this long, so a screen can arrive in the order it reads. */
	delay = 0,
	/** Milliseconds to leave in. Anything under the page's own HOLD is free. */
	leave = 260,
	className,
	children,
}: {
	enter?: number | null;
	delay?: number;
	leave?: number;
	className?: string;
	children: ReactNode;
}) {
	const leaving = useLeaving();
	const [phase, setPhase] = useState<"in" | "holding" | "out">("in");

	// BOTH STEPS INSIDE ANIMATION FRAMES, and keyed only on `leaving` so the effect
	// runs once per departure. Setting the first phase straight from the effect body
	// is a cascading render — and listing `phase` in the deps would re-run this the
	// moment it changed, cancelling the second frame from the previous run's
	// cleanup before it ever fired.
	useEffect(() => {
		if (!leaving) return;
		let settle = 0;
		const arm = requestAnimationFrame(() => {
			setPhase("holding");
			settle = requestAnimationFrame(() => setPhase("out"));
		});
		return () => {
			cancelAnimationFrame(arm);
			cancelAnimationFrame(settle);
		};
	}, [leaving]);

	const style: React.CSSProperties =
		phase === "in"
			? enter === null
				? {}
				: { animation: `veil-in ${enter}ms ease-out ${delay}ms both` }
			: {
					opacity: phase === "out" ? 0 : 1,
					transition: `opacity ${leave}ms ease-out`,
				};

	return (
		<div className={className} style={style}>
			{children}
		</div>
	);
}
