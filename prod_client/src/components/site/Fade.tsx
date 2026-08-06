"use client";

import { type ReactNode, useEffect, useState } from "react";
import { useLeaving } from "./PageTransition";

export default function Fade({
	enter = 520,
	delay = 0,
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
