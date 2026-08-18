"use client";

import { useEffect, useState } from "react";

const COUNT_MS = 950;

/** Owns its own frame loop, so the count-up never re-renders the scenes. */
export default function PctReadout({
	share,
	align,
}: {
	share: number;
	align: "left" | "right";
}) {
	const [value, setValue] = useState(0);

	useEffect(() => {
		let frame = 0;
		const start = performance.now();
		const step = (now: number) => {
			const t = Math.min(1, (now - start) / COUNT_MS);
			setValue(Math.round(share * (1 - (1 - t) ** 3)));
			if (t < 1) frame = requestAnimationFrame(step);
		};
		frame = requestAnimationFrame(step);
		return () => cancelAnimationFrame(frame);
	}, [share]);

	return (
		<div className={`arena-pct arena-pct--${align}`}>
			<span className="arena-pct__label">VOTERS PICKED THIS</span>
			<span className="arena-pct__value">{value}%</span>
		</div>
	);
}
