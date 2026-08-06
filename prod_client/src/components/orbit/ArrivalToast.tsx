"use client";

import { useEffect, useState } from "react";
import type { OrbitState } from "@/lib/orbit/types";

export default function ArrivalToast({
	arrival,
	trapped,
}: {
	arrival: NonNullable<OrbitState["arrival"]>;
	trapped: boolean;
}) {
	const [shown, setShown] = useState(true);
	useEffect(() => {
		const t = setTimeout(() => setShown(false), 2600);
		return () => clearTimeout(t);
	}, []);
	return (
		<div
			className={`pointer-events-none absolute bottom-20 left-1/2 z-20 -translate-x-1/2 rounded-full border border-mark-8 bg-ground/75 px-4 py-1.5 text-xs backdrop-blur transition-all duration-300 ${
				shown ? "opacity-100" : "translate-y-1 opacity-0"
			}`}
		>
			<span className='font-semibold text-ink'>{arrival.name}</span>
			<span className='text-ink-64'>
				{trapped ? " · sealed room" : ""} · {arrival.verb}
			</span>
		</div>
	);
}
