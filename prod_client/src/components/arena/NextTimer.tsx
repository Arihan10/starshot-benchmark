"use client";

import { useCallback, useEffect, useRef } from "react";
import Button from "@/components/ui/Button";
import { linear, useProgress } from "./useProgress";

const LEAD_MS = 0;
const RUN_MS = 3000;

export default function NextTimer({
	onNext,
	paused = false,
}: {
	onNext: () => void;
	paused?: boolean;
}) {
	const t = useProgress(RUN_MS, LEAD_MS, linear);

	const fired = useRef(false);
	const advance = useCallback(() => {
		if (fired.current) return;
		fired.current = true;
		onNext();
	}, [onNext]);

	useEffect(() => {
		if (paused) return;
		const timer = window.setTimeout(advance, LEAD_MS + RUN_MS);
		return () => window.clearTimeout(timer);
	}, [advance, paused]);

	return (
		<Button
			onClick={advance}
			className="group/next flex w-full flex-col items-center justify-center gap-xs"
			style={{ animation: "content-swap 400ms ease both" }}
		>
			<span className="font-label text-2xs whitespace-nowrap">
				NEXT
			</span>
			<span className="block h-px w-full overflow-hidden bg-mark-16">
				<span
					className="block h-full w-full origin-left bg-mark-64"
					style={{ transform: `scaleX(${t})` }}
				/>
			</span>
		</Button>
	);
}
