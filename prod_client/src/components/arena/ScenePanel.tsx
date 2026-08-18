"use client";

import { useEffect, useRef } from "react";
import OrbitViewer, { type OrbitViewerHandle } from "@/components/OrbitViewer";
import type { LocalCell } from "@/lib/localScenes";
import PctReadout from "./PctReadout";
import { shatter } from "./shatter";
import type { Outcome } from "./vote";

export type PanelRole = "paired" | "expanded" | "pushed";

export default function ScenePanel({
	cell,
	outcome,
	role,
	align,
	share,
	warm,
	commitVia,
	onWalkChange,
}: {
	cell: LocalCell;
	outcome: Outcome;
	role: PanelRole;
	align: "left" | "right";
	share: number | null;
	warm: LocalCell["source"] | null;
	commitVia: (commit: () => void) => void;
	onWalkChange: (inside: boolean) => void;
}) {
	const frameRef = useRef<HTMLElement>(null);
	const fxRef = useRef<HTMLDivElement>(null);
	const viewerRef = useRef<OrbitViewerHandle>(null);

	useEffect(() => {
		if (outcome !== "lost") return;
		const fx = fxRef.current;
		const frame = frameRef.current;
		if (!fx || !frame) return;

		let live = true;
		let cancel = () => {};
		void viewerRef.current?.capture().then((snapshot) => {
			if (!live) return;
			cancel = shatter(fx, snapshot, frame.clientWidth, frame.clientHeight);
		});
		return () => {
			live = false;
			cancel();
		};
	}, [outcome]);

	return (
		<section
			ref={frameRef}
			className="arena-panel"
			data-outcome={outcome ?? "open"}
			data-role={role}
			inert={role === "pushed"}
		>
			<div className="arena-panel__view">
				<OrbitViewer
					ref={viewerRef}
					source={cell.source}
					warm={warm}
					commitVia={commitVia}
					onFocusedChange={onWalkChange}
				/>
			</div>

			<div aria-hidden className="arena-panel__dim" />
			<div aria-hidden className="arena-panel__glow" />
			<div aria-hidden ref={fxRef} className="arena-panel__fx" />

			{share !== null && <PctReadout share={share} align={align} />}
		</section>
	);
}
