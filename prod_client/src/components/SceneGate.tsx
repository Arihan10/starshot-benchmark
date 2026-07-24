"use client";

import type { ReactNode } from "react";
import { useScene } from "@/components/SceneProvider";
import type { Scene } from "@/lib/scenes";

// Renders viewer content only once a scene is selected, showing a centered
// status message otherwise. Keeps the loading / error / empty handling in one
// place across the 3D, panorama, and orbit pages.
export default function SceneGate({ children }: { children: (scene: Scene) => ReactNode }) {
	const { selected, status, error } = useScene();

	if (status === "loading") return <Message>loading scenes…</Message>;
	if (status === "error") return <Message error>failed to load scenes: {error}</Message>;
	if (!selected) return <Message>no scenes published yet</Message>;

	return <>{children(selected)}</>;
}

function Message({ children, error }: { children: ReactNode; error?: boolean }) {
	return (
		<div className="absolute inset-0 flex items-center justify-center">
			<span className={`text-sm ${error ? "text-red-400" : "text-neutral-500"}`}>{children}</span>
		</div>
	);
}
