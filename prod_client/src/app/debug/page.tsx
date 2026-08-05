"use client";

import { Loader } from "@react-three/drei";
import SceneCanvas from "@/components/SceneCanvas";
import SceneGate from "@/components/SceneGate";
import ViewerHeader from "./ViewerHeader";
import { previewUrl } from "@/lib/scenes";

export default function Page() {
	return (
		<main className="relative flex h-dvh flex-col overflow-hidden bg-ground text-ink">
			<ViewerHeader />

			<div className="relative flex-1">
				<SceneGate>{(scene) => <SceneCanvas url={previewUrl(scene)} />}</SceneGate>

				<p className="pointer-events-none absolute left-4 top-4 text-xs text-ink-40">
					drag to orbit
				</p>
			</div>

			<Loader />
		</main>
	);
}
