"use client";

import { Loader } from "@react-three/drei";
import SceneCanvas from "@/components/SceneCanvas";
import ViewerHeader from "@/components/ViewerHeader";

export default function Page() {
	return (
		<main className="relative flex h-dvh flex-col overflow-hidden bg-neutral-950 text-neutral-100">
			<ViewerHeader />

			<div className="relative flex-1">
				<SceneCanvas />

				<p className="pointer-events-none absolute left-4 top-4 text-xs text-neutral-500">
					drag to orbit
				</p>
			</div>

			<Loader />
		</main>
	);
}
