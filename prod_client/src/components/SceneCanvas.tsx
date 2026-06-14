"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { Bounds, Center, OrbitControls, useGLTF } from "@react-three/drei";

function SceneModel({ url }: { url: string }) {
	// Vertex-colored, meshopt-compressed GLB. drei wires up the Meshopt decoder
	// by default, and there are no texture maps to load.
	const { scene } = useGLTF(url);
	return <primitive object={scene} />;
}

export default function SceneCanvas({ url }: { url: string }) {
	return (
		<Canvas dpr={[1, 2]} camera={{ position: [4, 3, 5], fov: 50, near: 0.01, far: 5000 }}>
			<color attach="background" args={["#0c0d10"]} />
			<hemisphereLight args={["#ffffff", "#202028", 1.0]} />
			<directionalLight args={["#ffffff", 1.1]} position={[3, 5, 4]} />
			<directionalLight args={["#ffffff", 0.5]} position={[-3, 2, -2]} />
			<Suspense fallback={null}>
				<Bounds key={url} fit clip observe margin={1.2}>
					<Center>
						<SceneModel url={url} />
					</Center>
				</Bounds>
			</Suspense>
			<OrbitControls makeDefault enableDamping dampingFactor={0.12} zoomToCursor />
		</Canvas>
	);
}
