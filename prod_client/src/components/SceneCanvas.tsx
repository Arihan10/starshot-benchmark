"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { Bounds, Center, OrbitControls, useGLTF } from "@react-three/drei";
import { groundColor, markColor } from "@/lib/ink";

function SceneModel({ url }: { url: string }) {
	const { scene } = useGLTF(url);
	return <primitive object={scene} />;
}

export default function SceneCanvas({ url }: { url: string }) {
	const ground = groundColor();
	const mark = markColor();
	return (
		<Canvas dpr={[1, 2]} camera={{ position: [4, 3, 5], fov: 50, near: 0.01, far: 5000 }}>
			<color attach="background" args={[ground]} />
			<hemisphereLight args={[mark, ground, 1.0]} />
			<directionalLight args={[mark, 1.1]} position={[3, 5, 4]} />
			<directionalLight args={[mark, 0.5]} position={[-3, 2, -2]} />
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
