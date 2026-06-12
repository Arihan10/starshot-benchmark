"use client";

import { useEffect, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import {
	EquirectangularReflectionMapping,
	SRGBColorSpace,
	Texture,
	TextureLoader,
} from "three";
import { panoramaPlaceholderUrl, panoramaUrl } from "@/lib/r2";

function asEquirect(texture: Texture): Texture {
	texture.mapping = EquirectangularReflectionMapping;
	texture.colorSpace = SRGBColorSpace;
	return texture;
}

function PanoramaScene({
	index,
	onFullLoaded,
}: {
	index: number;
	onFullLoaded: (index: number) => void;
}) {
	const scene = useThree((state) => state.scene);

	useEffect(() => {
		let active = true;
		let fullLoaded = false;
		const loader = new TextureLoader();

		// Assign the new background, then free the texture it replaced so VRAM
		// doesn't grow as the user browses panoramas (placeholder -> full, and
		// across index changes).
		const apply = (texture: Texture) => {
			const previous = scene.background;
			scene.background = asEquirect(texture);
			if (previous instanceof Texture && previous !== texture) {
				previous.dispose();
			}
		};

		// Low-res placeholder shows first — it's a real equirect background, so
		// it pans in 360 immediately while the full image streams in.
		loader.load(panoramaPlaceholderUrl(index), (lqip) => {
			if (!active || fullLoaded) {
				lqip.dispose();
				return;
			}
			apply(lqip);
		});

		// Full resolution sharpens in place at the same orientation.
		loader.load(panoramaUrl(index), (full) => {
			if (!active) {
				full.dispose();
				return;
			}
			fullLoaded = true;
			apply(full);
			onFullLoaded(index);
		});

		return () => {
			active = false;
		};
	}, [index, scene, onFullLoaded]);

	return null;
}

export default function PanoramaImage({ index }: { index: number }) {
	const [loadedIndex, setLoadedIndex] = useState<number | null>(null);
	const sharp = loadedIndex === index;

	return (
		<div className="absolute inset-0">
			<Canvas camera={{ position: [0, 0, 0.1], fov: 75, near: 0.01, far: 1000 }}>
				<PanoramaScene index={index} onFullLoaded={setLoadedIndex} />
				<OrbitControls
					makeDefault
					enableZoom={false}
					enablePan={false}
					enableDamping
					dampingFactor={0.1}
					rotateSpeed={-0.3}
				/>
			</Canvas>
			{/* Smooths the low-res preview's blockiness, then fades out once the
			    full-resolution panorama has loaded. */}
			<div
				aria-hidden
				style={{
					backdropFilter: "blur(40px)",
					WebkitBackdropFilter: "blur(40px)",
				}}
				className={`pointer-events-none absolute inset-0 transition-opacity duration-700 ${
					sharp ? "opacity-0" : "opacity-100"
				}`}
			/>
		</div>
	);
}
