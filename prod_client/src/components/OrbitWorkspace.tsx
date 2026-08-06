"use client";

import { useCallback, useState, type ReactNode } from "react";
import OrbitViewer from "@/components/OrbitViewer";
import { useScene } from "@/components/SceneProvider";
import { sceneId, type Scene } from "@/lib/scenes";

type Side = "left" | "right";

export default function OrbitWorkspace() {
	const { scenes, status, error } = useScene();
	const [leftId, setLeftId] = useState<string | null>(null);
	const [rightId, setRightId] = useState<string | null>(null);
	const [focused, setFocused] = useState<Side | null>(null);

	const onLeftFocus = useCallback(
		(f: boolean) =>
			setFocused((cur) => (f ? "left" : cur === "left" ? null : cur)),
		[],
	);
	const onRightFocus = useCallback(
		(f: boolean) =>
			setFocused((cur) => (f ? "right" : cur === "right" ? null : cur)),
		[],
	);

	if (status === "loading") return <Message>loading scenes…</Message>;
	if (status === "error")
		return <Message error>failed to load scenes: {error}</Message>;
	if (scenes.length === 0) return <Message>no scenes published yet</Message>;

	const left = scenes.find((s) => sceneId(s) === leftId) ?? scenes[0];
	const right =
		scenes.find((s) => sceneId(s) === rightId) ?? scenes[1] ?? scenes[0];

	return (
		<div className='flex min-h-0 w-full flex-1'>
			<Panel
				visible={focused !== "right"}
				bordered={focused === null}
				scene={left}
				scenes={scenes}
				showPicker={focused === null}
				onSelect={setLeftId}
				onFocusedChange={onLeftFocus}
			/>
			<Panel
				visible={focused !== "left"}
				bordered={false}
				scene={right}
				scenes={scenes}
				showPicker={focused === null}
				onSelect={setRightId}
				onFocusedChange={onRightFocus}
			/>
		</div>
	);
}

function Panel({
	visible,
	bordered,
	scene,
	scenes,
	showPicker,
	onSelect,
	onFocusedChange,
}: {
	visible: boolean;
	bordered: boolean;
	scene: Scene;
	scenes: Scene[];
	showPicker: boolean;
	onSelect: (id: string) => void;
	onFocusedChange: (focused: boolean) => void;
}) {
	return (
		<section
			className={`relative h-full min-w-0 ${visible ? "flex-1" : "hidden"} ${
				bordered ? "border-r border-mark-8" : ""
			}`}
		>
			<OrbitViewer scene={scene} onFocusedChange={onFocusedChange} />
			{showPicker && (
				<div className='absolute left-1/2 top-4 z-30 -translate-x-1/2'>
					<select
						aria-label='Select scene for this panel'
						value={sceneId(scene)}
						onChange={(e) => onSelect(e.target.value)}
						className='max-w-[40vw] rounded-md border border-mark-16 bg-ground/60 px-2 py-1 text-xs text-ink outline-none backdrop-blur transition hover:border-mark-40 focus:border-cyan-400'
					>
						{scenes.map((s) => (
							<option key={sceneId(s)} value={sceneId(s)}>
								{s.slot} · {s.model}
							</option>
						))}
					</select>
				</div>
			)}
		</section>
	);
}

function Message({
	children,
	error,
}: {
	children: ReactNode;
	error?: boolean;
}) {
	return (
		<div className='flex flex-1 items-center justify-center'>
			<span
				className={`text-sm ${error ? "text-red-400" : "text-ink-40"}`}
			>
				{children}
			</span>
		</div>
	);
}
