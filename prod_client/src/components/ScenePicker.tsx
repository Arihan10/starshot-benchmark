"use client";

import { useScene } from "@/components/SceneProvider";
import { sceneId, type Scene } from "@/lib/scenes";

// Scene selector for the header. Scenes are grouped by run (a benchmark run
// holds many slot/model/version takes), so the dropdown scales as more runs and
// versions are published.
export default function ScenePicker() {
	const { scenes, selected, status, select } = useScene();

	if (status === "loading") {
		return <span className="text-xs text-neutral-500">loading scenes…</span>;
	}
	if (status === "error" || scenes.length === 0) {
		return <span className="text-xs text-neutral-500">no scenes published</span>;
	}

	const groups = groupByRun(scenes);

	return (
		<select
			aria-label="Select scene"
			value={selected ? sceneId(selected) : ""}
			onChange={(e) => select(e.target.value)}
			className="max-w-[60vw] rounded-md border border-white/15 bg-black/40 px-2 py-1 text-xs text-neutral-200 outline-none transition hover:border-white/30 focus:border-cyan-400"
		>
			{groups.map(([run, runScenes]) => (
				<optgroup key={run} label={run} className="bg-neutral-900">
					{runScenes.map((s) => (
						<option key={sceneId(s)} value={sceneId(s)}>
							{s.slot} · {s.model} · v{s.version}
						</option>
					))}
				</optgroup>
			))}
		</select>
	);
}

function groupByRun(scenes: Scene[]): [string, Scene[]][] {
	const byRun = new Map<string, Scene[]>();
	for (const s of scenes) {
		const list = byRun.get(s.run);
		if (list) list.push(s);
		else byRun.set(s.run, [s]);
	}
	return [...byRun.entries()];
}
