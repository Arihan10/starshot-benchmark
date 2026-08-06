"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { fetchScenes, sceneId, type Scene } from "@/lib/scenes";

type Status = "loading" | "ready" | "error";

type SceneContextValue = {
	scenes: Scene[];
	selected: Scene | null;
	status: Status;
	error: string | null;
	select: (id: string) => void;
};

const SceneContext = createContext<SceneContextValue | null>(null);

export function SceneProvider({ children }: { children: ReactNode }) {
	const [scenes, setScenes] = useState<Scene[]>([]);
	const [selectedId, setSelectedId] = useState<string | null>(null);
	const [status, setStatus] = useState<Status>("loading");
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let active = true;
		fetchScenes()
			.then((list) => {
				if (!active) return;
				setScenes(list);
				setSelectedId(list.length ? sceneId(list[0]) : null);
				setStatus("ready");
			})
			.catch((e: unknown) => {
				if (!active) return;
				setError(e instanceof Error ? e.message : "failed to load scenes");
				setStatus("error");
			});
		return () => {
			active = false;
		};
	}, []);

	const value = useMemo<SceneContextValue>(
		() => ({
			scenes,
			selected: scenes.find((s) => sceneId(s) === selectedId) ?? null,
			status,
			error,
			select: setSelectedId,
		}),
		[scenes, selectedId, status, error],
	);

	return <SceneContext.Provider value={value}>{children}</SceneContext.Provider>;
}

export function useScene(): SceneContextValue {
	const ctx = useContext(SceneContext);
	if (!ctx) throw new Error("useScene must be used within <SceneProvider>");
	return ctx;
}
