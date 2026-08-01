// The archived developer viewer, parked under /debug.
//
// Everything that used to be served at the root — the dollhouse view (/), the
// panorama browser (/panorama) and the A/B orbit workspace (/orbit) — now hangs
// off /debug/*, so the real SceneBench site can own the root. This whole folder
// is disposable: deleting src/app/debug/ removes every debug route and its
// chrome (ViewerHeader, ScenePicker) in one go. The pieces it borrows from
// src/components and src/lib (SceneProvider, SceneGate, SceneCanvas,
// PanoramaImage, OrbitViewer/OrbitWorkspace, lib/orbit/*) are the production
// viewer and stay.
//
// The scene catalog provider lives here rather than in the root layout: only
// these routes read the catalog, and it still persists across client-side
// navigation between them.
import { SceneProvider } from "@/components/SceneProvider";

export default function DebugLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return <SceneProvider>{children}</SceneProvider>;
}
