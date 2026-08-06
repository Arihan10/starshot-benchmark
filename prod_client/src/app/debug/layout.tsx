import { SceneProvider } from "@/components/SceneProvider";

export default function DebugLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return <SceneProvider>{children}</SceneProvider>;
}
