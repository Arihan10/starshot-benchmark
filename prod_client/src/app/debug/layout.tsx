import { notFound } from "next/navigation";
import { SceneProvider } from "@/components/SceneProvider";
import { DEBUG_ENABLED } from "@/lib/flags";

export default function DebugLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	if (!DEBUG_ENABLED) notFound();
	return <SceneProvider>{children}</SceneProvider>;
}
