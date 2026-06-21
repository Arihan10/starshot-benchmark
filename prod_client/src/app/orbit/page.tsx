import OrbitWorkspace from "@/components/OrbitWorkspace";
import ViewerHeader from "@/components/ViewerHeader";

export default function OrbitPage() {
	return (
		<main className="relative flex h-dvh flex-col overflow-hidden bg-neutral-950 text-neutral-100">
			<ViewerHeader />
			<OrbitWorkspace />
		</main>
	);
}
