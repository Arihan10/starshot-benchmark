import OrbitWorkspace from "@/components/OrbitWorkspace";
import ViewerHeader from "../ViewerHeader";

export default function OrbitPage() {
	return (
		<main className="relative flex h-dvh flex-col overflow-hidden bg-ground text-ink">
			<ViewerHeader />
			<OrbitWorkspace />
		</main>
	);
}
