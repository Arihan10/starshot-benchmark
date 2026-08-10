"use client";

import CurvedPrompt, { usePromptChord } from "@/components/CurvedPrompt";
import Masthead from "@/components/site/Masthead";

/**
 * Masthead whose moon chord tracks a name-voice title, with the title set on
 * the limb arc — same construction as the arena prompt.
 */
export default function CurvedMasthead({
	label,
	title,
	placement = "flow",
}: {
	label: string;
	title: string;
	placement?: "overlay" | "flow";
}) {
	const chord = usePromptChord(title, "name");

	return (
		<Masthead label={label} placement={placement} chord={chord}>
			{({ moonRadius }) => (
				<h1 className="m-0 w-full min-w-0">
					<CurvedPrompt text={title} radius={moonRadius} voice="name" />
				</h1>
			)}
		</Masthead>
	);
}
