"use client";

import CurvedPrompt, { useCaptionWidth } from "@/components/CurvedPrompt";
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
	const captionWidth = useCaptionWidth(title, "name");

	return (
		<Masthead
			label={label}
			placement={placement}
			captionWidth={captionWidth}
		>
			{({ promptRadius }) => (
				<h1 className="m-0 w-full min-w-0">
					<CurvedPrompt text={title} radius={promptRadius} voice="name" />
				</h1>
			)}
		</Masthead>
	);
}
