"use client";

import Masthead, { Title, useTitleWidth } from "@/components/site/Masthead";

/**
 * Named-page masthead: flat title on the white trapezoid plate.
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
	const captionWidth = useTitleWidth(title, "name");

	return (
		<Masthead
			label={label}
			placement={placement}
			captionWidth={captionWidth}
		>
			<h1 className="m-0 w-full min-w-0 text-center">
				<Title voice="name">{title}</Title>
			</h1>
		</Masthead>
	);
}
