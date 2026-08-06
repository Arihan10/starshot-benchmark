"use client";

const EXPAND = [
	"M8 3H5a2 2 0 0 0-2 2v3",
	"M21 8V5a2 2 0 0 0-2-2h-3",
	"M3 16v3a2 2 0 0 0 2 2h3",
	"M16 21h3a2 2 0 0 0 2-2v-3",
];

const COLLAPSE = [
	"M8 3v3a2 2 0 0 1-2 2H3",
	"M21 8h-3a2 2 0 0 1-2-2V3",
	"M3 16h3a2 2 0 0 1 2 2v3",
	"M16 21v-3a2 2 0 0 1 2-2h3",
];

const GLASS = ["M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14", "M16.2 16.2 21 21"];

const KEY =
	"group/key relative flex size-[clamp(30px,2.4vw,38px)] cursor-pointer items-center justify-center text-ink-40 transition-colors duration-quick hover:text-ink focus-visible:outline-none focus-visible:text-ink disabled:cursor-default disabled:text-ink-8 disabled:hover:text-ink-8";

function Key({
	label,
	disabled,
	onPress,
	children,
}: {
	label: string;
	disabled?: boolean;
	onPress: () => void;
	children: React.ReactNode;
}) {
	return (
		<button
			type="button"
			onClick={onPress}
			disabled={disabled}
			title={label}
			aria-label={label}
			className={KEY}
		>
			<svg
				viewBox="0 0 24 24"
				aria-hidden
				className="size-[54%]"
				fill="none"
				stroke="currentColor"
				strokeWidth={2}
				strokeLinecap="round"
				strokeLinejoin="round"
			>
				{children}
			</svg>
		</button>
	);
}

export default function ViewerControls({
	align,
	zoom,
	isFullscreen,
	fullscreen,
	onZoom,
	onFullscreen,
}: {
	align: "left" | "right";
	zoom: { in: boolean; out: boolean };
	isFullscreen: boolean;
	fullscreen: boolean;
	onZoom: (step: number) => void;
	onFullscreen: () => void;
}) {
	return (
		<div
			className={`pointer-events-auto absolute bottom-sm z-7 flex flex-col items-center gap-0 ${
				align === "left" ? "left-sm" : "right-sm"
			}`}
		>
			{fullscreen && (
				<Key
					label={isFullscreen ? "Exit full screen" : "Full screen"}
					onPress={onFullscreen}
				>
					{(isFullscreen ? COLLAPSE : EXPAND).map((d) => (
						<path key={d} d={d} />
					))}
				</Key>
			)}

			<Key label="Zoom in" disabled={!zoom.in} onPress={() => onZoom(1)}>
				{GLASS.map((d) => (
					<path key={d} d={d} />
				))}
				<path d="M8.4 11h5.2" />
				<path d="M11 8.4v5.2" />
			</Key>

			<Key label="Zoom out" disabled={!zoom.out} onPress={() => onZoom(-1)}>
				{GLASS.map((d) => (
					<path key={d} d={d} />
				))}
				<path d="M8.4 11h5.2" />
			</Key>
		</div>
	);
}
