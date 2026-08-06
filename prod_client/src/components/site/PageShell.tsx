import type { ReactNode } from "react";

export default function PageShell({
	eyebrow,
	title,
	lede,
	masthead,
	footer,
	measure = "900px",
	children,
}: {
	eyebrow?: string;
	title?: string;
	masthead?: ReactNode;
	footer?: ReactNode;
	lede?: ReactNode;
	measure?: string;
	children?: ReactNode;
}) {
	return (
		<div className="flex min-h-dvh flex-col bg-ground">
			{masthead}

			<main
				className={`mx-auto w-full flex-1 px-lg pt-lg ${footer ? "" : "pb-lg"}`}
				style={{ maxWidth: measure }}
			>
				{title && (
					<header className="mb-xl">
						{eyebrow && (
							<p className="font-label text-2xs text-accent">{eyebrow}</p>
						)}
						<h1 className="mt-sm font-sans text-xl leading-[1.08] font-bold tracking-[-0.02em] text-ink text-balance">
							{title}
						</h1>
						{lede && (
							<div className="mt-md max-w-[62ch] font-sans text-sm leading-[1.65] text-ink-64">
								{lede}
							</div>
						)}
					</header>
				)}

				{children}
			</main>

			{footer && (
				<div className="sticky bottom-0 z-30">{footer}</div>
			)}
		</div>
	);
}
