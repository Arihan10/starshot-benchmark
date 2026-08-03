import type { Metadata } from "next";
import { Anton, Archivo, IBM_Plex_Mono, Instrument_Serif } from "next/font/google";
import "./globals.css";

// FOUR FACES, AND EACH HAS ONE JOB. The interface is Archivo throughout, and the
// other three are reserved for a single thing each — the mark, the machine-set
// labels, and the prompt. A face used for two unrelated things stops being a
// signal, so none of them is.

// THE INTERFACE. Buttons, results, everything that is read as part of the page
// rather than as a piece of identity. Variable, so no `weight` is named — one
// file covers 500 through 900 and asking for named cuts would download more.
const archivo = Archivo({
	variable: "--font-archivo",
	subsets: ["latin"],
});

// THE MARK, and nothing else: one word, once, at the top left. Anton is a single
// heavy width with no other cuts to reach for, which is exactly why it can only
// be the wordmark — there is nothing to build a hierarchy out of.
const anton = Anton({
	variable: "--font-anton",
	subsets: ["latin"],
	weight: "400",
});

// THE MACHINE VOICE: the byline, the nav links, "WHO BUILT IT BETTER?", anything
// set in small tracked-out capitals. Monospace is what makes those read as
// labelling the page rather than speaking in it.
const plexMono = IBM_Plex_Mono({
	variable: "--font-plex-mono",
	subsets: ["latin"],
	weight: ["400", "500", "700"],
});

// THE PROMPT, italic, on the moon. The one human sentence on the page — someone
// asked for this scene in their own words — and the only place a serif appears.
const instrumentSerif = Instrument_Serif({
	variable: "--font-instrument-serif",
	subsets: ["latin"],
	weight: "400",
	style: ["normal", "italic"],
});

export const metadata: Metadata = {
	title: "SceneBench",
	description: "A spatial reasoning benchmark for LLMs.",
};

// The root shell: fonts, global stylesheet, black ground. Nothing else lives
// here — the scene-catalog provider the old viewer needed moved down into
// src/app/debug/layout.tsx, so the root stays free for the real site.
export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html
			lang="en"
			className={`${archivo.variable} ${anton.variable} ${plexMono.variable} ${instrumentSerif.variable} h-full antialiased`}
		>
			{/* `font-sans` here rather than a rule in globals.css: it resolves through
			    the same theme token every `font-sans` utility uses, so the document
			    default and the utilities can never disagree. */}
			<body className="min-h-full font-sans">{children}</body>
		</html>
	);
}
