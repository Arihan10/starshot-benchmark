import type { Metadata } from "next";
import { Anton, Archivo, IBM_Plex_Mono, Instrument_Serif } from "next/font/google";
import MoonStage from "@/components/site/MoonStage";
import PageTransition from "@/components/site/PageTransition";
import "./globals.css";

// FOUR FACES, AND EACH HAS ONE JOB. The interface is Archivo, the mark is Anton,
// the machine voice is a monospace and the prompt is a serif. A face used for two
// unrelated things stops being a signal, so none of them is.

// THE INTERFACE. Buttons, results, the leaderboard's names and figures —
// everything read as part of the page rather than as a piece of identity.
//
// ARCHIVO, back in place of Space Grotesk. The leaderboard is set in it end to
// end, and its heavy cuts are what the board is built out of: a rank at 900 and a
// model name at 800 have to hold their shape at 18px against a black ground, which
// is a job a 700-max grotesque cannot do. Its neutrality is the point here rather
// than the problem — the table is a lot of numbers in a column, and a face with
// opinions in the letterforms is a face arguing with the data.
//
// Variable, so no `weight` is named: one file covers 100 through 900 (and a WIDTH
// axis, left at its default — see Button, which no longer reaches for it).
const archivo = Archivo({
	variable: "--font-archivo",
	subsets: ["latin"],
});

// THE MACHINE VOICE: column heads, lab names, vote counts, anything set in small
// tracked-out capitals beside a figure. It is the register the benchmark reads in —
// a readout rather than a sentence — and a monospace is what makes the difference
// legible at 9px, where tracking alone only makes type look stretched.
//
// Not variable, so the three cuts it actually uses are named. Asking for more would
// download more.
const plexMono = IBM_Plex_Mono({
	variable: "--font-ibm-plex-mono",
	subsets: ["latin"],
	weight: ["400", "500", "700"],
});

// THE MARK, and nothing else: one word, once, at the top left. Anton is a
// condensed poster face with a single weight — narrow per em, so the wordmark
// rides at 2.4527x the byline to come out flush with it (see Navbar, where
// that number is measured rather than chosen).
const anton = Anton({
	variable: "--font-anton",
	subsets: ["latin"],
	// ONE WEIGHT, and that is the face. Anton ships a single cut, so there is no
	// `font-bold` to reach for — asking for one only invites a synthesised bold.
	weight: "400",
	display: "swap",
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
			className={`${archivo.variable} ${plexMono.variable} ${anton.variable} ${instrumentSerif.variable} h-full antialiased`}
		>
			{/* `font-sans` here rather than a rule in globals.css: it resolves through
			    the same theme token every `font-sans` utility uses, so the document
			    default and the utilities can never disagree. */}
			{/* PAGE TRANSITIONS ARE A SITE-WIDE FACT, so the one thing that knows a
			    navigation is under way lives at the root. It catches the click once
			    and holds the route change; the moon, the masthead's type and every
			    screen of content subscribe rather than each catching their own. */}
			<body className="min-h-full font-sans">
				{/* ONE MOON, FOR THE WHOLE SITE, FOR THE LIFE OF THE TAB. It lives here
				    because the root layout is the only thing a navigation does not tear
				    down — which is the entire reason it can travel between pages instead
				    of being rebuilt on each of them. Pages place a MoonAnchor; this
				    follows it. */}
				<MoonStage />
				<PageTransition>{children}</PageTransition>
			</body>
		</html>
	);
}
