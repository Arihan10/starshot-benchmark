import type { Metadata } from "next";
import { Instrument_Serif, Manrope, Public_Sans } from "next/font/google";
import ColorLab from "@/components/debug/ColorLab";
import FontLab from "@/components/debug/FontLab";
import MoonStage from "@/components/site/MoonStage";
import PageTransition from "@/components/site/PageTransition";
import { DEBUG_ENABLED } from "@/lib/flags";
import "./globals.css";

const manrope = Manrope({
	variable: "--font-manrope",
	subsets: ["latin"],
});

// THE WORDMARK'S FACE, and the only thing that uses `--font-display`.
//
// Loaded as the VARIABLE font — no `weight` — because the lockup wants a weight
// the rest of the site does not, and pinning a single static cut here would mean
// a second download the day anything else wants a different one. Anton, which
// this replaces, had no choice in the matter: it ships in one weight, which is
// part of why the wordmark's size had to be tuned by hand against it.
const publicSans = Public_Sans({
	variable: "--font-public-sans",
	subsets: ["latin"],
	display: "swap",
});

const instrumentSerif = Instrument_Serif({
	variable: "--font-instrument-serif",
	subsets: ["latin"],
	weight: "400",
	style: ["normal", "italic"],
});

export const metadata: Metadata = {
	title: "SceneBench",
	description: "A spatial reasoning benchmark for LLMs.",
	icons: {
		icon: "/icon.png",
	},
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html
			lang="en"
			className={`${manrope.variable} ${publicSans.variable} ${instrumentSerif.variable} h-full antialiased`}
		>
			<body className="min-h-full font-sans">
				<MoonStage />
				<PageTransition>{children}</PageTransition>
				{DEBUG_ENABLED && (
					<>
						<FontLab />
						<ColorLab />
					</>
				)}
			</body>
		</html>
	);
}
