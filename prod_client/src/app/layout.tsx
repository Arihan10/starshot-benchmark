import type { Metadata } from "next";
import { Anton, Instrument_Serif, Manrope } from "next/font/google";
import ColorLab from "@/components/debug/ColorLab";
import FontLab from "@/components/debug/FontLab";
import MoonStage from "@/components/site/MoonStage";
import PageTransition from "@/components/site/PageTransition";
import "./globals.css";

const manrope = Manrope({
	variable: "--font-manrope",
	subsets: ["latin"],
});

const anton = Anton({
	variable: "--font-anton",
	subsets: ["latin"],
	weight: "400",
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
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html
			lang="en"
			className={`${manrope.variable} ${anton.variable} ${instrumentSerif.variable} h-full antialiased`}
		>
			<body className="min-h-full font-sans">
				<MoonStage />
				<PageTransition>{children}</PageTransition>
				{process.env.NODE_ENV === "development" && (
					<>
						<FontLab />
						<ColorLab />
					</>
				)}
			</body>
		</html>
	);
}
