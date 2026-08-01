import type { Metadata } from "next";
import { Archivo, Figtree } from "next/font/google";
import "./globals.css";

// TWO FAMILIES, AND ONLY TWO. Archivo sets anything that is meant to be read as
// a title; Figtree sets everything else. Neither takes a `weight` because both
// ship as variable fonts — one file covers the whole range, and asking for named
// cuts would download more, not less.

// The display face: headings and the wordmark, never body copy.
const archivo = Archivo({
	variable: "--font-archivo",
	subsets: ["latin"],
});

// The text face, and the document default.
const figtree = Figtree({
	variable: "--font-figtree",
	subsets: ["latin"],
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
			className={`${archivo.variable} ${figtree.variable} h-full antialiased`}
		>
			{/* `font-sans` here rather than a rule in globals.css: it resolves through
			    the same theme token every `font-sans` utility uses, so the document
			    default and the utilities can never disagree. */}
			<body className="min-h-full font-sans">{children}</body>
		</html>
	);
}
