import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
	variable: "--font-geist-sans",
	subsets: ["latin"],
});

const geistMono = Geist_Mono({
	variable: "--font-geist-mono",
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
			className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
		>
			<body className="min-h-full">{children}</body>
		</html>
	);
}
