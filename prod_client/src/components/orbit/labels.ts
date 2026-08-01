// Text the viewer chrome writes about a scene. Kept apart from the components
// that print it because several of them print the same words, and the rules are
// about the SCENE's vocabulary rather than about any one panel.

// "floor 2" / "deck 2" / "section 2" — the storey word comes from the capture
// profile (see anchors.py `choose_profile`), so a ship stops being described as
// having floors. `Storey` for the places a sentence starts with it.
export const storey = (word: string, level: number) => `${word} ${level + 1}`;

export const Storey = (word: string, level: number) =>
	`${word.charAt(0).toUpperCase()}${word.slice(1)} ${level + 1}`;

// `antique_display_sextant` → "Antique display sextant". The ids are authored by
// the pipeline, so they read as words already — they just need unpicking.
export function prettyLabel(id: string): string {
	const words = id.replace(/[_-]+/g, " ").trim();
	return words ? words[0].toUpperCase() + words.slice(1) : id;
}
