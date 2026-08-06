
export const storey = (word: string, level: number) => `${word} ${level + 1}`;

export const Storey = (word: string, level: number) =>
	`${word.charAt(0).toUpperCase()}${word.slice(1)} ${level + 1}`;

export function prettyLabel(id: string): string {
	const words = id.replace(/[_-]+/g, " ").trim();
	return words ? words[0].toUpperCase() + words.slice(1) : id;
}
