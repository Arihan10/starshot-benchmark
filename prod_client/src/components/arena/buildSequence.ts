export type BuildPart =
	| "ray"
	| "beam"
	| "wipe"
	| "composer"
	| "rule";

const RAY = 400;

const OPEN = 260;

const REVEAL = 360;

const COMPOSER = 240;

const CRULE = 300;

export const BEAM_MS = OPEN + REVEAL;

export const BEAM_OPEN_PCT = (OPEN / BEAM_MS) * 100;

export const REVEAL_EASING = "cubic-bezier(0.22, 0.75, 0.3, 1)";

const OUT = 0.55;

const AT_RAY = 0;
const AT_BEAM = AT_RAY + RAY;
const AT_WIPE = AT_BEAM + OPEN;
const AT_COMPOSER = AT_WIPE + REVEAL;
const AT_RULE = AT_COMPOSER + COMPOSER;

const BACK_WIPE = 0;
const BACK_BEAM = 0;
const BACK_RAY = BACK_BEAM + BEAM_MS * OUT;

const BACK_RULE = 0;
const BACK_COMPOSER = BACK_RULE + CRULE * OUT;

const SEQUENCE: Record<BuildPart, { at: number; back: number; dur: number }> = {
	ray: { at: AT_RAY, back: BACK_RAY, dur: RAY },
	beam: { at: AT_BEAM, back: BACK_BEAM, dur: BEAM_MS },
	wipe: { at: AT_WIPE, back: BACK_WIPE, dur: REVEAL },
	composer: { at: AT_COMPOSER, back: BACK_COMPOSER, dur: COMPOSER },
	rule: { at: AT_RULE, back: BACK_RULE, dur: CRULE },
};

export function buildStep(part: BuildPart, building: boolean) {
	const { at, back, dur } = SEQUENCE[part];
	return {
		transitionDuration: `${Math.round(building ? dur : dur * OUT)}ms`,
		transitionDelay: `${Math.round(building ? at : back)}ms`,
		transitionTimingFunction:
			part === "wipe"
				? REVEAL_EASING
				: building
					? "cubic-bezier(0.16, 0.84, 0.28, 1)"
					: "cubic-bezier(0.5, 0, 0.75, 0.35)",
	} as const;
}

export const BUILD_MS = AT_RULE + CRULE;
