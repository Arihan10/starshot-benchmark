import * as THREE from "three";

/**
 * THE SITE'S ACCENT GRADIENT, READ OUT OF THE STYLESHEET AND PAINTED ONTO CUBES.
 *
 * The buttons light up in it on hover, the wordmark is filled with it, and the
 * podium's three plinths are made of it. The first two are CSS and get it for
 * free. The third is WebGL, where there is no background-image to set — so the
 * ramp is evaluated in a shader, per fragment, and every plinth cube carries the
 * whole gradient across its own faces exactly as a button carries it across its
 * own width.
 *
 * PER FRAGMENT, NOT PER CUBE, and that distinction is the whole of this file.
 * Handing each cube one flat colour sampled along a sweep is the cheap version,
 * and it is a different picture: twenty-seven solid tiles stepping through the
 * ramp, which reads as three coloured places rather than as one material. A
 * gradient is a thing a surface HAS. Given to each cube, the plinths are made of
 * the accent; given to each place, the accent is a label for the place.
 *
 * IT READS THE STOPS RATHER THAN REPEATING THEM. Copying five hex values out of
 * `globals.css` would work today and be wrong the first time someone warms the
 * accent by a few percent — the buttons would move and the podium would not, and
 * nothing would fail loudly enough to notice. `--sweep-0` … `--sweep-4` are
 * declared once there and composed into `--accent-sweep`; this reads the same
 * declarations, so there is one ramp and two renderers.
 */

/** Where each stop sits along the ramp, matching the percentages in the gradient.
 *  The last is the first again — the sweep returns to plain ink, which is what
 *  makes it read as light crossing a white surface rather than as a rainbow. */
const AT = [0, 0.24, 0.46, 0.68, 0.88, 1];
const NAMES = [
	"--sweep-0",
	"--sweep-1",
	"--sweep-2",
	"--sweep-3",
	"--sweep-4",
	"--sweep-0",
];

/**
 * THE DIRECTION THE RAMP RUNS, IN THE WORLD — derived from the direction it runs
 * on the SCREEN, because that is the only place a gradient angle means anything.
 *
 * CSS measures a gradient angle clockwise from up, so this one is (sin, −cos) in
 * a y-down system: just past horizontal, tipping gently down to the right. Two
 * world directions are enough to hit it, and they are the two the whole
 * composition is planned in — the screen's horizontal (1, 0, −1), which projects
 * to (2·cos45, 0), and world up, which projects to (0, −cos35.264). Solving
 * a·(2·cos45) = sinθ and −b·cos35.264 = −cosθ gives the mix of the two that lands
 * on the gradient's own angle, and it is exact rather than eyeballed.
 *
 * The cube's LOCAL position is what gets projected onto this, so the gradient is
 * paint: it turns with a cube that is thrown and tumbling, rather than sliding
 * across it like a reflection.
 */
const ANGLE = (104 * Math.PI) / 180;
const ACROSS = 0.7071;
const RISE = 0.8165;
export const SWEEP_AXIS = new THREE.Vector3(1, 0, -1)
	.multiplyScalar(Math.sin(ANGLE) / (2 * ACROSS))
	.addScaledVector(new THREE.Vector3(0, 1, 0), Math.cos(ANGLE) / RISE)
	.normalize();

/** How far a unit cube reaches along that axis, corner to corner — the distance
 *  the ramp has to cover to arrive at both ends of every cube. */
export const SWEEP_SPAN =
	Math.abs(SWEEP_AXIS.x) + Math.abs(SWEEP_AXIS.y) + Math.abs(SWEEP_AXIS.z);

/**
 * Resolve the five stops to colours, through the browser.
 *
 * VIA A PROBE ELEMENT, because `getComputedStyle` hands back a custom property
 * almost exactly as it was written — `--sweep-0` comes out as the literal
 * `rgb(237 237 237)`, space-separated CSS Color 4, which three's parser does not
 * accept. Assigning it to a real `color` and reading THAT back makes the browser
 * do the resolving, and the answer is always comma-separated `rgb(r, g, b)`
 * whatever the stop was written as. Hex, `color-mix`, a channel token — all one
 * code path, and no colour parser of our own.
 */
export function sweepStops(): THREE.Color[] {
	const probe = document.createElement("span");
	probe.style.display = "none";
	document.body.appendChild(probe);
	try {
		return NAMES.map((name) => {
			probe.style.color = `var(${name})`;
			// `setStyle` treats the value as sRGB and converts it into the renderer's
			// working space, which is linear — the space the shader below has to mix
			// in, and the space a colour-managed CSS gradient is interpolated in too.
			// Mixing in sRGB instead is what gives gradients that grey sag through
			// the middle.
			return new THREE.Color().setStyle(getComputedStyle(probe).color);
		});
	} finally {
		probe.remove();
	}
}

/** The name of the per-instance attribute that says whether a cube is a plinth. */
export const PLINTH_ATTRIBUTE = "aPlinth";

/**
 * Teach a standard material to paint the sweep on the instances that ask for it.
 *
 * PATCHED RATHER THAN WRITTEN FROM SCRATCH. These cubes are lit — they have the
 * same roughness and take the same light as the nine hundred white ones around
 * them, and a plinth that ignored the lighting would look like a sticker of a
 * gradient rather than like stone that happens to be coloured. Everything that
 * makes that work is already in `MeshStandardMaterial`; the only thing it does
 * not know is where the diffuse colour comes from, so that is the only thing
 * replaced.
 *
 * MASKED PER INSTANCE, because the plinths share their mesh with the whole city.
 * Six hundred cubes are one instanced draw and splitting twenty-seven of them out
 * would cost a second mesh, a second material, and a second copy of every path
 * that animates or throws them. An attribute of one float is cheaper than all of
 * that, and it means a plinth is decided by data rather than by which mesh
 * something got put in.
 *
 * The stops arrive later, from `sweepStops`, which needs a document — so the
 * uniform is created here and filled in on mount.
 */
export function paintSweep(material: THREE.Material): {
	stops: { value: THREE.Color[] };
} {
	const stops = { value: AT.map(() => new THREE.Color(1, 1, 1)) };
	const at = { value: AT.slice() };

	material.onBeforeCompile = (shader) => {
		shader.uniforms.uSweepStops = stops;
		shader.uniforms.uSweepAt = at;

		shader.vertexShader = shader.vertexShader
			.replace(
				"#include <common>",
				`#include <common>
				attribute float ${PLINTH_ATTRIBUTE};
				varying float vPlinth;
				varying float vSweep;`,
			)
			// `position` is the cube's own untransformed corner, so the ramp is fixed
			// to the cube: unchanged by the scale it grows through on arrival, and
			// carried around with it when it is picked up and thrown.
			.replace(
				"#include <begin_vertex>",
				`#include <begin_vertex>
				vPlinth = ${PLINTH_ATTRIBUTE};
				vSweep = clamp(
					dot(position, vec3(${SWEEP_AXIS.x}, ${SWEEP_AXIS.y}, ${SWEEP_AXIS.z}))
						/ ${SWEEP_SPAN} + 0.5,
					0.0, 1.0
				);`,
			);

		shader.fragmentShader = shader.fragmentShader
			.replace(
				"#include <common>",
				`#include <common>
				uniform vec3 uSweepStops[${AT.length}];
				uniform float uSweepAt[${AT.length}];
				varying float vPlinth;
				varying float vSweep;

				// Every segment, mixed in order. Segments already passed clamp to 1 and
				// overwrite; segments not yet reached clamp to 0 and leave it alone; the
				// one the fragment is inside contributes its fraction. Which is
				// piecewise-linear interpolation without a branch, and a branch on a
				// value that varies across a triangle is the thing worth avoiding.
				vec3 sweepAt(float t) {
					vec3 c = uSweepStops[0];
					for (int i = 1; i < ${AT.length}; i++) {
						c = mix(c, uSweepStops[i], clamp(
							(t - uSweepAt[i - 1]) / max(uSweepAt[i] - uSweepAt[i - 1], 1e-5),
							0.0, 1.0
						));
					}
					return c;
				}`,
			)
			// AFTER the chunk that applies vertex and instance colour, so this is the
			// last word on what the surface is — and before every lighting chunk, so
			// it is lit like everything else.
			.replace(
				"#include <color_fragment>",
				`#include <color_fragment>
				diffuseColor.rgb = mix(diffuseColor.rgb, sweepAt(vSweep), vPlinth);`,
			);
	};

	return { stops };
}
