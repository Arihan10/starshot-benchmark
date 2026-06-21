import {
	type Mesh,
	type Object3D,
	type PerspectiveCamera,
	Raycaster,
	type Scene,
	Vector2,
} from "three";
import { OutlinePass } from "three/examples/jsm/postprocessing/OutlinePass.js";
import type { OrbitState } from "./types";

const _ndc = new Vector2();

// Proxy parts whose name calls out a door or a stair are pinned to the orange
// outline as a permanent wayfinding cue (see ObjectAddressing.pinOutlinesByName).
const PINNED_OUTLINE_RE = /door|stair/i;

// The addressable objects of a loaded root. The exporter/loader can wrap the
// real objects under a single node, so unwrap single unnamed non-mesh wrappers,
// then take that container's mesh-bearing children. Fall back to "every mesh"
// when the structure is flat or collapses to one node.
export function collectObjects(root: Object3D): Object3D[] {
	let container = root;
	while (
		container.children.length === 1 &&
		!(container.children[0] as Mesh).isMesh &&
		!container.children[0].name &&
		container.children[0].children.length > 0
	) {
		container = container.children[0];
	}
	const hasMesh = (o: Object3D) => {
		let found = false;
		o.traverse((c) => {
			if ((c as Mesh).isMesh) found = true;
		});
		return found;
	};
	let objs = container.children.filter(hasMesh);
	if (objs.length <= 1) {
		objs = [];
		root.traverse((o) => {
			if ((o as Mesh).isMesh) objs.push(o);
		});
	}
	return objs;
}

// Per-object addressing for a loaded root (lite + proxy): hover (cyan), hide,
// and persistent outline (orange), plus the right-click menu state. The two
// OutlinePasses live here and are handed to the engine's composer; the engine
// owns emit() and decides which root is addressable (passed into pick/open).
export class ObjectAddressing {
	private readonly picker = new Raycaster();
	private readonly hiddenObjects = new Set<Object3D>();
	private readonly outlinedObjects = new Set<Object3D>();
	// Objects pinned to the orange outline regardless of hover / right-click —
	// populated by pinOutlinesByName, dropped only by reset().
	private readonly pinnedOutlines = new Set<Object3D>();
	private hoveredObj: Object3D | null = null;
	private menuTarget: Object3D | null = null;
	private contextMenu: OrbitState["contextMenu"] = null;

	readonly selectPass: OutlinePass;
	readonly hoverPass: OutlinePass;

	constructor(
		scene: Scene,
		private readonly camera: PerspectiveCamera,
		private readonly canvas: HTMLCanvasElement,
	) {
		this.selectPass = makeOutlinePass(scene, camera, 0xffa23a, 4, 2);
		this.hoverPass = makeOutlinePass(scene, camera, 0x66e0ff, 3, 1.5);
	}

	register(root: Object3D) {
		collectObjects(root).forEach((o, i) => {
			o.userData.objId = i;
			o.userData.objLabel =
				o.name && o.name.trim() ? o.name.trim() : `object ${i + 1}`;
		});
	}

	// Permanently orange-outline every object in `root` whose label names a door
	// or a stair. These outlines ignore the hover toggle and survive
	// clearOutlines / the right-click menu — only reset() drops them. Call after
	// register() so objLabel is populated.
	pinOutlinesByName(root: Object3D) {
		collectObjects(root).forEach((o) => {
			if (PINNED_OUTLINE_RE.test((o.userData.objLabel as string) ?? ""))
				this.pinnedOutlines.add(o);
		});
	}

	get hoveredObject(): Object3D | null {
		return this.hoveredObj;
	}
	get hoverLabel(): string | null {
		return this.hoveredObj
			? (this.hoveredObj.userData.objLabel as string)
			: null;
	}
	get menu(): OrbitState["contextMenu"] {
		return this.contextMenu;
	}
	get hasMenu(): boolean {
		return !!this.contextMenu;
	}

	setHover(obj: Object3D | null): boolean {
		if (obj === this.hoveredObj) return false;
		this.hoveredObj = obj;
		return true;
	}

	// Nearest visible addressable object under a screen point within `root`.
	// Raycaster doesn't skip invisible objects, so skip hidden ones explicitly —
	// otherwise a hidden object would still shadow the geometry behind it.
	pickAt(
		clientX: number,
		clientY: number,
		root: Object3D | null,
	): Object3D | null {
		if (!root || !root.visible) return null;
		const rect = this.canvas.getBoundingClientRect();
		_ndc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		this.picker.setFromCamera(_ndc, this.camera);
		for (const h of this.picker.intersectObject(root, true)) {
			const obj = this.findObjectRoot(h.object, root);
			if (obj && obj.visible) return obj;
		}
		return null;
	}

	// Feed the live hover / selection sets to the two outline passes (run every
	// frame from tick). Selection (orange) wins over hover (cyan) for an object
	// that's both, and hidden objects are dropped; an empty list is a no-op pass.
	updateOutlines() {
		const selected: Object3D[] = [];
		for (const o of this.pinnedOutlines) if (o.visible) selected.push(o);
		for (const o of this.outlinedObjects)
			if (o.visible && !this.pinnedOutlines.has(o)) selected.push(o);
		this.selectPass.selectedObjects = selected;
		const h = this.hoveredObj;
		this.hoverPass.selectedObjects =
			h &&
			h.visible &&
			!this.outlinedObjects.has(h) &&
			!this.pinnedOutlines.has(h)
				? [h]
				: [];
	}

	// Right-click an object → per-object menu (hide / outline). Right-clicking
	// empty space still surfaces the recovery actions (you can't re-pick a hidden
	// object), but only if there's something to recover.
	openMenu(
		clientX: number,
		clientY: number,
		root: Object3D | null,
	) {
		const obj = this.pickAt(clientX, clientY, root);
		if (
			!obj &&
			this.hiddenObjects.size === 0 &&
			this.outlinedObjects.size === 0
		) {
			this.closeMenu();
			return;
		}
		this.menuTarget = obj;
		this.contextMenu = {
			x: clientX,
			y: clientY,
			label: obj ? (obj.userData.objLabel as string) : null,
			hidden: !!obj && this.hiddenObjects.has(obj),
			outlined: !!obj && this.outlinedObjects.has(obj),
			hiddenCount: this.hiddenObjects.size,
			outlinedCount: this.outlinedObjects.size,
		};
	}

	closeMenu() {
		this.contextMenu = null;
		this.menuTarget = null;
	}

	toggleMenuTargetHidden() {
		const obj = this.menuTarget;
		if (obj) this.setHidden(obj, !this.hiddenObjects.has(obj));
		this.closeMenu();
	}

	toggleMenuTargetOutline() {
		const obj = this.menuTarget;
		if (obj) this.toggleOutline(obj);
		this.closeMenu();
	}

	showAllHidden() {
		for (const o of this.hiddenObjects) o.visible = true;
		this.hiddenObjects.clear();
		this.closeMenu();
	}

	clearOutlines() {
		this.outlinedObjects.clear();
		this.closeMenu();
	}

	// Drop all per-object state when the scene is torn down; the nodes themselves
	// are disposed with their roots.
	reset() {
		this.hiddenObjects.clear();
		this.outlinedObjects.clear();
		this.pinnedOutlines.clear();
		this.hoveredObj = null;
		this.menuTarget = null;
		this.contextMenu = null;
	}

	private setHidden(obj: Object3D, hidden: boolean) {
		obj.visible = !hidden;
		if (hidden) {
			this.hiddenObjects.add(obj);
			if (obj === this.hoveredObj) this.hoveredObj = null; // can't hover what's gone
		} else {
			this.hiddenObjects.delete(obj);
		}
	}

	private toggleOutline(obj: Object3D) {
		if (this.outlinedObjects.has(obj)) this.outlinedObjects.delete(obj);
		else this.outlinedObjects.add(obj);
	}

	private findObjectRoot(node: Object3D, root: Object3D): Object3D | null {
		let cur: Object3D | null = node;
		while (cur && cur !== root) {
			if (cur.userData.objId !== undefined) return cur;
			cur = cur.parent;
		}
		return null;
	}
}

// Build an OutlinePass tuned to a colour. hiddenEdgeColor is black so the
// occluded part of a silhouette adds nothing (the overlay blends additively) —
// the outline shows only where the object is actually visible, reading as an
// in-place highlight rather than an x-ray.
function makeOutlinePass(
	scene: Scene,
	camera: PerspectiveCamera,
	color: number,
	edgeStrength: number,
	edgeThickness: number,
): OutlinePass {
	const pass = new OutlinePass(new Vector2(1, 1), scene, camera);
	pass.visibleEdgeColor.set(color);
	pass.hiddenEdgeColor.set(0x000000);
	pass.edgeStrength = edgeStrength;
	pass.edgeThickness = edgeThickness;
	return pass;
}
