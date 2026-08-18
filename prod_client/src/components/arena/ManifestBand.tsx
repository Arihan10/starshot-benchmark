"use client";

import { useState, type Ref } from "react";

const STEPS = [
	["AN LLM ORCHESTRATES", "THE DIFFUSION"],
	["STARSHOT’S MODEL BUILDS", "THE GEOMETRY — TWICE"],
	["TWO MODELS RACE —", "YOU JUDGE THEM"],
] as const;

const HINT = "describe any scene (e.g. a modern house)";

export default function ManifestBand({
	inputRef,
}: {
	inputRef: Ref<HTMLInputElement>;
}) {
	const [draft, setDraft] = useState("");
	const filled = draft.trim().length > 0;

	return (
		<div className="arena-band arena-chrome">
			<form
				className="arena-band__cell arena-composer"
				data-draft={filled}
				onSubmit={(event) => event.preventDefault()}
			>
				<label htmlFor="arena-composer" className="arena-composer__label">
					<span>0 · PROMPT</span>
					<span aria-hidden={!filled}>PRESS ENTER ↵</span>
				</label>

				<span aria-hidden className="arena-composer__divider" />

				<span className="arena-composer__field">
					<input
						id="arena-composer"
						ref={inputRef}
						type="text"
						className="arena-composer__input"
						placeholder={HINT}
						value={draft}
						onChange={(event) => setDraft(event.target.value)}
					/>
					<span aria-hidden className="arena-composer__rule" />
				</span>

				<button
					type="submit"
					className="arena-composer__submit"
					aria-hidden={!filled}
					tabIndex={filled ? 0 : -1}
				>
					↵
				</button>
			</form>

			{STEPS.map(([head, tail], index) => (
				<div key={head} className="arena-band__cell arena-band__step">
					<span className="arena-band__number">{index + 1}</span>
					<span className="arena-band__line">
						{head}
						<br />
						{tail}
					</span>
				</div>
			))}
		</div>
	);
}
