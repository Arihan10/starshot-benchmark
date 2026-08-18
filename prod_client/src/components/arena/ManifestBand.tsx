"use client";

import { useState, type Ref } from "react";

const STEPS = [
	["AN LLM ORCHESTRATES", "THE DIFFUSION"],
	["STARSHOT’S MODEL BUILDS", "THE GEOMETRY — TWICE"],
	["TWO MODELS RACE —", "YOU JUDGE THEM"],
] as const;

const HINT = "describe any scene…";

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
					0 · PROMPT
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
					<span aria-hidden className="arena-composer__rule arena-composer__rule--lit" />
				</span>

				<span className="arena-glow arena-composer__go">
					<button
						type="submit"
						className="arena-key arena-key--solid arena-composer__submit"
					>
						GO!
					</button>
				</span>
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
