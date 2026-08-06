import Link from "next/link";

export default function ExitBar() {
	return (
		<div className="relative overflow-hidden bg-mark">
			<Link
				href="/"
				className="group/exit relative block w-full py-md text-center font-sans text-sm font-black tracking-[0.07em] uppercase text-ground"
			>
				<span
					aria-hidden
					className="absolute inset-0 opacity-0 transition-opacity duration-[420ms] ease-out group-hover/exit:opacity-100"
					style={{ backgroundImage: "var(--accent-sweep)" }}
				/>
				<span className="relative">
					Enter the arena
					<span
						aria-hidden
						className="ml-sm inline-block transition-transform duration-quick group-hover/exit:translate-x-1"
					>
						→
					</span>
				</span>
			</Link>
		</div>
	);
}
