# GitHub source

repo: Starshot-Labs/starshot-benchmark
branch: jace/prod-client
path: prod_client

## Last sync
date: 2026-08-12T22:30:00Z

### Updated in this project
- Leaderboard page recreated from jace/prod-client: standings table (sort, search, paged show-more, hover-to-paper rows, foot fade mask, geometric scrollbar) and the 3D voxel podium island (three.js port of Podium/podiumLayout/loose/sweep: intro build, pillar swell, challenger rise on row hover, draggable/throwable blocks with physics) inside the current trapezoid navbar shell
- Arena LEADERBOARD buttons now navigate to the leaderboard page; its ENTER THE ARENA exit bar returns to the arena

## Sync history
- 2026-08-11T21:46:54Z — font tweaks populated from google/fonts quant.csv (1,708 families); navbar/prompt-plate/bottom-bar trapezoid redesign
- 2026-08-06T00:17:57Z — Editorial direction built end-to-end; SOG splat assets + splat.js adapted
- 2026-08-05T23:30:55Z — initial read of prod_client on jace/prod-client; recreation + direction explorations

## Secondary sources
| Purpose | Repo files |
| --- | --- |
| Font tweak option lists | google/fonts@main:tags/all/quant.csv (parsed, not kept in project) |

## Screen map
| Project screen | Repo files |
| --- | --- |
| SceneBench Arena (Current).dc.html | prod_client/src/app/page.tsx, prod_client/src/app/globals.css, prod_client/src/app/layout.tsx, prod_client/src/components/site/Masthead.tsx, prod_client/src/components/site/Navbar.tsx, prod_client/src/components/site/MoonStage.tsx, prod_client/src/components/site/MoonAnchor.tsx, prod_client/src/components/Moon.tsx, prod_client/src/components/CurvedPrompt.tsx, prod_client/src/components/LogoMark.tsx, prod_client/src/components/arena/VoteBar.tsx, prod_client/src/components/arena/VoteButton.tsx, prod_client/src/components/arena/RevealCard.tsx, prod_client/src/components/arena/NextTimer.tsx, prod_client/src/components/arena/PctReadout.tsx, prod_client/src/components/arena/ScenePanel.tsx, prod_client/src/components/arena/shatter.ts, prod_client/src/components/arena/Composer.tsx, prod_client/src/components/arena/buildSequence.ts, prod_client/src/components/ui/Button.tsx, prod_client/src/lib/localScenes.ts |
| Design Directions.dc.html | same sources as above (restyled variations) |
| Arena — Editorial.dc.html | same arena sources + SceneBench spatial benchmark(1)/splat.js, client/public/assets/hotel-room-lod/{1_0,2_0}/* |
| SceneBench Arena — Seam.dc.html | same arena sources, restyled to the "Seam" direction (1a); splats via splat.js per-panel stages |
| SceneBench Arena — Totem.dc.html | same arena sources, restyled to the "Totem / Glass" direction (4d + 6b): full-bleed scenes, bottom-center totem (glass prompt, unified A/SKIP/B dock, composer) |
| SceneBench Arena — Z.dc.html | same arena sources, "Interlock / Z" direction (7d mirrored): left scene bottom-flush, right scene top-flush, Z-seam; top-left notch = lockup + hamburger + LEADERBOARD/GENERATE, bottom-right notch = prompt + ballot, composer floats bottom-left |
| SceneBench Leaderboard.dc.html | prod_client/src/app/leaderboard/page.tsx, prod_client/src/lib/leaderboard.ts, prod_client/src/components/leaderboard/{LeaderboardStage,StandingsTable,Podium,BrandMark,ExitBar}.tsx, prod_client/src/components/leaderboard/{podiumLayout,loose,sweep,markContrast}.ts, prod_client/src/components/site/{ScrollBox,VoxelSky,Fade}.tsx, prod_client/src/lib/ink.ts, prod_client/src/app/globals.css |
| podium.js | prod_client/src/components/leaderboard/{podiumLayout,loose,sweep}.ts + Podium.tsx (three.js port) |
| brandmarks.js | prod_client/src/components/leaderboard/BrandMark.tsx (tones; icons via @lobehub/icons-static-svg CDN) |
| splat.js | SceneBench spatial benchmark(1)/splat.js (adapted; engine playcanvas@2.20.6) |

## Notes
- prod_client exists ONLY on branch jace/prod-client (not on main).
- Design tokens: ground #000, ink/mark #ededed with 64/40/16/8 opacity ramp, surface #131313 / #262626, accent #c3cdff, accent-deep #6b4f9e, rise #35c46a, fall #e5342b, sweep stops #ededed → #e6f0ff → #d8dcf5 → #e6d5e6 → #fbdcd2.
- Fonts: Anton (wordmark only), Archivo (interface), IBM Plex Mono 400/500/700 (readouts), Instrument Serif italic (prompt/composer).
- Type scale: 2xs clamp(9,0.72vw,12) / xs clamp(11,.9vw,14) / sm clamp(13,1.1vw,17) / base clamp(16,1.45vw,23) / lg clamp(22,2.2vw,34) / xl clamp(34,3.6vw,56). Space: 2xs..xl clamp(3..34, .3vw..3.8vw, 5..58).
- font-label utility: Archivo 600, tracking .22em, uppercase.
- Moon: MOON_DIAMETER min(48vw,690px), hangs off top, bottom anchored to masthead band (BAND_PAD = .42*spacing-xl below navbar, lifted spacing-md), crater layer spins 260s. Label "WHO BUILT IT BETTER?" mono 9.5px tracking .24em black on disc. Prompt = curved SVG textPath, radius 480/1000 viewBox, ±33°, Instrument Serif italic, font fitted 17–54 viewBox units.
- Vote bar: 3 clip-path slabs (RAKE 13px), sides 4.3*xl wide (voting) / measured (revealed), middle 2.05*xl; segments overlap -13px; solid = #ededed bg + black 900 text + 1px edge layer + accent-sweep hover; SKIP = black trapezoid, border-y only. Expand 880ms cubic-bezier(.62,.02,.24,1).
- Reveal: cards (BUILT BY / model 800 caps; ELO / number), winner = solid mark + text-background, loser = bg-background text-ink-40 border-mark-8; content-swap 400ms. NextTimer: NEXT label + scaleX draining line, 3s. PctReadout: "VOTERS PICKED THIS" + count-up % at calc(text-xl*1.85) Archivo 900, bottom corners at bottom-[calc(xl*2.6)].
- Panel outcomes: won = inset glow (inset 0 0 0 1px mark/.85, 34px/.3, 90px/.16) + -translate-y-2; lost = flat rgb(0,0,0,.62) wash 540ms + shatter (24 clipped shards, corner sweep 380ms, projectile arcs, WAAPI).
- Build sequence: ray 400ms → beam opens 260ms → bar wipes down 360ms → composer 240ms → rule converges 300ms → seam tail 220ms. Seam = 3px solid mark down center, breaks around controls (--seam-break/--seam-under measured).
- Rounds (hard-coded): "A modern house" leftShare 44 — Gemini Flash elo 2091 vs Claude Opus elo 2108; "A super mario style platformer level" leftShare 61 — Claude Opus 2108 vs Gemini Pro 2143. Reveal → 3s countdown → prompt-roll-out 420ms / moon-cycle 80° / prompt-settle 1000ms.
- Composer: collapsed 290px/15.5px placeholder "want to generate your own?" typed at 42ms/char, caret 2px accent glow blink 1.1s steps(1); open min(620px,58vw)/31px "describe any scene…", spring cubic-bezier(.28,1.35,.42,1); suggestions: "a cliffside villa at dusk" / "a sky-island temple world" / "a neon noodle alley"; GENERATE solid standalone sweep at left-full.
- Navbar: lockup (LogoMark mask + SCENEBENCH Anton at 2.4527×lockup, tracking .08em + BY STARSHOT LABS label at lockup=text-2xs, mark-40) | About FAQ Arena quiet cap-start pair | right: Leaderboard ghost standalone + "Build one yourself" solid sweep standalone (ml-2xs). Active nav: 2px accent underline + 10px glow, undraws on hover. Wordmark hover: glow mask sweep 820ms + star twinkle. Logo hover: color terminator reveal + glint 820ms.
