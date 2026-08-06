# Prompt versions

Every subfolder of `versions/` is one **prompt version**. The folder name is the
version name. A version is a complete set of prompt templates for the pipeline —
when you create a run in the dashboard you pick exactly one version, and its
folder is copied verbatim into `runs/<run>/prompts/` as that run's **snapshot**.
The run renders its prompts from the snapshot: editing a source version
afterwards never changes a started run, and a run survives server restarts and
resumes with its own snapshot. The one deliberate exception is the prompt
lab's "apply to run", which writes edits INTO the run's snapshot (optionally
keeping the run's source version folder in sync) for the in-place
edit → re-run-step → compare iteration loop.

To create a new version, copy an existing folder (e.g. `baseline/`) and edit the
text files. The server picks up new folders immediately — no restart needed.

## Files

Each version folder must contain one `<step>.system.txt` and one
`<step>.user.txt` for every pipeline step below. A missing file fails run
creation / resume loudly — with the exception of the steps marked **optional**,
which a version may omit (the pipeline skips that pass when its templates are
absent), so versions predating an optional step keep loading and resuming.

| step | fired | purpose |
| --- | --- | --- |
| `zone_plan_root` | once per run | plan the overall scene from the user prompt + decide `is_atomic` |
| `zone_plan` | per nested region | plan one subregion + decide `is_atomic` |
| `overall_bbox` | once per run | size the root canvas from the scene plan |
| `zone_decompose_root` | once per run (root non-atomic) | split the root into top-level subregions |
| `zone_decompose` | per non-atomic region | split a region into subregions |
| `child_bbox_batch` | per decomposed region | resolve every child subregion's bbox in one call |
| `encapsulating_decompose` | per region | decide + enumerate the region's shell / perimeter objects |
| `anchor_decompose` | per atomic region | enumerate the defining objects of an atomic leaf |
| `negative_space_decompose` | per non-atomic region + root | fill interstitial gaps with ambient objects |
| `object_decomp` *(optional)* | per object batch, before `object_bbox_batch` | split any proposed object the mesh step can't build as one coherent mesh (containers, collections, surfaces with openings) into its constituent pieces; pass the rest through. Omitted on versions that split inline in the decompose steps instead. |
| `object_bbox_batch` | per object batch | resolve object bboxes in one call |
| `next_object` | loop per atomic region | propose more objects or declare the region done |
| `image_prompt` | per object | distill the object into a noun phrase for image generation |

The structured **output schemas** (JSON the model must return) are fixed in code
(`server/app/core/schemas.py`); templates only control the natural-language
prompt text. The `<step>` names above are template names — the event log keeps
logging the root and nested variants of a step under the same step id
(`zone_plan`, `zone_decompose`).

## Variables

Templates reference runtime scene state with variables. A variable token is a
brace token **wrapped in backticks**, with an UPPER_SNAKE name:

```
User prompt for the scene: '`{ROOT_PROMPT}`'
```

The whole token — backticks included — is replaced at render time, immediately
before each LLM call (the example above renders as
`User prompt for the scene: 'A modern house'`). Rules:

- A variable is exactly: backtick, `{UPPER_SNAKE_CASE}`, backtick. Both parts
  matter, which removes every collision with literal prompt text:
  - plain braces are never variables — writing `{target, kind}` or any other
    `{...}` prose without backticks is always literal;
  - backticked brace text whose body is not an UPPER_SNAKE name (e.g. the
    literal `` `{target, kind}` `` used when describing the relationships
    schema) is also literal.
- EVERY variable below may be used in EVERY template. Each step *natively
  populates* the variables listed for it; outside that set a variable
  resolves to the empty string (or the canonical empty-scene placeholder for
  `SCENE_CONTEXT` / `ROOT_OBJECTS` on the pre-scene steps) instead of failing
  the run. Only a token outside the vocabulary entirely (e.g. a typo) fails
  loudly.
- A template does not have to use every variable available to it — drop
  `` `{SCENE_CONTEXT}` `` from a step and that step simply runs without scene
  context.
- Both the `.system.txt` and `.user.txt` of a step are resolved with the same
  variable set.

### Scene-wide variables

Natively populated on every step except `zone_plan_root` (nothing exists yet:
`ROOT_PROMPT` is real, `SCENE_CONTEXT`/`ROOT_OBJECTS` render the empty-scene
placeholders, the rest are empty) and `overall_bbox` (adds `ROOT_PLAN`; the
root bbox itself is what that step produces, so the bbox-derived variables
are empty).

| variable | resolves to |
| --- | --- |
| `` `{ROOT_PROMPT}` `` | the run's root prompt (the text the user submitted) |
| `` `{ROOT_PLAN}` `` | the scene plan paragraph authored by `zone_plan_root` |
| `` `{ROOT_DIMENSIONS}` `` | root bbox dimensions, `(x, y, z) m` |
| `` `{ROOT_ORIGIN}` `` | root bbox world origin corner, `(x, y, z) m` |
| `` `{ROOT_HEADER}` `` | canonical scene header block: root prompt, plan, and overall bbox in prose |
| `` `{ROOT_OBJECTS}` `` | flat list of objects anchored directly to the root (shared shells / ground geometry), or a placeholder line when none exist |
| `` `{SCENE_CONTEXT}` `` | the embedded subregion tree — every subregion with its plan, bbox, and inline objects, recursing into nested subregions; the step's target region carries an inline `<-- TARGET:` marker |

Model-specific quirks (e.g. DeepSeek's reasoning pin) are NOT variables: they
are injected automatically at the LLM call boundary and never appear in
templates. (`DEEPSEEK_SUFFIX` from older snapshots still resolves, to the
empty string.)

`PRIOR_SUBJECTS` is retired the same way. It listed the noun phrases distilled
for the objects before this one, which chained every `image_prompt` call to its
predecessor; the pass now runs concurrently and off the structural path, so the
token resolves to the empty string in the snapshots that still carry it.

### Target-zone variables

Natively populated on every step that operates on a region (`zone_plan`,
`zone_decompose*`, `child_bbox_batch`, `anchor_decompose`,
`encapsulating_decompose`, `negative_space_decompose`, `object_bbox_batch`,
`next_object`, and `image_prompt`, whose zone is the region that owns the
object). For `zone_plan_root` / `overall_bbox` the zone is the root with only
its prompt (and plan, for `overall_bbox`) known. For `child_bbox_batch` the
"zone" is the parent region whose children are being placed.

| variable | resolves to |
| --- | --- |
| `` `{ZONE_ID}` `` | target region id |
| `` `{ZONE_PROMPT}` `` | target region seed prompt |
| `` `{ZONE_PLAN}` `` | target region plan (empty string until authored) |
| `` `{ZONE_PLACEMENT}` `` | target region placement — the semantic spatial description of where it sits within its parent (empty for the root, or until authored) |
| `` `{ZONE_DIMENSIONS}` `` | target region bbox dimensions, `(x, y, z) m` |
| `` `{FORMATTED_ZONE_DIMENSIONS}` `` | target region bbox dimensions in spoken form, `Wm by Hm by Dm` (e.g. `3.00m by 2.50m by 4.00m`) |
| `` `{ZONE_ORIGIN}` `` | target region bbox world origin corner, `(x, y, z) m` |
| `` `{ZONE_OBJECTS}` `` | flat list of the objects placed DIRECTLY inside the target region (its current contents) — e.g. to remind `next_object` what's already in the zone before it decides what to add. Placeholder line until the zone holds objects. |
| `` `{PARENT_ZONE_ID}` `` | the target region's ENCLOSING parent region id — the zone one level up the tree, distinct from `` `{ZONE_ID}` `` (the region in focus). Empty for the root, which has no parent. |
| `` `{PARENT_ZONE_PLAN}` `` | the enclosing parent region's plan (empty for the root, or until the parent is authored) |
| `` `{PARENT_ZONE_ORIGIN}` `` | the enclosing parent region's bbox world origin corner, `(x, y, z) m` (empty for the root) |

### Step-specific variables

| variable | natively populated on | resolves to |
| --- | --- | --- |
| `` `{TO_PLACE}` `` | `child_bbox_batch`, `object_bbox_batch` | pseudo-JSON block of the specs whose bboxes the solver must emit (id, parent, relationship kind, parent dims/origin, proxy shape, orientation for objects, prompt, placement, relationships) |
| `` `{PROPOSED_OBJECTS}` `` | `object_decomp` | pseudo-JSON block of the objects a decompose step proposed for the region (same shape as `` `{TO_PLACE}` ``), which `object_decomp` analyses and may split into buildable pieces |
| `` `{RETRY_BLOCK}` `` | `anchor_decompose`, `encapsulating_decompose`, `negative_space_decompose`, `next_object` | empty on the first attempt; after a rejected attempt, the prior emissions + rejection reasons with instructions to fix them |
| `` `{ADJACENT_ZONES}` `` | `encapsulating_decompose` | the regions adjacent to the target zone — the nearest region in each direction, found by casting a sphere of rays from the zone's centre (occlusion-aware, distance uncapped) — rendered in the same embedded form as `` `{SCENE_CONTEXT}` `` (plan, bbox, inline objects, nested subregions) but trimmed to just those neighbours; empty when the zone has no neighbours |
| `` `{OBJECT_PROMPT}` `` | `image_prompt` | the object's original prompt text |
| `` `{OBJECT_DIMENSIONS}` `` | `image_prompt` | `width=W.WWm, height=H.HHm, depth=D.DDm` |
| `` `{PROXY_SHAPE}` `` | `image_prompt` | `BOX` / `SPHERE` / `CAPSULE` / `HEMISPHERE` |
| `` `{IMAGE_TEMPLATE_FRONT}` `` / `` `{IMAGE_TEMPLATE_SIDE}` `` / `` `{IMAGE_TEMPLATE_TOP}` `` | `image_prompt` | the fixed image-generation wrapper for each view with `<<<SUBJECT>>>` marking where the model's phrase is slotted |
| `` `{SIBLING_OBJECTS}` `` | `image_prompt` | the objects already placed directly within the subject's OWN region (its siblings), each with id, prompt, noun phrase, structural parent (`<id> (<kind>)`), placement, relationships, and dimensions — no world coordinates |
| `` `{ROOT_OBJECTS_BRIEF}` `` | `image_prompt` | the root-region objects (shared shells / ground / fill) in bare form — id, prompt, and noun phrase only |
| `` `{OTHER_SUBREGIONS_BRIEF}` `` | `image_prompt` | every region EXCEPT the root and the subject's own region, as a flat list — each its name + prompt with a bare id/prompt/noun-phrase list of the objects inside it |

> `image_prompt` deliberately runs on a REDUCED, graduated scene context: its own
> region in semantic detail (`` `{ZONE_ID}` ``/`` `{ZONE_PROMPT}` ``/`` `{ZONE_PLAN}` ``/`` `{ZONE_PLACEMENT}` `` + `` `{SIBLING_OBJECTS}` ``) and
> everything beyond as bare id/prompt/noun-phrase (`` `{ROOT_PROMPT}` ``/`` `{ROOT_PLAN}` `` +
> `` `{ROOT_OBJECTS_BRIEF}` ``, `` `{OTHER_SUBREGIONS_BRIEF}` ``). It does NOT populate the heavy
> `` `{SCENE_CONTEXT}` `` / `` `{ROOT_HEADER}` `` / `` `{ROOT_OBJECTS}` `` / `` `{ZONE_OBJECTS}` `` blocks.
