import sharp from "sharp";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const SHOT = process.argv[2];
const list = await (await fetch("http://127.0.0.1:9222/json/list")).json();
const page = list.find((t) => t.url.includes("localhost:3001"));
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
ws.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); } });
await new Promise((r) => ws.addEventListener("open", r));
const send = (m, p = {}) => new Promise((res) => { const n = ++id; pending.set(n, res); ws.send(JSON.stringify({ id: n, method: m, params: p })); });
const ev = async (e) => (await send("Runtime.evaluate", { expression: e, returnByValue: true })).result?.value;
const grab = async () => Buffer.from((await send("Page.captureScreenshot", { format: "png" })).data, "base64");
await send("Runtime.enable");
const b = await ev(`(function(){var x=[...document.querySelectorAll('button')].find(function(y){return y.textContent.trim().toUpperCase()==='A WINS';}); if(!x) return null; var r=x.getBoundingClientRect(); return JSON.stringify({x:Math.round(r.x+r.width/2),y:Math.round(r.y+r.height/2)});})()`);
if (!b) { console.log("not in voting state"); process.exit(0); }
const p = JSON.parse(b);
await send("Input.dispatchMouseEvent", { type: "mousePressed", x: p.x, y: p.y, button: "left", buttons: 1, clickCount: 1 });
await sleep(50);
await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: p.x, y: p.y, button: "left", buttons: 0, clickCount: 1 });
await sleep(2500);
console.log(await ev(`(function(){
  var rows = [...document.querySelectorAll('main > div.absolute.inset-0.flex')];
  if (rows.length < 2) return 'no warm row';
  var w = getComputedStyle(rows[1]), s = getComputedStyle(rows[0]);
  return 'shown row: opacity=' + s.opacity + ' z=' + s.zIndex + '   warm row: opacity=' + w.opacity + ' z=' + w.zIndex;
})()`));
const before = await grab();
// Simulate every warm splat finishing its load and activating.
const forced = await ev(`(function(){
  var rows = [...document.querySelectorAll('main > div.absolute.inset-0.flex')];
  if (rows.length < 2) return 0;
  var n = 0;
  rows[1].querySelectorAll('canvas').forEach(function(c){ c.style.visibility = 'visible'; n++; });
  return n;
})()`);
await sleep(2500);
const after = await grab();
const [a, c] = await Promise.all([sharp(before).raw().toBuffer(), sharp(after).raw().toBuffer()]);
let diff = 0;
for (let i = 0; i < a.length; i += 3) if (Math.abs(a[i] - c[i]) > 6) diff++;
console.log(`forced ${forced} warm canvases to visible`);
console.log(`pixels changed: ${diff}  ${diff === 0 ? "— the warm row cannot paint through opacity:0 ✓" : "— LEAK: the warm row is still painting"}`);
