import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';

export function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const rr=(rng,a,b)=>a+rng()*(b-a);
const mat=(c,o={})=>new THREE.MeshStandardMaterial({color:c,roughness:o.r??.85,metalness:o.m??0,flatShading:true,emissive:o.e??0x000000,emissiveIntensity:o.ei??1});
const BOX=new THREE.BoxGeometry(1,1,1);
const ICO=new THREE.IcosahedronGeometry(1,0);
const SPH=new THREE.IcosahedronGeometry(1,1);
const SHARED=new Set([BOX,ICO,SPH]);
const CYL=(rt,rb,h,n=8)=>new THREE.CylinderGeometry(rt,rb,h,n);
const CONE=(n=7)=>new THREE.ConeGeometry(1,1,n);
function mesh(parent,geo,m,x=0,y=0,z=0,sx=1,sy,sz,ry=0){const o=new THREE.Mesh(geo,m);o.position.set(x,y,z);o.scale.set(sx,sy??sx,sz??sx);o.rotation.y=ry;o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}

/* ---------- recipes ---------- */
function alpine(vp,rng,v){
  vp.setLight({dir:[0xffab73,1.9,[-8,6,5]],hemi:[0x5b6f96,0x141821,.55],fog:.03,ty:1.4,rad:13.5,pol:1.06});
  const R=vp.root;
  const g=mesh(R,CYL(7.4,7.7,.34,30),mat(0xe9eef4,{r:.95}),0,-.17,0);g.castShadow=false;
  for(let i=0;i<4;i++){const a=rr(rng,0,6.28),r=rr(rng,6.2,7.2),h=rr(rng,3.4,6.4),x=Math.cos(a)*r,z=Math.sin(a)*r;
    mesh(R,CONE(6),mat(0x2b323f),x,h/2,z,rr(rng,1.7,2.6),h,rr(rng,1.7,2.6),rr(rng,0,3));
    mesh(R,CONE(6),mat(0xe9eef4),x,h*.84,z,.85,h*.34,.85);}
  const pine=(x,z,s)=>{mesh(R,CYL(.09,.13,.5,6),mat(0x4a3628),x,.25*s,z,s);
    const c1=mat(0x2f4d3e),c2=mat(0x3a5c4a);
    mesh(R,CONE(7),c1,x,s*1.05,z,.62*s,1.15*s,.62*s);
    mesh(R,CONE(7),c1,x,s*1.72,z,.46*s,.9*s,.46*s);
    mesh(R,CONE(7),c2,x,s*2.3,z,.3*s,.65*s,.3*s);};
  for(let i=0;i<12+v*3;i++){const a=rr(rng,0,6.28),r=rr(rng,3.4,6.6);pine(Math.cos(a)*r,Math.sin(a)*r,rr(rng,.7,1.25));}
  const cabin=(x,z,ry,s)=>{
    x+=rr(rng,-.2,.2);z+=rr(rng,-.2,.2);
    mesh(R,BOX,mat(0x74503c),x,.55*s,z,1.7*s,1.1*s,1.25*s,ry);
    mesh(R,CYL(.03,1,1,4),mat(0xeef2f7,{r:.95}),x,1.5*s,z,(0.85*s+0.2)*1.414,.8*s,(0.625*s+0.2)*1.414,ry+Math.PI/4);
    const sn=Math.sin(ry),cs=Math.cos(ry),hd=.625*s+.035;
    mesh(R,BOX,mat(0x201812),x+sn*hd+cs*(-.35*s),.35*s,z+cs*hd-sn*(-.35*s),.34*s,.7*s,.07,ry);
    mesh(R,BOX,mat(0xffb36b,{e:0xff9c4a,ei:2.4,r:.6}),x+sn*hd+cs*(.38*s),.6*s,z+cs*hd-sn*(.38*s),.3*s,.3*s,.07,ry);
    mesh(R,BOX,mat(0x8d8d94),x+cs*.45*s,1.6*s,z-sn*.45*s,.2*s,.6*s,.2*s,ry);};
  cabin(-1.3,.7,.25,1.1);cabin(1.5,-.3,-.35,.9);cabin(.1,-1.9,.1,.75);
  for(const[lx,lz]of[[.4,1.9],[-2.3,-1.4]]){
    mesh(R,CYL(.05,.07,1.4,6),mat(0x20242c),lx,.7,lz);
    mesh(R,SPH,mat(0xffd9a0,{e:0xffc06a,ei:3}),lx,1.48,lz,.15);
    const pl=new THREE.PointLight(0xffb36b,7,6);pl.position.set(lx,1.55,lz);R.add(pl);}
  for(let i=0;i<4;i++){const a=rr(rng,0,6.28),r=rr(rng,4.5,6.8),s=rr(rng,.3,.7);
    const d=mesh(R,ICO,mat(0xe9eef4,{r:.98}),Math.cos(a)*r,s*.2,Math.sin(a)*r,s,s*.35,s,rr(rng,0,3));d.castShadow=false;}
}
function cyber(vp,rng,v){
  vp.setLight({dir:[0x9db4ff,.85,[6,9,-4]],hemi:[0x2a3352,0x0a0c12,.5],fog:.055,ty:1.9,rad:12.5,pol:1.12});
  const R=vp.root,NE=[0x36e2ff,0xff3d7e,0xffc23d,0x8f6bff,0x3dff9c];
  const g=mesh(R,BOX,mat(0x101319,{r:.32,m:.55}),0,-.15,0,7.6,.3,11);g.castShadow=false;
  const lane=mesh(R,BOX,mat(0x161a22,{r:.25,m:.6}),0,-.14,0,2.2,.32,11);lane.castShadow=false;
  for(const side of[-1,1]){let z=-4.6;while(z<4.4){
    const d=rr(rng,1.4,2.6),h=rr(rng,2.6,6.5),x=side*rr(rng,2.5,3.1),w=rr(rng,1.2,1.9);
    mesh(R,BOX,mat(rng()<.3?0x1d222e:0x171b24,{r:.8}),x,h/2,z+d/2,w,h,d);
    const n=1+((rng()*2)|0);
    for(let i=0;i<n;i++){const c=NE[(rng()*NE.length)|0],vert=rng()<.5;
      mesh(R,BOX,mat(c,{e:c,ei:2.6,r:.4}),x-side*(w/2+.04),rr(rng,.8,Math.max(1,h-.4)),z+d/2+rr(rng,-d*.3,d*.3),.06,vert?rr(rng,.9,2):.12,vert?.1:rr(rng,.6,d*.7));}
    z+=d+rr(rng,.15,.5);}}
  for(let i=0;i<4+v;i++){const side=rng()<.5?-1:1,c=NE[(rng()*NE.length)|0];
    mesh(R,BOX,mat(c,{e:c,ei:3,r:.4}),side*rr(rng,1.8,2.2),rr(rng,2,4.5),rr(rng,-4,4),.12,rr(rng,.5,1.1),rr(rng,.4,.9));}
  for(let i=0;i<3;i++){const y=rr(rng,3.2,5),z=rr(rng,-3.5,3.5),c=mesh(R,CYL(.015,.015,6.2,4),mat(0x0c0e13),0,y,z);c.rotation.z=Math.PI/2;c.castShadow=false;}
  for(let i=0;i<6;i++){const p=mesh(R,CYL(rr(rng,.3,.8),rr(rng,.3,.8),.02,10),mat(0x04060b,{r:.05,m:1}),rr(rng,-1.3,1.3),.03,rr(rng,-4.5,4.5));p.castShadow=false;}
  const l1=new THREE.PointLight(0x36e2ff,30,9),l2=new THREE.PointLight(0xff3d7e,26,9);
  l1.position.set(-1.6,2.6,-1.5);l2.position.set(1.8,2.2,2.2);R.add(l1);R.add(l2);
  vp.anim.push(t=>{l1.intensity=26+Math.sin(t*9.7)*4+Math.sin(t*23)*3;});
}
function island(vp,rng,v){
  vp.setLight({dir:[0xfff2dd,2.4,[7,10,4]],hemi:[0x9db8ff,0x18202e,.85],fog:.02,ty:2.7,rad:12.5,pol:1.08});
  const R=vp.root,TY=2.3;
  mesh(R,CYL(3.15,3.4,.5,9),mat(0x5d9152,{r:.95}),0,TY,0);
  mesh(R,CYL(3.4,.35,2.7,9),mat(0x565866,{r:.95}),0,TY-1.6,0);
  const py=TY+.25;
  mesh(R,BOX,mat(0xcfd3da),0,py+.18,0,1.7,.36,1.7);
  for(const ax of[-.6,.6])for(const az of[-.6,.6])mesh(R,CYL(.08,.1,1.05,7),mat(0xb8402f,{r:.7}),ax,py+.36+.52,az);
  mesh(R,BOX,mat(0x23272f),0,py+1.5,0,2.2,.16,2.2);
  mesh(R,CYL(.03,1,1,4),mat(0x2b303c),0,py+1.95,0,1.75,.7,1.55,Math.PI/4);
  const tor=new THREE.Group();tor.position.set(2.15,py,.2);tor.rotation.y=-.9;R.add(tor);
  mesh(tor,CYL(.06,.08,1.3,7),mat(0xc2453a,{r:.7}),-.5,.65,0);
  mesh(tor,CYL(.06,.08,1.3,7),mat(0xc2453a,{r:.7}),.5,.65,0);
  mesh(tor,BOX,mat(0xc2453a,{r:.7}),0,1.26,0,1.5,.12,.16);
  mesh(tor,BOX,mat(0x23272f),0,1.44,0,1.8,.12,.2);
  for(let i=0;i<4+v;i++){const a=rr(rng,0,6.28),r=rr(rng,1.7,2.7);
    if(Math.abs(Math.atan2(Math.sin(a+.4),Math.cos(a+.4)))<.5)continue;
    const x=Math.cos(a)*r,z=Math.sin(a)*r,s=rr(rng,.6,1);
    mesh(R,CYL(.07,.1,.7,6),mat(0x5a4232),x,py+.35*s,z,s);
    mesh(R,SPH,mat(0x6fae62,{r:.95}),x+rr(rng,-.1,.1),py+.85*s,z,.55*s,.48*s,.55*s);}
  for(const[lx,lz]of[[1.3,-1.4],[-1.5,1.1]]){
    mesh(R,CYL(.11,.15,.5,6),mat(0x9aa0ad),lx,py+.25,lz);
    mesh(R,BOX,mat(0xffd9a0,{e:0xffc06a,ei:2.2}),lx,py+.57,lz,.2,.16,.2);
    mesh(R,CYL(.02,.24,.16,4),mat(0x3a3f4c),lx,py+.72,lz,1,1,1,Math.PI/4);}
  for(let i=0;i<6;i++){const a=rr(rng,0,6.28),r=rr(rng,3.6,5.4),y=rr(rng,1.2,4.4),s=rr(rng,.2,.55);
    const rock=mesh(R,ICO,mat(0x565866),Math.cos(a)*r,y,Math.sin(a)*r,s,s*rr(rng,.7,1.3),s);
    const off=rr(rng,0,6.28),sp=rr(rng,.5,1);
    vp.anim.push(t=>{rock.position.y=y+Math.sin(t*sp+off)*.3;rock.rotation.y=t*.15*sp;});}
}
function desert(vp,rng,v){
  vp.setLight({dir:[0xffdcae,2.6,[6,8,3]],hemi:[0x8fa7c9,0x2a2118,.7],fog:.026,ty:1.1,rad:12,pol:1.05});
  const R=vp.root;
  const g=mesh(R,CYL(7.2,7.5,.34,30),mat(0xd9bc90,{r:1}),0,-.17,0);g.castShadow=false;
  const W=mat(0xf3efe6,{r:.9});
  mesh(R,BOX,W,0,.8,0,3.6,1.6,2.4);
  mesh(R,BOX,mat(0xf3efe6,{r:.9}),-1.4,2.05,.3,2.2,.9,1.8);
  mesh(R,BOX,mat(0xe8e2d4),.4,1.68,0,4.6,.16,3);
  mesh(R,BOX,mat(0xdcd6c6),-1.4,2.57,.3,2.9,.14,2.3);
  mesh(R,BOX,mat(0xffb36b,{e:0xff9a45,ei:1.6}),.9,.72,1.21,1.4,1.1,.04);
  mesh(R,BOX,mat(0x18242e,{r:.15,m:.7}),.9,.75,1.24,1.6,1.3,.05);
  mesh(R,BOX,mat(0x18242e,{r:.15,m:.7}),-1.4,2.05,1.21,1.4,.6,.05);
  mesh(R,BOX,mat(0xf3efe6),2.9,.09,-.6,2.7,.18,1.7);
  const w=mesh(R,BOX,mat(0x2e9fd8,{r:.15,m:.1,e:0x1179b0,ei:.55}),2.9,.15,-.6,2.2,.12,1.2);w.castShadow=false;
  const cac=(x,z,s)=>{const c=mat(0x4f7a41,{r:.95});
    mesh(R,CYL(.14,.17,1.1,7),c,x,.55*s,z,s);
    const a1=mesh(R,CYL(.09,.1,.5,6),c,x+.22*s,.72*s,z,s);a1.rotation.z=-.9;
    if(rng()<.6){const a2=mesh(R,CYL(.08,.09,.4,6),c,x-.2*s,.6*s,z,s);a2.rotation.z=.95;}};
  for(let i=0;i<4;i++){const a=rr(rng,2.6,5.9),r=rr(rng,3.4,6.2);cac(Math.cos(a)*r,Math.sin(a)*r,rr(rng,.7,1.2));}
  for(let i=0;i<7;i++){const a=rr(rng,0,6.28),r=rr(rng,3,6.6),s=rr(rng,.15,.5);
    mesh(R,ICO,mat(0xa8916d),Math.cos(a)*r,s*.4,Math.sin(a)*r,s,s*.7,s,rr(rng,0,3));}
  for(let i=0;i<5;i++){const st=mesh(R,BOX,mat(0xcbb894),.9+rr(rng,-.08,.08),.02,1.7+i*.55,.5,.08,.34);st.castShadow=false;}
}
function platform(vp,rng,v){
  vp.setLight({dir:[0xffffff,2.3,[5,9,6]],hemi:[0xbcd7ff,0x2c3a2c,.9],fog:.018,ty:1.9,rad:13,pol:1.02});
  const R=vp.root;
  const plat=(x,y,z,w,d)=>{mesh(R,BOX,mat(0x74c94e,{r:.95}),x,y,z,w,.28,d);
    mesh(R,BOX,mat(0x8a5a38,{r:1}),x,y-.44,z,w*.93,.62,d*.9);return[x,y,z];};
  const P=[[-3.5,.55,0,2.6,2.1],[-.8,1.35,-.3,2.2,1.8],[1.8,2.15,.3,2.3,1.9],[4.3,2.95,-.2,2,1.7]]
    .map(p=>plat(p[0]+rr(rng,-.2,.2),p[1],p[2]+rr(rng,-.2,.2),p[3]*rr(rng,.9,1.1),p[4]));
  mesh(R,CYL(.38,.38,1.1,12),mat(0x2fae54,{r:.5}),P[0][0]-.5,P[0][1]+.69,P[0][2]+.4);
  mesh(R,CYL(.46,.46,.34,12),mat(0x2fae54,{r:.5}),P[0][0]-.5,P[0][1]+1.31,P[0][2]+.4);
  mesh(R,CYL(.3,.3,.36,12),mat(0x145c2a),P[0][0]-.5,P[0][1]+1.31,P[0][2]+.4);
  for(let i=0;i<3;i++){const b=mesh(R,BOX,mat(0xf2b53a,{e:0x8a5b12,ei:.5,r:.6}),P[1][0]-.6+i*.62,P[1][1]+1.45,P[1][2],.55);
    vp.anim.push(t=>{b.position.y=P[1][1]+1.45+Math.sin(t*1.8+i*1.1)*.05;});}
  const gold=mat(0xf7c948,{e:0xb8860b,ei:.7,r:.4});
  for(let i=0;i<3;i++){const cg=new THREE.Group();cg.position.set(P[2][0]-.5+i*.5,P[2][1]+1.05,P[2][2]);R.add(cg);
    const c=mesh(cg,CYL(.26,.26,.05,16),gold,0,0,0);c.rotation.x=Math.PI/2;
    vp.anim.push(t=>{cg.rotation.y=t*2.8+i;cg.position.y=P[2][1]+1.05+Math.sin(t*1.6+i)*.08;});}
  mesh(R,CYL(.04,.05,1.5,6),mat(0xd8dde4),P[3][0],P[3][1]+.89,P[3][2]);
  mesh(R,SPH,gold,P[3][0],P[3][1]+1.66,P[3][2],.08);
  mesh(R,BOX,mat(0xff5d5d,{r:.8}),P[3][0]+.3,P[3][1]+1.45,P[3][2],.55,.34,.04);
  const cloud=(x,y,z,s)=>{const cm=mat(0xf2f5f9,{r:1});
    const grp=new THREE.Group();grp.position.set(x,y,z);R.add(grp);
    mesh(grp,SPH,cm,0,0,0,s,s*.6,s).castShadow=false;
    mesh(grp,SPH,cm,s*.8,-.05*s,0,s*.6,s*.4,s*.6).castShadow=false;
    mesh(grp,SPH,cm,-s*.75,-.08*s,.1,s*.55,s*.38,s*.55).castShadow=false;
    const off=rr(rng,0,6.28);vp.anim.push(t=>{grp.position.y=y+Math.sin(t*.5+off)*.15;});};
  cloud(-2.5,5,-2,rr(rng,.6,.9));cloud(2.8,5.6,1.5,rr(rng,.5,.8));cloud(.2,6.3,-3,rr(rng,.5,.7));
  const h1=mesh(R,CONE(9),mat(0x4f9450),-2,-.5,-8.5,4.5,3.4,4.5);h1.castShadow=false;
  const h2=mesh(R,CONE(9),mat(0x3f7a42),3,-.5,-9.5,3.6,2.6,3.6);h2.castShadow=false;
}
export const RECIPES=[alpine,cyber,island,desert,platform];

/* ---------- viewport ---------- */
export class Viewport{
  constructor(el,opts={}){
    this.el=el;this.anim=[];
    const r=this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
    r.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
    r.shadowMap.enabled=opts.shadows!==false;r.shadowMap.type=THREE.PCFSoftShadowMap;
    r.toneMapping=THREE.ACESFilmicToneMapping;r.toneMappingExposure=1.12;
    r.domElement.style.cssText='position:absolute;inset:0;width:100%;height:100%;display:block';
    el.appendChild(r.domElement);
    this.scene=new THREE.Scene();this.scene.fog=new THREE.FogExp2(0x000000,.03);
    this.camera=new THREE.PerspectiveCamera(38,1,.1,300);
    this.root=new THREE.Group();this.scene.add(this.root);
    this.hemi=new THREE.HemisphereLight(0x8899bb,0x11131a,.7);this.scene.add(this.hemi);
    this.dir=new THREE.DirectionalLight(0xffffff,2);this.dir.position.set(6,9,4);this.dir.castShadow=true;
    const sc=this.dir.shadow;sc.mapSize.set(2048,2048);sc.camera.left=-9;sc.camera.right=9;sc.camera.top=9;sc.camera.bottom=-9;sc.camera.near=.5;sc.camera.far=50;sc.bias=-.0004;sc.normalBias=.02;
    this.scene.add(this.dir);this.scene.add(this.dir.target);
    this.target=new THREE.Vector3(0,1.2,0);
    this.az=opts.az0??-.6;this.pol=1.05;this.rad=13;
    this.autoSpeed=opts.orbitSpeed??.14;this.vel=0;this.swingV=0;this._intro=1;this._pulse=1;
    el.style.touchAction='none';el.style.cursor='grab';
    this._down=e=>{this.dragging=true;this._px=e.clientX;this._py=e.clientY;el.setPointerCapture?.(e.pointerId);el.style.cursor='grabbing';};
    this._move=e=>{if(!this.dragging)return;const dx=e.clientX-this._px,dy=e.clientY-this._py;this._px=e.clientX;this._py=e.clientY;
      this.az-=dx*.0055;this.vel=-dx*.16;this.pol=Math.max(.5,Math.min(1.38,this.pol-dy*.004));};
    this._up=()=>{this.dragging=false;el.style.cursor='grab';};
    el.addEventListener('pointerdown',this._down);el.addEventListener('pointermove',this._move);
    el.addEventListener('pointerup',this._up);el.addEventListener('pointercancel',this._up);
    this._ro=new ResizeObserver(()=>this._size());this._ro.observe(el);this._size();
  }
  _size(){const w=this.el.clientWidth||2,h=this.el.clientHeight||2;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}
  setLight(o){if(o.dir){this.dir.color.set(o.dir[0]);this.dir.intensity=o.dir[1];this.dir.position.set(o.dir[2][0],o.dir[2][1],o.dir[2][2]);}
    if(o.hemi){this.hemi.color.set(o.hemi[0]);this.hemi.groundColor.set(o.hemi[1]);this.hemi.intensity=o.hemi[2];}
    this.scene.fog.density=o.fog??.028;this.target.set(0,o.ty??1.2,0);this.rad=o.rad??13.5;this.pol=o.pol??1.05;}
  setShadows(v){const r=this.renderer;if(r.shadowMap.enabled!==v){r.shadowMap.enabled=v;r.shadowMap.needsUpdate=true;
    this.root.traverse(m=>{if(m.material)m.material.needsUpdate=true;});}}
  load(i,seed,variant=0){
    const R=this.root;
    for(let n=R.children.length-1;n>=0;n--){const c=R.children[n];R.remove(c);
      c.traverse?.(o=>{if(o.geometry&&!SHARED.has(o.geometry))o.geometry.dispose();
        if(o.material)(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose&&m.dispose());});}
    this.anim=[];this._intro=0;
    RECIPES[i%RECIPES.length](this,mulberry32(seed),variant);
  }
  swing(){this.swingV+=1.5;}
  pulse(){this._pulse=0;}
  frame(dt,t){
    if(!this.dragging){this.az+=(this.autoSpeed+this.vel)*dt;this.vel*=Math.pow(.03,dt);}
    this.az+=this.swingV*dt;this.swingV*=Math.pow(.008,dt);
    this._intro=Math.min(1,this._intro+dt/.75);const ei=1-Math.pow(1-this._intro,3);
    this._pulse=Math.min(1,this._pulse+dt/.55);const ep=Math.sin(this._pulse*Math.PI)*.035;
    this.root.scale.setScalar((.92+.08*ei)*(1+ep));this.root.position.y=-(1-ei)*.5;
    for(const f of this.anim)f(t);
    this.camera.position.setFromSphericalCoords(this.rad,this.pol,this.az).add(this.target);
    this.camera.lookAt(this.target);
    this.renderer.render(this.scene,this.camera);
  }
  dispose(){this._ro.disconnect();this.renderer.dispose();this.renderer.domElement.remove();}
}
export function startLoop(vps){let last=performance.now(),stop=false;
  function tick(now){if(stop)return;const dt=Math.min(.05,(now-last)/1000);last=now;
    for(const v of vps)v.frame(dt,now/1000);requestAnimationFrame(tick);}
  requestAnimationFrame(tick);return()=>{stop=true;};}
