
![bubbles 12 46 01 (1)](https://github.com/user-attachments/assets/ecd4a0dc-a11f-45b3-b603-b4e27e8e576b)
# Fluoddity
I struggle to describe Fluoddity. Think somewhere between interactive lava lamp and evolvable ant farm. 
Sometimes I'll see a meandering river, a candle flame, or branching lightning. Sometimes it's more like looking under a microscope as little amoebas devour each other and break apart. And sometimes, it's stranger than all that.

## WebGL Demo: https://aphid91.github.io/Fluoddity-Core/
<img width="1920" height="1129" alt="lavalamp_20260120_152448" src="https://github.com/user-attachments/assets/e8eda829-40d1-4add-afd6-80548a34cf5c" />
<img width="1920" height="1129" alt="lavalamp_20260120_152543" src="https://github.com/user-attachments/assets/6bf3ce1c-8a7f-487f-ad9e-1da67f73686c" />
<img width="1920" height="1129" alt="lavalamp_20260120_152527" src="https://github.com/user-attachments/assets/f1c1b933-f5fd-4802-b2b6-7887d483b71d" />

Fluoddity is a 2d particle system designed for realtime exploration. I've been tinkering with this idea for years, and it still feels like there's an ocean of possibilities I have yet to fully explore (~~3d generalization chief among them~~ https://github.com/aphid91/Fluoddity3D). There is a well considered algorithm that runs the actual physics, with an extensively Claude-Coded user interface built around it. 
The physics engine itself is a generalization of this excellent Sage Jenson page about physarum transport models: 
https://cargocollective.com/sagejenson/physarum

I strongly recommend reading at least the first few paragraphs if you want to understand how this project works. 
## Fluoddity-Core: https://github.com/aphid91/Fluoddity-Core
The algorithm that drives the Fluoddity particle system is pretty simple, but Fluoddity itself has a lot of bells and whistles. Fluoddity-Core exists as a minimal shell that is easier to understand and tinker with. It has just enough machinery to load and run a basic Fluoddity config with no UI fluff. Fluoddity-Core also hosts a Claude-Code port of the core engine to webgl that runs on github pages (This is the demo linked above).
Any advice or criticism is welcome. This is a toy I made for myself and I am more artist than engineer. 

## Features
 - "physics sliders" to customize simulation parameters.
 - particle selection/mutation to customize particle behavior
 - mouse drawing mode for making trails
 - Save/load system for physics + behavior
 - save strings with copy/paste from clipboard
 - parameter sweeps mode allows varying physics sliders across the canvas. X and Y sweeps for exploring 2d parameter space.
 - variable physics frequency with motion blur
 - ffmpeg based video recording
 - Experimental system for mixing different saved configs.
 - Tournament mode: run a grid of creatures side by side and breed from the ones you like, by hand or by CLIP.
 - Swappable particle brains: Fourier, Gabor, Lenia or MLP.
 - Undo and redo for every setting, with a browsable history.

## Undo

`Ctrl+Z` takes back the last change; `Ctrl+Shift+Z` or `Ctrl+Y` puts it back. A
whole slider drag counts as one step, however long you spent on it, and so does
loading a preset or pressing `Z` or `G`.

`Extras > Undo History` lists the steps, newest first. Hover one to see it live,
click to go back to it. Steps ahead of where you are sit greyed out, and making
a new change discards them.

It covers the settings and the brain, not the picture: the pattern regrows from
the restored settings rather than rewinding the screen, so Clear Canvas, Reset
and Fill are outside it, as is deleting a saved preset. Under a tournament the
sliders step back and the grid stays with whatever is running it.

## Tournament Mode

`Extras > Tournament Mode` splits the canvas into a grid of independent tiles, each running its own creature. Three tabs:

**Manual** — click the tiles you like, on the canvas or on the numbered buttons, then **Next Generation** to breed from them. `Mutation strength` sets how far the children stray, `Inject randoms` adds fresh creatures each round, `Crossover` mixes selected parents. **Undo** steps back a generation, **Save Selected...** writes the tiles you picked to your configs folder.

**Auto (Prompt)** — type a prompt, press **Set**, then **Start**. CMA-ES climbs the grid toward whatever the encoder scores as the closest match. `Grid` is tiles per side (2–8), `Steps per Gen` is how long each generation runs before it is scored. Tick `Search Physics Too` to let it move the physics sliders as well as the brain. `Encoder` picks which vision model does the scoring — CLIP B/32 is the default and by far the fastest; the larger ones see finer detail and cost proportionally more per generation.

**Explore (IMGEP)** — no prompt. It hunts for patterns *unlike* the ones it already has and files each keeper in an archive you can browse, sort and load from. Pick or create an archive, press **Start**; text goals are optional and steer it without confining it. `Search Physics Too` works here as it does in Auto, and the physics sliders stay live while it is on — they set the centre of the search rather than each tile's value. The **New** button asks which encoder the archive should use, and that is the only time you can choose: everything in an archive is measured in one encoder's space, so the `Encoder` box on the tab afterwards just shows which. To use a different one, make a new archive. Full description in [docs/imgep.md](docs/imgep.md).

One archive holds **every brain you use it with**, in a directory per layout. Novelty, admission, the map and the record book pool across all of them — they are about pictures — but the search can only breed from and seed on the brain that is running, because another brain's genome is a different creature under this one's decode. So changing brain mid-archive is fine and keeps everything you made: the browser tells you how many entries the current brain owns, and switching back picks up where you left off. Selecting an entry names the brain it was authored under, and clicking a foreign one switches to it. Changing brain also prints a line to the console, since it redirects where new results are filed.

`Search Brain Layouts Too`, beside it, lets the search change the brain's SHAPE as well as its weights — growing or shrinking a Fourier centre count, widening an MLP layer, adding or dropping one. A move carries the parent's brain into the new shape rather than starting over, so a grown brain begins by computing exactly what its parent computed and every bit of novelty it earns afterwards is its own. It then spends one expedition there and the archive decides: produce something genuinely unlike what is already stored and the new shape stays, produce nothing and the brain is handed back and that move is never tried again. The `Brain Layout` section bounds where it may roam — how often it moves, the largest brain it may build, the deepest and widest stack — and the walk it has taken is listed underneath. Ticking another brain type lets it JUMP to one, which carries no genome and is a restart, so it is off by default. Every verdict is recorded in `layouts.jsonl` in the archive folder; delete that file to let it retry moves it has already given up on.

Auto and Explore need the optional packages in `requirements.txt` (`onnxruntime-directml`, `tokenizers`, `cmaes`), and offer a **Download** button named for whichever encoder is selected the first time it is used. Manual mode needs none of that. The archive browser — `Extras > Archive Browser` — opens without an encoder too.

## Brain Modality

`Extras > Brain Modality` picks the function each particle's brain computes:

| Modality | Response |
| --- | --- |
| **Fourier** | sum of sine waves — the original, smooth periodic noise |
| **Gabor** | the same oscillation under a Gaussian envelope, so a unit answers near its own centre and is silent elsewhere |
| **Lenia** | a growth band: positive inside a narrow window of sensor values, negative outside it |
| **MLP** | a small neural net of one or more layers — `tanh`, `sin` or `gelu` per layer |

Fourier, Gabor and Lenia each have one size setting (Centers, Filters, Bumps). MLP instead has a **layer stack**: one row per hidden layer, with its own width slider and activation, an `x` to remove it and **+ Add layer** at the bottom. A readout underneath shows how much of the parameter budget the stack uses.

Every layer can be up to 48 units wide, at any depth. What stops a stack growing is the parameter budget in the readout — the sliders stop where the next unit would not fit, and `+ Add layer` greys out at 8 layers or when there is no room for one.

Width is not free past one layer. A deep stack needs scratch space the shader is built for, so the readout names the size it was built for and stepping over it costs speed — sharply, and in steps rather than smoothly. Two 16-wide layers run a few times slower than one; two 24-wide layers, several times. A single layer of any width is unaffected, as are the other three modalities.

**Right-click a layer** to work on its weights without changing its shape: pick a distribution (`normal`, `uniform`, `sparse`, `heavy-tail`), scale the layer up or down, reroll its weights or its biases, or reset it to how it was when the menu opened. These are ordinary rule edits — `Z` undoes them, and `Scale` records one undo step per drag, not one per frame. They are unavailable while the tournament grid is running, or while a hover preview is borrowing the brain.

**Source** at the top of the window names which brain the Inspector draws and the layer menu edits. With a rule loaded there is only one — every particle reads it. With no rule loaded each cohort runs its own generated brain, so the combo lists them and **Adopt as loaded rule** promotes the one you picked: edits to a cohort show at once but `File > Save` writes the loaded rule, so adopting is how you keep one. Under a tournament the list is the tiles, read-only.

Whatever the modality, the size settings fix the search dimension printed under the stack. Creatures are **not** portable between layouts, so changing anything that alters the parameter count resets the search and switches to that layout's archive — a width drag applies once you release it, everything else immediately. Presets record the brain they were saved under, including the whole stack, and switch to it on load.

The **Inspector** in the same window draws what the brain actually computes — one tile per unit, plus the whole brain — as a 2D slice through the 4D sensor space. For a deep MLP the tiles are the **last** hidden layer's units, the only ones that add up to the output. `Slice` chooses the plane and **Reseed plane** redraws the random one; `Output` chooses what is drawn, defaulting to a random projection of all four outputs into red, green and blue. The single-value views are blue for negative and orange for positive.

## Perform Mode

**Extras > Perform Mode Panel**, or **F11**, puts the picture on a second display with no interface on top of it, while every control stays here on the laptop.

Pick the display from the panel and press **Start**. The chosen one is remembered between sessions, but perform mode itself never resumes on its own — launching the app does not light up a projector.

The second window mirrors this one exactly: the same camera, the same view mode, the same exposure and bloom. It is not squashed to fit — you get your window's rectangle scaled up, with black bars on whichever axis does not match. It accepts no mouse or keyboard of its own and never takes focus, so **F11** always reaches this window even if the display is unplugged and Windows drops the leftover window on top of you.

Two things are hidden on **both** screens while performing: the parameter-sweep reticle and the draw-brush circle. Both are painted into the frame itself rather than drawn over it, so there is no clean copy to send without them.

## Inject Texture

**Extras > Inject Texture** injects texture into the fields the particles move through — procedural noise, an image, your own GLSL, or the simulation's own canvas fed back into itself.

**Inject** turns the whole stack off at once without losing anything, and **Keep on preset load** stops a preset replacing it — a preset carrying no stack of its own never touches yours either way.

The stack is a list of **layers**, composited top to bottom, and the whole thing is rebuilt every frame — so unticking a layer removes it, with nothing left behind. Each layer opens into three sections — what the **Picture** is, what it **Becomes**, what it **Drives**:

- **Picture** — the **Source**, then its own settings right beneath it: a file browser for `image` and `shader`, a camera and resolution for `webcam`, a Clear button for `brush`. Sources are `noise` (fbm, with scale, octaves, speed and domain warp), `image`, `gradient` (linear or radial), `shader` (your own `.frag`), `brush` (what you paint with the mouse), `webcam` (a live camera — pair it with `edge`, `gradient` or `curl` and moving in front of it pushes the particles around, or send it to `trail` and the brains read what the camera sees), and `feedback` (the sim's own trails, which closes the loop).
- **Becomes** — the **Mapping** from texture to vector field. `gradient` makes it a potential particles run down and `curl` turns it a quarter turn so they circulate along contours instead of piling into the bright spots; `edge` and `edge_flow` are the same pair read through a Sobel filter, with the direction normalised and the contrast in the length, so they follow outlines rather than slopes. `rg_direct` reads the channels straight as a vector — a camera's channels are all positive, so that one pushes everything into a corner and `rg_signed` is the version that takes mid grey as still. `polar` reads hue as an angle, `luminance` reads brightness alone.
- **Drives** — **Destination** (`force` accelerates particles; `strafe` slides them sideways without changing their velocity; `trail` paints into the trail canvas itself, so the particles' brains sense it as trail somebody else left rather than as a push), **Blend** (`replace`, `add`, `multiply` or `max` against the layers beneath — a `multiply` layer at the bottom has nothing to multiply and yields zero) and **Strength**.

**Blur** softens the source before the mapping reads it — worth turning up for `gradient` and the edge mappings, which are otherwise reading sensor grain — and **Direction** appears only for the mappings that take a slope, flipping whether particles are drawn to the bright regions or driven out of them.

On a `trail` layer, **Strength** is a deposit RATE rather than a level: the trail keeps accumulating it and decaying, so 1.0 settles near the level a busy simulation already runs at, and turning it up further will bury the particles' own trails under it.

Every dropdown explains itself while it is open — hover an entry in the list to read what it does, without having to pick it first.

A source shaped differently from the canvas is scaled to fill and centre-cropped, so a 4:3 camera keeps its proportions.

Every slider resets to its default on right-click, and each layer is a collapsing header so a deep stack stays readable.

Click a row's thumbnail (or **Inspect**) to open the **Inspect** panel, which shows three different things: the **source texture** as the layer produced it, **after mapping** — that layer alone, with direction drawn as hue and strength as brightness — and the **whole field** every layer composited together. Under **Advanced**, **Resolution** composites below canvas resolution; a forcing field is smooth, so Half usually looks identical and costs a quarter as much.

### Writing a field shader

Drop a `.frag` in `Documents/Fluoddity/shaders/` and pick it on a `shader` layer. Annotate a uniform and you get a slider for it; leave it unannotated and it keeps its default, so shaders written before this existed still work.

```glsl
#version 430
in  vec2 texcoord;
out vec4 fragColor;

uniform vec2  canvas_resolution;   // always provided
uniform float time;
uniform int   frame_count;
uniform vec2  mouse, prev_mouse;
uniform vec3  camera_pos, camera_dir;

uniform float speed;    // 0..5 = 1.0       "Speed"
uniform int   octaves;  // 1..8 = 4         "Octaves"
uniform vec3  tint;     // color = 1,.5,0   "Tint"
uniform bool  invert;   // = false          "Invert"

void main(){ fragColor = vec4(0.0); }
```

Press **V** to recompile without restarting. A shader that fails to build shows its compile error on its own row in red and contributes nothing; every other layer keeps rendering.

The stack is saved with a preset. Field injection is switched off while a tournament grid is running, because one field shared across isolated tiles would be scored instead of the creature.

## Design
Particles in Fluoddity have no direct interactions with each-other. Instead, they leave trails as they move. These trails decay and diffuse over time. Particles respond to the density and direction of trails around them.
There is no fixed rule that determines how particles respond to their senses. Instead, each particle has a simple neural-net like brain with only a few dozen parameters — 80 for the default Fourier brain, and see [Brain Modality](#brain-modality) for the others. These parameters are randomized on startup, and then mutated as the user selects which lineages to explore.

## Screenshots
<img width="797" height="595" alt="image" src="https://github.com/user-attachments/assets/343b2f6a-c09b-41c1-a370-247c223c33a7" />
<img width="797" height="597" alt="image" src="https://github.com/user-attachments/assets/c70ce389-fe63-4635-bb5f-bbd62bd7a317" />


## Model
Fluoddity generalizes the traditional physarum model in a couple ways.
### Trail interference
Particle trails have a velocity/flow vector which records the net "current" of particles. Thus, particle trails can interfere, and the trails from an equal number of particles flowing in opposite directions will cancel out.
### Behavior - Rules
Particle behavior is governed by a somewhat arbitrary black box function called a 'Rule'. I use a simple sum of sin waves because i wanted smooth, periodic noise. Trail sensor values are fed into this noise function, and the outputs are used to accelerate and reposition the particle.
### "Strafe"
In addition to forces causing acceleration, each paricle has a limited ability to "strafe", changing position independently from velocity. This is the least "principled" of my generalizations, but it is incredibly simple and enables some really beautiful patterns. Strafe allows particles to leave velocity trails which disagree with their direction of travel, enabling things like "swimming upstream" without turning around or "sidle to the left" without losing track of which way is "forward". 
### Symmetry
The traditional physarum model has some important symmetries that we would like to impose on our otherwise arbitrary noise functions. These symmetries can be toggled (or dialed down) in additional settings.

- Rotational: 
Rotate the whole world by 90°, and nothing should change: the dynamics are independent of global orientation. Particles should never favor the bottom left corner of the screen, for example. Achieving this symmetry is as simple as calculating all sensors/forces in a local coordinate system where "up" == particle velocity.

- Chiral:
Reflect the world across the X axis and nothing should change: the dynamics are identical when viewed in a mirror. Particles in the traditional physarum model display bilateral symmetry, they are not "left handed" or "right handed". Without this property, fluoddity particles show clockwise/counterclockwise bias, and the behavior space consists mostly of particles which are always turning left, or always turning right. This symmetry is achieved by calculating physics twice: once in mirrored coordinates, and averaging the results.

Enforcing these symmetries drastically reduces the prevalence of boring and degenerate Rules.

### Future Exploration
- Trail diffusion step replaced with arbitrary continuous cellular automata. wave equation or advection along flow lines could be interesting
- More than just two sensors.
- Disentangle "local orientation" from "particle velocity". Strafe mechanic hints at this being worthwhile.
- Particle internal state/ memory. Current particle behavior is memoryless aside from velocity persistence.
- Trails need not correspond to particle velocity. "Trail vector" could be just another output of the Rule function. Trail dimensionality could be increased.
- A more universal framework for describing these kinds of systems. One could generalize all the way to continuous cellular automata + continuous turmites.
  
### Requirements

- Python 3.x
- OpenGL-compatible graphics card

### Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Either:
pip install requirements, then run main.py
OR
Download a release and run Fluoddity.exe 

## Building

For instructions on building a standalone executable, see [BUILD.md](BUILD.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
