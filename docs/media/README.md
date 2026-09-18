# media

`teleop.mp4` and `teleop.gif` are the same screen recording of `make teleop`,
captured 2026-09-18 on a MacBook Air (Apple silicon). Nothing is sped up or cut.

The 2940x954 / 25 s original is 48 MB and stays out of the repo. To rebuild both
derivatives from a fresh capture:

```bash
# Full clip, half size, no audio -- 1.5 MB. What the GIF links to.
ffmpeg -i in.mov -an -vf "scale=1470:-2" -c:v libx264 -crf 22 -preset slow \
  -pix_fmt yuv420p -movflags +faststart docs/media/teleop.mp4

# First 11 s, 800 px, 10 fps -- 3.0 MB. The README's inline preview.
ffmpeg -ss 0 -t 11 -i in.mov -filter_complex \
  "fps=10,scale=800:-1:flags=lanczos,split[a][b];\
   [a]palettegen=max_colors=96:stats_mode=diff[p];\
   [b][p]paletteuse=dither=none:diff_mode=rectangle" docs/media/teleop.gif
```

A GIF is used because it is the only format GitHub will render inline. A
`<video>` tag is stripped by GitHub's markdown sanitizer (verified against
`POST /markdown`: the tag renders as an empty paragraph), and both raw URLs
serve the mp4 as `application/octet-stream`, which downloads rather than plays.
So the mp4 is linked as a file, not promised as a player. 96 colours with dithering
off measured smallest at acceptable quality: 128+bayer was 3.8 MB, 64+bayer
2.8 MB but visibly dithered on flat wall, 96+none 3.0 MB and clean.
