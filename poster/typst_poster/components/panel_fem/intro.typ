// Intro: fixational eye movements, noise or sensing? Eye and one-neuron raster
// with their usual readings and reinterpretations, a plain-language explanation,
// and a map of the answers onto the panels. Pictures come from render_plots.py.
#set page(width: 1183.79pt, height: 470pt, margin: 0pt, fill: white)
#set text(font: "Arial", size: 22pt, fill: rgb("202632"))
#set par(leading: 7pt, spacing: 0pt)
#let navy = rgb("193654")
#let copper = rgb("B8651B")
#let muted = rgb("526474")
#let grey = rgb("8a96a3")
#let chip = rgb("f0f3f6")

// Headline
#place(top + left, dx: 21pt, dy: 18pt)[
  #text(size: 32pt, weight: "bold", fill: muted)[Fixational eye movements:]
]
#place(top + left, dx: 21pt, dy: 58pt)[
  #text(size: 54pt, weight: "bold", fill: copper)[noise, or sensing?]
]

// Pictures with their usual reading and reinterpretation
#let reading(usual, here) = stack(dir: ttb, spacing: 8pt,
  text(size: 20pt, fill: grey)[usually: #usual],
  text(size: 22pt, weight: "bold", fill: copper)[↓ here: #here],
)
#place(top + left, dx: 21pt, dy: 140pt)[#image("../../assets/intro_eye.pdf", width: 280pt)]
#place(top + left, dx: 30pt, dy: 300pt)[#reading[motor noise][how the eye senses]]
#place(top + left, dx: 345pt, dy: 146pt)[#image("../../assets/intro_raster.pdf", width: 280pt)]
#place(top + left, dx: 345pt, dy: 300pt)[#reading[neural noise][what the eye delivers]]

// Explanation and closing line
#place(top + left, dx: 690pt, dy: 128pt)[
  #block(width: 470pt)[
    During fixation the eye keeps moving the image across the center of gaze, where
    each neuron sees a patch only about a quarter of a thumbnail wide at arm’s length.
  ]
]
#place(top + left, dx: 690pt, dy: 248pt)[
  #block(width: 470pt)[
    #text(size: 25pt, weight: "bold", fill: navy)[
      Is what looks like noise — in the eye and in cortex — part of how we see?
    ]
  ]
]
#place(top + left, dx: 690pt, dy: 342pt)[
  #text(size: 27pt, style: "italic", font: "Liberation Sans", fill: copper)[A fingertip, not a camera?]
]

// Answer map
#let answer(n, body) = box(fill: chip, radius: 10pt, inset: (x: 14pt, y: 11pt),
  width: 372pt, height: 70pt)[
  #grid(columns: (auto, 1fr), column-gutter: 12pt, align: horizon,
    text(size: 28pt, weight: "bold", fill: copper)[#n],
    text(size: 19pt)[#body],
  )
]
#place(top + left, dx: 21pt, dy: 388pt)[
  #grid(columns: (372pt, 372pt, 372pt), column-gutter: 12pt,
    answer[1–5][Most of the “noise” comes \ from eye motion],
    answer[6][It’s the image moving on the \ retina, not a brain signal],
    answer[7–8][In a model, the motion helps \ neurons encode fine detail],
  )
]
