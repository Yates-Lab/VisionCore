// Introduction panel: motivation, the two competing views, and the two
// questions the poster answers. Native Typst text; the eye-trace schematic is
// rendered by render_plots.py from the frozen plot_data.npz.
#set page(width: 1183.79pt, height: 470pt, margin: 0pt, fill: white)
#set text(font: "Arial", size: 24pt, fill: rgb("202632"))
#set par(leading: 9pt, spacing: 0pt)
#let navy = rgb("193654")
#let copper = rgb("B8651B")
#let muted = rgb("526474")
#let cite(t) = super(baseline: -9pt, size: 15pt)[[#t]]

#place(top + left, dx: 21pt, dy: 18pt)[
  #text(size: 40pt, weight: "bold", fill: navy)[Fixational eye movements: noise or signal?]
]

#place(top + left, dx: 21pt, dy: 92pt)[
  #block(width: 780pt)[
    Even during fixation the eyes drift and make microsaccades, so each repeat
    of an image lands on a different part of the foveal receptive fields.
    #v(22pt)
    #grid(columns: (380pt, 380pt), column-gutter: 20pt,
      block(fill: rgb("f1f3f6"), inset: 14pt, radius: 6pt, width: 100%, height: 150pt)[
        #text(weight: "bold", fill: navy)[Noise view]
        #v(12pt)
        #text(size: 22pt)[A repeated stimulus is a repeated input, so trial-to-trial
        variability is noise to be averaged away.]
      ],
      block(fill: rgb("f1f3f6"), inset: 14pt, radius: 6pt, width: 100%, height: 150pt)[
        #text(weight: "bold", fill: copper)[Active-sensing view]
        #v(12pt)
        #text(size: 22pt)[Eye motion turns space into time and is part of how the
        fovea encodes fine detail#cite[5,6].]
      ],
    )
    #v(24pt)
    #set text(size: 25pt)
    #grid(columns: (52pt, 1fr), row-gutter: 14pt,
      text(weight: "bold", fill: copper)[Q1],
      [How much V1 “noise” do fixational eye movements drive? #text(fill: muted)[→ 1–5]],
      text(weight: "bold", fill: copper)[Q2],
      [Does that activity carry visual information? #text(fill: muted)[→ 6–8]],
    )
  ]
]

#place(top + left, dx: 840pt, dy: 72pt)[
  #image("../../assets/intro_trace.pdf", width: 330pt, height: 330pt)
]
#place(top + left, dx: 772pt, dy: 408pt)[
  #block(width: 405pt)[#align(center)[#text(size: 20pt, fill: muted)[
    Median gaze on 88 repeats (navy: one path) \ vs a typical foveal V1 RF (0.42°)
  ]]]
]
