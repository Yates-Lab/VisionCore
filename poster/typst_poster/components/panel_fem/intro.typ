// Intro: the variability problem, a possible sensory contribution, and the two
// questions the poster answers. Text only; panel 1 supplies the pictures.
#set page(width: 1183.79pt, height: 470pt, margin: 0pt, fill: white)
#set text(font: "Arial", size: 28pt, fill: rgb("193654"))
#set par(leading: 13pt, spacing: 0pt)
#let navy = rgb("193654")
#let copper = rgb("B8651B")
#let muted = rgb("526474")

#place(top + left, dx: 40pt, dy: 34pt)[
  #block(width: 1104pt)[
    #text(size: 46pt, weight: "bold", fill: copper)[What looks like noise may reflect active sensing]
    #v(44pt)
    Neurons respond differently across repeated presentations of the same stimulus.
    These #box[trial-to-trial] differences are often described as “noise,” but they can
    include sensory effects of the animal’s own movements. Even during fixation,
    small eye movements continually reshape visual input.
    #v(52pt)
    #grid(columns: (1fr, 1fr), column-gutter: 40pt,
      text(size: 30pt, weight: "bold", fill: navy)[How much cortical variability \ comes from the eye?],
      text(size: 30pt, weight: "bold", fill: navy)[Does that motion help \ encode spatial detail?],
    )
    #v(46pt)
    #text(size: 23pt, fill: muted)[
      We combine precise eye tracking, recordings from primary visual cortex (V1),
      and a model fitted to neural responses to address these questions.
    ]
  ]
]
