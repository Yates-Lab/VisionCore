// Intro: the variability problem, the transition to fixational eye movements,
// and the two questions the poster answers. Text only; panel 1 supplies the
// pictures. Blocks are distributed evenly down the panel.
#set page(width: 1183.79pt, height: 470pt, margin: (left: 59.74pt, right: 30pt, top: 36pt, bottom: 38pt), fill: white)
#set text(font: "Arial", size: 30pt, fill: rgb("202632"))
#set par(leading: 14pt, spacing: 0pt, justify: false)
#let navy = rgb("193654")
#let copper = rgb("B8651B")

#let question(body) = grid(columns: (7pt, 1fr), column-gutter: 16pt,
  rect(width: 7pt, height: 92pt, fill: copper, radius: 2pt),
  align(horizon, text(size: 34pt, weight: "bold", fill: navy)[#body]),
)

// Navy headline matching the panel titles; copper marks the turn in the argument.
#block(height: 100%)[
  #text(size: 47pt, weight: "bold", fill: navy)[What looks like #text(fill: copper)[noise] may reflect #text(fill: copper)[active sensing]]
  #v(1fr)
  #block(width: 1000pt)[
    Neurons respond differently across repeated presentations of the same stimulus.
    These #box[trial-to-trial] differences are often described as “noise,” but they can
    include sensory effects of the animal’s own movements.
  ]
  #v(30pt)
  #block[Even during fixation, small eye movements continually reshape visual input.]
  #v(1fr)
  #grid(columns: (1fr, 1fr), column-gutter: 44pt,
    question[How much cortical variability \ comes from the eye?],
    question[Do fixational eye movements \ help encode spatial detail?],
  )
]
