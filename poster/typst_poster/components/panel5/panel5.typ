// Panel 5 pilot: same scientific wording and physical design as the approved panel.
// Typst owns typography and layout. Matplotlib owns all data marks and axes.
#set page(width: 1479.74pt, height: 807.21pt, margin: 0pt, fill: white)
#set text(font: "Arial", size: 21pt, fill: rgb("202632"), top-edge: "cap-height", bottom-edge: "baseline")
#set par(leading: 8pt, spacing: 0pt)
#let navy = rgb("193654")
#let corrected = rgb("667480")
#let copper = rgb("B8651B")
#let muted = rgb("526474")
#let matrix(name, label, color, weight: 2.5pt) = stack(dir: ttb, spacing: 0pt,
  block(width: 190pt, height: 54pt)[#align(center)[#move(dy: 1.23pt)[#text(size: 22pt, fill: color)[#label]]]],
  block(width: 190pt, height: 190pt)[
    #image("assets/" + name + ".pdf", width: 190pt, height: 190pt)
    #place(top + left)[#rect(width: 190pt, height: 190pt, stroke: (paint: color, thickness: weight), fill: none)]
  ],
)
#let operator(symbol) = block(width: 60pt, height: 244pt)[
  #align(center + horizon)[#move(dy: 27pt)[#text(size: 30pt)[#symbol]]]
]
#let matrix-row(a, b, c) = grid(columns: (190pt, 60pt, 190pt, 60pt, 190pt), a, operator("="), b, operator("+"), c)
#let summary(title, before, after, asset) = stack(dir: ttb, spacing: 0pt,
  block(width: 330pt, height: 36pt)[#align(center)[#text(size: 25pt, weight: "bold", fill: navy)[#title]]],
  block(width: 330pt, height: 44pt)[#align(center)[
    #text(size: 26pt, weight: "bold", fill: navy)[#before]#h(18pt)#text(size: 26pt, fill: muted)[→]#h(18pt)#text(size: 26pt, weight: "bold", fill: corrected)[#after]
  ]],
  block(width: 330pt, height: 40.37pt)[#align(center)[p ≤ 0.001]],
  move(dy: -10pt, image("assets/" + asset + ".pdf", width: 315pt, height: 515pt)),
)
#place(top + left, dx: 9pt, dy: 15pt)[
  #grid(columns: (42pt, 1fr), column-gutter: 9pt, align: horizon,
    circle(radius: 21pt, fill: copper, stroke: none)[#align(center + horizon)[#text(size: 29pt, weight: "bold", fill: white)[4]]],
    text(size: 40pt, weight: "bold", fill: copper)[FEM correction reduces variability and shared noise],
  )
]
#place(top + left, dx: 30.13pt, dy: 109.37pt)[
  #stack(dir: ttb, spacing: 71pt,
    matrix-row(matrix("total", [Total\ covariance], rgb("80858e")), matrix("stimulus", [Stimulus\ covariance], rgb("D55E00")), matrix("uncorrected", [Uncorrected\ residual], navy, weight: 7pt)),
    matrix-row(matrix("uncorrected", [Uncorrected\ residual], navy, weight: 7pt), matrix("fem", [FEM\ component], rgb("0072B2")), matrix("corrected", [Corrected\ residual], corrected, weight: 7pt)),
  )
]
#place(top + left, dx: 800.13pt, dy: 103pt)[
  #grid(columns: (315pt, 315pt), column-gutter: 20pt,
    summary([Population Fano factor], [1.25], [0.81], "fano"),
    summary([Mean noise correlation], [0.076], [0.010], "noise"),
  )
]
#place(top + left, dx: 30.13pt, dy: 682.37pt)[#block(width: 690pt)[#align(center)[Neuron × neuron · identical unit order]]]
#place(top + left, dx: 234.37pt, dy: 710.37pt)[
  #image("assets/colorbar.pdf", width: 281.52pt, height: 10.5pt)
  #place(top + left, dx: -80pt, dy: 16pt)[#block(width: 160pt)[#align(center)[−5.2 spikes²]]]
  #place(top + left, dx: 100.76pt, dy: 16pt)[#block(width: 80pt)[#align(center)[0]]]
  #place(top + left, dx: 201.52pt, dy: 16pt)[#block(width: 160pt)[#align(center)[5.2 spikes²]]]
]
#place(top + left, dx: 20.13pt, dy: 754.37pt)[
  #text(fill: muted)[19 sessions; 25-ms bins. Points: mean ± SD. Gray bands: 95% shuffle null; violins: pairwise correlations.]
]
#place(top + left, dx: 20.13pt, dy: 781.37pt)[
  #text(fill: muted)[Corrected mean noise correlation: 95% CI −0.010 to 0.029. Both tests compare correction with shuffle.]
]

#place(top + left, dx: 0.13pt, dy: 353.37pt)[#image("assets/residual_connector.svg", width: 700pt, height: 85pt)]
