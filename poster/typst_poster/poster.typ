// Full poster migration. Physical dimensions and scientific wording are frozen
// to the approved design; each region and text run is independently editable.
#let poster = json("poster_content.json")
#set page(width: poster.width * 1pt, height: poster.height * 1pt, margin: 0pt, fill: none)
#set par(spacing: 0pt, leading: 0pt)
#set text(font: "Arial", size: 21pt, top-edge: 0pt, bottom-edge: 0pt)

#let label(t) = place(top + left, dx: t.x * 1pt, dy: t.baseline * 1pt)[
  #rotate(t.angle * 1deg, origin: top + left, reflow: false)[
    #text(font: t.font, size: t.size * 1pt, weight: t.weight,
      fill: rgb(t.color), top-edge: 0pt, bottom-edge: 0pt)[#t.text]
  ]
]
#let region(r) = place(top + left, dx: r.x * 1pt, dy: r.y * 1pt)[
  #block(width: r.width * 1pt, height: r.height * 1pt, clip: false)[
    #place(top + left)[#image(r.file, width: r.width * 1pt, height: r.height * 1pt)]
    #for t in r.labels { label(t) }
  ]
]
#place(top + left)[#image("assets/background.svg", width: poster.width * 1pt, height: poster.height * 1pt)]
#for r in poster.regions { region(r) }
