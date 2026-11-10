// Panel 1 composition: Matplotlib layer (plots, rig, mask) plus the vector
// marmoset and brain drawings, placed where render_plots.py reserved them.
#let box = json("marmoset_box.json")
#set page(width: box.panel_width * 1pt, height: box.panel_height * 1pt, margin: 0pt, fill: none)
#place(top + left)[#image("../../assets/panel_fem_plots.pdf", width: box.panel_width * 1pt,
  height: box.panel_height * 1pt)]
#place(top + left, dx: box.x * 1pt, dy: box.y * 1pt)[
  #image("assets/marmoset.svg", width: box.width * 1pt)
]
#place(top + left, dx: box.brain_x * 1pt, dy: box.brain_y * 1pt)[
  #image("assets/brain_v1.svg", width: box.brain_width * 1pt)
]
#place(top + left, dx: box.eye_x * 1pt, dy: box.eye_y * 1pt)[
  #image("../../assets/panel_fem_eye.pdf", width: box.eye_size * 1pt,
    height: box.eye_size * 1pt)
]
