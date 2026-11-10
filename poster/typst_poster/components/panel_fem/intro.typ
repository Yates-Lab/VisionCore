// Slide-deck framing: two debates, then the connection between them.
#set page(width: 1183.79pt, height: 470pt, margin: 0pt, fill: white)
#set text(font: "Arial", size: 24pt, fill: rgb("202632"))
#set par(leading: 9pt, spacing: 0pt)
#let navy = rgb("193654")
#let copper = rgb("B8651B")
#let muted = rgb("526474")
#let cite(t) = super(baseline: -6pt, size: 15pt)[[#t]]

#place(top + left, dx: 59.74pt, dy: 18pt)[
  #text(size: 48pt, weight: "bold", fill: navy)[Two debates in visual neuroscience]
]

#place(top + left, dx: 90pt, dy: 111pt)[
  #text(size: 25pt, weight: "bold", fill: navy)[Fixational eye movements]
]
#place(top + left, dx: 650pt, dy: 111pt)[
  #text(size: 25pt, weight: "bold", fill: navy)[Shared cortical variability]
]
#place(top + left, dx: 90pt, dy: 151pt)[
  #text(size: 35pt, weight: "bold", fill: copper)[Feature or bug?]
]
#place(top + left, dx: 650pt, dy: 151pt)[
  #text(size: 35pt, weight: "bold", fill: copper)[Noise or signal?]
]
#place(top + left, dx: 90pt, dy: 211pt)[
  #block(width: 500pt)[
    Are these tiny movements motor noise, \
    or part of how we see fine detail#cite[5,6]?
  ]
]
#place(top + left, dx: 650pt, dy: 211pt)[
  #block(width: 500pt)[
    The same image evokes different responses. \
    Internal noise, or meaningful sensory input?
  ]
]

// The two branches meet at the shared question, without imposing a causal
// direction between the debates themselves.
#place(top + left, dx: 90pt, dy: 282pt)[
  #image(bytes("<svg xmlns='http://www.w3.org/2000/svg' width='1060' height='34' viewBox='0 0 1060 34'><path d='M250 0 V12 L530 30 L810 12 V0' fill='none' stroke='#aab8c4' stroke-width='2'/></svg>"), width: 1060pt, height: 34pt)
]
#place(top + left, dx: 90pt, dy: 334pt)[
  #block(width: 1060pt)[#align(center)[
    #text(size: 32pt, weight: "bold", fill: navy)[
      How much apparent cortical noise comes from \
      the eye’s own movements?
    ]
  ]]
]
#place(top + left, dx: 90pt, dy: 424pt)[
  #block(width: 1060pt)[#align(center)[
    Can that movement-driven activity improve visual information?
  ]]
]
