-- Number display equations that carry an amsmath \tag{...}.
--
-- pandoc's MathML writer (2.14) silently drops \tag{}, so equations render
-- without numbers and the (N) call-outs in the prose have nothing to point at.
-- This filter strips the \tag{N} from each display equation and re-attaches the
-- number as an absolutely-positioned, right-aligned span, wrapping the equation
-- in a relatively-positioned div. The companion CSS lives in the writeup's YAML
-- header-includes (class .numbered-equation / .eqno).
--
-- Two adjacent $$...$$ blocks with no blank line between them parse as a single
-- Para holding several DisplayMath inlines, so the filter splits such a Para
-- into one numbered div per tagged equation (otherwise only the first would be
-- numbered and the rest would keep a literal \tag).

local function eqno_block(inl)
  local tag = inl.text:match("\\tag%s*%b{}")
  local num = tag:match("\\tag%s*{(.-)}")
  inl.text = inl.text:gsub("\\tag%s*%b{}", "")
  return pandoc.Div(
    { pandoc.Para({ inl }),
      pandoc.RawBlock("html", '<span class="eqno">(' .. num .. ')</span>') },
    pandoc.Attr("", { "numbered-equation" }, {})
  )
end

function Para(el)
  local has_tag = false
  for _, inl in ipairs(el.content) do
    if inl.t == "Math" and inl.mathtype == "DisplayMath"
        and inl.text:match("\\tag%s*%b{}") then
      has_tag = true
      break
    end
  end
  if not has_tag then return el end

  local blocks = {}
  for _, inl in ipairs(el.content) do
    if inl.t == "Math" and inl.mathtype == "DisplayMath" then
      if inl.text:match("\\tag%s*%b{}") then
        table.insert(blocks, eqno_block(inl))
      else
        table.insert(blocks, pandoc.Para({ inl }))
      end
    elseif inl.t == "Space" or inl.t == "SoftBreak" or inl.t == "LineBreak" then
      -- drop whitespace between stacked display equations
    else
      table.insert(blocks, pandoc.Para({ inl }))
    end
  end
  return blocks
end
