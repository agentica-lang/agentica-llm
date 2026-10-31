## Small helpers the other files share.

## The smaller of two Ints (`Int` has no `min`, see docs/gaps.md).
export def smaller(a: Int, b: Int) -> Int
  if a < b
    return a
  end

  return b
end
