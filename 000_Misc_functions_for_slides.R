# Functions usefull for slides
#
# Date updated:   2023-10-23
# Auhtor:         Christian Vedel 
# Purpose:        Contains miscellaneous functions used in the slides

# ==== ToPct ====
# Converts x to '100*x%' string

ToPct = function(x){
  pct = x * 100
  return(
    paste0(pct,"%")
  )
}
