

# Libraries
library(tidyverse)

x = read_csv(
  "Data/census.csv", 
  col_types = "c",
  locale = locale(encoding = "ISO-8859-1")
)

x %>% 
  select(first_names, sex) %>% 
  distinct() %>% 
  rename(gender = sex) %>% 
  write_csv("Data/Names_gender1787.csv")

