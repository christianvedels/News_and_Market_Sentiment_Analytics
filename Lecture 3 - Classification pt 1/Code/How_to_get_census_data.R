# Libraries
library(tidyverse)

# Read data
x = read_csv(
  "Data/census.csv", # Can be found here: https://www.rigsarkivet.dk/udforsk/link-lives-data/
  col_types = "c",
  locale = locale(encoding = "UTF-8")
)

# Clean sample and save
x %>% 
  select(first_names, sex) %>% 
  group_by(first_names, sex) %>% 
  count() %>% 
  rename(gender = sex) %>% 
  mutate(
    gender = case_when(
      gender == "f" ~ "female",
      gender == "m" ~ "male",
      TRUE ~ NA
    )
  ) %>% 
  write_csv("Data/Names_gender1787.csv")

