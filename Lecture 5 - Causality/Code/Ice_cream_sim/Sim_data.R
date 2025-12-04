# Christian Vedel
# Created: 2025-12-04
# Purpose: Simulate some data for examples

# ==== Libraries =====
library(tidyverse)

# ==== Make data ====
gen_data = function(capN = 1000){
  month = sample(1:12, capN, replace = TRUE)
  
  # Sinodial'ish function of month
  weather = 15 + 10 * sin((month - 3) / 12 * 2 * pi) + rnorm(capN, mean = 0, sd = 5)
  
  # Ice creams as function of weather
  icecream = exp(0.1 * weather + rnorm(capN, mean = 0, sd = 0.2))
  
  # Drownings as function of month
  drownings = (15 + 10 * sin((month - 3) / 12 * 2 * pi) + rnorm(capN, mean = 0, sd = 5))*0.1
  
  data.frame(
    month = month,
    weather = weather,
    icecream = icecream,
    drownings = drownings
  )

}

# Test data generation
set.seed(20)
df = gen_data(1000)

# Test correlation
mod1 = lm(drownings ~ log(icecream), data = df)
mod2 = lm(drownings ~ log(icecream) + factor(month), data = df)
mod3 = lm(drownings ~ log(icecream) + weather, data = df) # Proxy

summary(mod1)
summary(mod2)
summary(mod3)

# Test unbiasedness
set.seed(20)
coef = c()
for(i in 1:1000){
  dat = gen_data(1000)
  mod = lm(drownings ~ log(icecream) + factor(month), data = dat)
  coef = c(coef, coef(mod)[2])
}
coef %>% mean()
hist(coef)

# ==== Plots ====
p1 = df %>%
  ggplot(aes(x = icecream, y = drownings)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Drownings vs Ice cream sales",
       x = "Ice cream sales",
       y = "Number of drownings") +
  theme_bw()

p2 = df %>%
  ggplot(aes(x = month, y = drownings)) +
  geom_point() +
  geom_smooth() +
  labs(title = "Drownings vs Month",
       x = "Month",
       y = "Number of drownings") +
  theme_bw()

ggsave("Figures/Icecream_drownings.png", plot = p1, width = 5, height = 4)
ggsave("Figures/Drownings_month.png", plot = p2, width = 5, height = 4)

# ==== Save data =====
df %>% write_csv("Code/Icecream_kills.csv")
