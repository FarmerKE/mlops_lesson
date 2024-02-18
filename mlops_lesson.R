library(tidyverse)
library(tidymodels)
library(palmerpenguins)
library(gt)
library(ranger)
library(brulee)
library(pins)
library(vetiver)
library(plumber)
library(conflicted)
tidymodels_prefer()
options(tidymodels.dark = TRUE)
conflicts_prefer("penguins", "palmerpenguins")

# EDA ----
palmerpenguins::penguins %>% 
  filter(!is.na(sex)) %>% 
  ggplot(aes(
    x = flipper_length_mm,
    y = bill_length_mm,
    color = sex,
    size = body_mass_g
  )) +
  geom_point(alpha = 0.5) +
  facet_wrap(~species)

penguins_df <- palmerpenguins::penguins %>% 
  drop_na(sex) %>% 
  select(-year, -island)

set.seed(1234)

penguin_split <- initial_split(penguins_df, strata = sex)
penguin_train <- training(penguin_split)
penguin_test <- testing(penguin_split)
