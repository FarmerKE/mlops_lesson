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
library(torch)
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

# Training and Test split
set.seed(1234)

penguin_split <- initial_split(penguins_df, strata = sex)
penguin_train <- training(penguin_split)
penguin_test <- testing(penguin_split)

penguin_folds <- vfold_cv(penguin_train)

# Feature engineering
penguin_rec <- 
  recipe(sex ~ ., data = penguin_train) %>% 
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_dummy(species) %>% 
  step_normalize(all_numeric_predictors())

# Models ----
# Log Regression
glm_spec <- 
  logistic_reg(penalty = 1) %>% 
  set_engine("glm")

# Random Forest
tree_spec <- 
  rand_forest(min_n = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# Torch model
mlp_brulee_spec <- 
  mlp(
    hidden_units = tune(),
    epochs = tune(),
    penalty = tune(),
    learn_rate = tune()
  ) %>% 
  set_engine("brulee") %>% 
  set_mode("classification")

# Bayes optimisation for Hyperparameter tuning----
bayes_control <- control_bayes(no_improve = 10L,
                               time_limit = 20,
                               save_pred = TRUE,
                               verbose = TRUE,
                               seed = 5678)

# Fit Models and Tune Hyper Parameters----
workflow_set <- workflow_set(
  preproc = list(penguin_rec),
  models = list(glm = glm_spec,
                tree = tree_spec,
                torch = mlp_brulee_spec)
) %>% 
  workflow_map(
    "tune_bayes",
    iter = 50L,
    resamples = penguin_folds,
    control = bayes_control
  )

rank_results(workflow_set,
             rank_metric = "roc_auc",
             select_best = TRUE) %>% 
  gt()

workflow_set %>% 
  autoplot()

best_model_id <- "recipe_torch"

# Select best model
best_fit <- 
  workflow_set %>% 
  extract_workflow_set_result(best_model_id) %>% 
  select_best(metric = "accuracy")

final_workflow <- 
  workflow_set %>% 
  extract_workflow(best_model_id) %>% 
  finalize_workflow(best_fit)

final_fit <-
  final_workflow %>% 
  last_fit(penguin_split)

#Fit model on the test/train data
final_fit %>% 
  collect_metrics() %>% 
  gt()

# Plot final metric
final_fit %>% 
  collect_predictions() %>% 
  roc_curve(sex, .pred_female) %>% 
  autoplot()

# Create vetiver model
final_fit_to_deploy <-final_fit %>% 
  extract_workflow()

v <- vetiver_model(
  final_fit_to_deploy,
  model_name = "penguins_model"
)

pin_loc <- 
  pins:::github_raw("FarmerKE/mlops_lesson/master/pins-r/_pins.yaml")

model_board <- pins:::board_url(pin_loc)
model_board %>% vetiver_pin_write(v)

model_board %>% 
  vetiver_write_plumber("penguins_model")
write_board_manifest(model_board)

model_board %>% 
  vetiver_write_docker()
























