library(dplyr)

# 1. Cria as 8 combinações fatoriais
base <- expand.grid(
  norm_data  = c("zscore", "minmax"),
  data    = c("50", "100"),
  norm_metric  = c("custom", "minmax")
)

# 2. Repete com randomização dentro de cada réplica
set.seed(42)  # Para reprodutibilidade

replicado_random <- bind_rows(
  base %>% sample_n(nrow(.)) %>% mutate(replica = 1),
  base %>% sample_n(nrow(.)) %>% mutate(replica = 2),
  base %>% sample_n(nrow(.)) %>% mutate(replica = 3)
) %>%
  mutate(run_order = row_number())  # ordem geral

# 3. Codifica os níveis
plan_final <- replicado_random %>%
  mutate(
    algo_cod = ifelse(norm_data == "zscore", -1, +1),
    dt_cod   = ifelse(data == "50", -1, +1),
    norm_cod = ifelse(norm_metric == "custom", -1, +1)
  )

# 4. Visualiza
print(plan_final)

# (opcional) Salva como CSV
write.csv(plan_final, "design_fatorial.csv", row.names = FALSE)
