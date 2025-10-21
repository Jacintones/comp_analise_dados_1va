if(!require(gbm)) install.packages("gbm", dependencies = TRUE)
if(!require(MASS)) install.packages("MASS", dependencies = TRUE)
if(!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)

library(gbm)
library(MASS)
library(ggplot2)

data("Boston")
head("Boston")


set.seed(42)
n <- nrow(Boston)
index <- sample(1:n, 0.8 * n)
train <- Boston[index, ]
test  <- Boston[-index, ]

modelo <- gbm(
  formula = medv ~ .,
  data = train,
  distribution = "gaussian",
  n.trees = 800,
  interaction.depth = 4,
  shrinkage = 0.01,
  n.minobsinnode = 5,
  verbose = FALSE
)

pred <- predict(modelo, test, n.trees = 800)

rmse <- sqrt(mean((pred - test$medv)^2))
mae  <- mean(abs(pred - test$medv))
r2   <- 1 - (sum((test$medv - pred)^2) / sum((test$medv - mean(test$medv))^2))

cat("\n---------------------------\n")
cat("Métricas de desempenho (Regressão - Boston)\n")
cat("---------------------------\n")
cat(paste("RMSE:", round(rmse, 3), "\n"))
cat(paste("MAE: ", round(mae, 3), "\n"))
cat(paste("R²:  ", round(r2, 3), "\n"))
cat("---------------------------\n\n")

imp <- summary(modelo)
print(imp)

ggplot(imp, aes(x = reorder(var, rel.inf), y = rel.inf)) +
  geom_col(fill = "#00796B") +
  coord_flip() +
  labs(
    title = "Importância das Variáveis - GBM Regressão (Boston Housing)",
    x = "Variável",
    y = "Importância Relativa (%)"
  ) +
  theme_minimal(base_size = 13)

comparacao <- data.frame(Real = test$medv, Previsto = pred)

ggplot(comparacao, aes(x = Real, y = Previsto)) +
  geom_point(color = "#0288D1", alpha = 0.7, size = 3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Valores Reais vs. Previsto - GBM Regressão (Boston)",
    x = "Valor Real (medv)",
    y = "Valor Previsto"
  ) +
  theme_minimal(base_size = 13)
