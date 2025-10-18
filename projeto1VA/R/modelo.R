if(!require(gbm)) install.packages("gbm", dependencies = TRUE)
library(gbm)

heart <- read.csv("heart.csv")
head(heart)

heart$target <- as.numeric(heart$target)
set.seed(42)
n <- nrow(heart)
index <- sample(1:n, 0.8 * n)
train <- heart[index, ]
test  <- heart[-index, ]

modelo <- gbm(
  formula = target ~ .,
  data = train,
  distribution = "bernoulli",
  n.trees = 300,
  interaction.depth = 3,
  shrinkage = 0.05,
  n.minobsinnode = 10,
  verbose = FALSE
)

pred_prob <- predict(modelo, test, n.trees = 300, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

acuracia <- mean(pred_class == test$target)
print(paste("Acurácia:", round(acuracia * 100, 2), "%"))

matriz_confusao <- table(Predito = pred_class, Real = test$target)
print("Matriz de Confusão:")
print(matriz_confusao)

TP <- sum(pred_class == 1 & test$target == 1)
FP <- sum(pred_class == 1 & test$target == 0)
TN <- sum(pred_class == 0 & test$target == 0)
FN <- sum(pred_class == 0 & test$target == 1)

acuracia <- (TP + TN) / (TP + TN + FP + FN)
sensibilidade <- TP / (TP + FN)
precisao <- TP / (TP + FP)
especificidade <- TN / (TN + FP)
f1 <- 2 * (precisao * sensibilidade) / (precisao + sensibilidade)

cat("\n---------------------------\n")
cat("Métricas de desempenho do modelo GBM\n")
cat("---------------------------\n")
cat(paste("Acurácia:      ", round(acuracia * 100, 2), "%\n"))
cat(paste("Sensibilidade: ", round(sensibilidade * 100, 2), "%\n"))
cat(paste("Precisão:      ", round(precisao * 100, 2), "%\n"))
cat(paste("Especificidade:", round(especificidade * 100, 2), "%\n"))
cat(paste("F1-Score:      ", round(f1 * 100, 2), "%\n"))
cat("---------------------------\n\n")

pred_train <- predict(modelo, train, n.trees = 300, type = "response")
pred_train_class <- ifelse(pred_train > 0.5, 1, 0)

acuracia_treino <- mean(pred_train_class == train$target)
acuracia_teste <- mean(pred_class == test$target)

cat("Acurácia Treino:", round(acuracia_treino * 100, 2), "%\n")
cat("Acurácia Teste: ", round(acuracia_teste * 100, 2), "%\n")

matriz_confusao <- table(Predito = pred_class, Real = test$target)

cat("\n===============================\n")
cat(" MATRIZ DE CONFUSÃO (Base R)\n")
cat("===============================\n\n")

print(matriz_confusao)

cat("\nLegenda:\n")
cat("TP = Verdadeiro Positivo (1 previsto e 1 real)\n")
cat("TN = Verdadeiro Negativo (0 previsto e 0 real)\n")
cat("FP = Falso Positivo (1 previsto, 0 real)\n")
cat("FN = Falso Negativo (0 previsto, 1 real)\n")

imp <- summary(modelo)
print(imp)

if(!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
library(ggplot2)

ggplot(imp, aes(x = reorder(var, rel.inf), y = rel.inf)) +
  geom_col(fill = "#1976D2") +
  coord_flip() +
  labs(
    title = "Importância das variáveis - Gradient Boosting (Heart Disease)",
    x = "Variável",
    y = "Importância Relativa (%)"
  ) +
  theme_minimal(base_size = 13)
