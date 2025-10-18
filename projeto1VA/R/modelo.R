###############################################
# GRADIENT BOOSTING - HEART DISEASE DATASET
# Autor: Thiago Jacinto
# Objetivo: Prever se o paciente tem ou não doença cardíaca
###############################################

#----------------------------------------
# 1. Instalar e carregar pacote
#----------------------------------------
if(!require(gbm)) install.packages("gbm", dependencies = TRUE)
library(gbm)

#----------------------------------------
# 2. Carregar dataset (repositório alternativo confiável)
#----------------------------------------
heart <- read.csv("heart.csv")

head(heart)

#----------------------------------------
# 3. Separar variáveis preditoras e alvo
#----------------------------------------
heart$target <- as.numeric(heart$target)
set.seed(42)
n <- nrow(heart)
index <- sample(1:n, 0.8 * n)  # 80% treino, 20% teste

train <- heart[index, ]
test  <- heart[-index, ]

#----------------------------------------
# 4. Treinar modelo Gradient Boosting
#----------------------------------------
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

#----------------------------------------
# 5. Fazer previsões
#----------------------------------------
pred_prob <- predict(modelo, test, n.trees = 300, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

#----------------------------------------
# 6. Avaliar desempenho
#----------------------------------------
acuracia <- mean(pred_class == test$target)
print(paste("Acurácia:", round(acuracia * 100, 2), "%"))

# Matriz de confusão
matriz_confusao <- table(Predito = pred_class, Real = test$target)
print("Matriz de Confusão:")
print(matriz_confusao)

# Extrair valores
TP <- sum(pred_class == 1 & test$target == 1)
FP <- sum(pred_class == 1 & test$target == 0)
TN <- sum(pred_class == 0 & test$target == 0)
FN <- sum(pred_class == 0 & test$target == 1)

# Cálculo das métricas
acuracia <- (TP + TN) / (TP + TN + FP + FN)
sensibilidade <- TP / (TP + FN)         # Recall
precisao <- TP / (TP + FP)              # Precision
especificidade <- TN / (TN + FP)        # True Negative Rate
f1 <- 2 * (precisao * sensibilidade) / (precisao + sensibilidade)

# Exibir resultados formatados
cat("\n---------------------------\n")
cat("Métricas de desempenho do modelo GBM\n")
cat("---------------------------\n")
cat(paste("Acurácia:      ", round(acuracia * 100, 2), "%\n"))
cat(paste("Sensibilidade: ", round(sensibilidade * 100, 2), "%\n"))
cat(paste("Precisão:      ", round(precisao * 100, 2), "%\n"))
cat(paste("Especificidade:", round(especificidade * 100, 2), "%\n"))
cat(paste("F1-Score:      ", round(f1 * 100, 2), "%\n"))
cat("---------------------------\n\n")


# Previsões no treino
pred_train <- predict(modelo, train, n.trees = 300, type = "response")
pred_train_class <- ifelse(pred_train > 0.5, 1, 0)

# Acurácia no treino
acuracia_treino <- mean(pred_train_class == train$target)
acuracia_teste <- mean(pred_class == test$target)

cat("Acurácia Treino:", round(acuracia_treino * 100, 2), "%\n")
cat("Acurácia Teste: ", round(acuracia_teste * 100, 2), "%\n")

#----------------------------------------
# Matriz de confusão - forma simples
#----------------------------------------

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


#----------------------------------------
# 7. Importância das variáveis
#----------------------------------------
imp <- summary(modelo)
print(imp)

#----------------------------------------
# 8. Gráfico da importância
#----------------------------------------
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
