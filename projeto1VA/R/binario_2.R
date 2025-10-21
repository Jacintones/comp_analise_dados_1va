if(!require(gbm)) install.packages("gbm", dependencies = TRUE)
if(!require(caret)) install.packages("caret", dependencies = TRUE)
if(!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)

library(gbm)
library(caret)
library(ggplot2)

#----------------------------------------
# 1. Carregar o dataset (Banknote Authentication)
#----------------------------------------
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
cols <- c("variance", "skewness", "curtosis", "entropy", "class")
banknote <- read.table(url, sep = ",", col.names = cols)

#----------------------------------------
# 2. Preparar os dados
#----------------------------------------
banknote$class <- as.factor(banknote$class)  # transformar alvo em fator
set.seed(42)
n <- nrow(banknote)
index <- sample(1:n, 0.8 * n)
train <- banknote[index, ]
test  <- banknote[-index, ]

#----------------------------------------
# 3. Treinar o modelo GBM
#----------------------------------------
modelo <- gbm(
  formula = class ~ .,
  data = train,
  distribution = "bernoulli",  # classificação binária
  n.trees = 500,
  interaction.depth = 3,
  shrinkage = 0.05,
  n.minobsinnode = 10,
  verbose = FALSE
)

#----------------------------------------
# 4. Fazer previsões
#----------------------------------------
pred_prob <- predict(modelo, test, n.trees = 500, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
pred_class <- as.factor(pred_class)

#----------------------------------------
# 5. Avaliar desempenho
#----------------------------------------
cm <- confusionMatrix(pred_class, test$class, positive = "1")

cat("\n---------------------------\n")
cat("Métricas de desempenho - GBM (Banknote)\n")
cat("---------------------------\n")
print(cm)

#----------------------------------------
# 6. Importância das variáveis
#----------------------------------------
imp <- summary(modelo)
print(imp)

ggplot(imp, aes(x = reorder(var, rel.inf), y = rel.inf)) +
  geom_col(fill = "#00796B") +
  coord_flip() +
  labs(
    title = "Importância das Variáveis - GBM (Banknote Authentication)",
    x = "Variável",
    y = "Importância Relativa (%)"
  ) +
  theme_minimal(base_size = 13)

#----------------------------------------
# 7. Visualizar probabilidades
#----------------------------------------
prob_plot <- data.frame(Real = test$class, Prob = pred_prob)
ggplot(prob_plot, aes(x = Prob, fill = Real)) +
  geom_histogram(position = "identity", bins = 20, alpha = 0.6) +
  labs(
    title = "Distribuição das Probabilidades - GBM (Banknote)",
    x = "Probabilidade prevista de ser verdadeira",
    y = "Frequência"
  ) +
  theme_minimal(base_size = 13)
