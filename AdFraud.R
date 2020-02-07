
# Problema de negócio: Detectar fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile

# Descrição: Projeto realizado com o objetivo de criar um modelo de machine learning para determinar
# a possibilidade de um usuário realizar o download de um aplicativo infectado, após o click
# em um anúncio fraudulento.


#Dataset disponível em: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

#Configurando o ambiente
setwd("C:/FCD/BigDataRAzure/ProjetoFinal")
getwd()

#install packages

library(readr)
library(dplyr)
library(caret)
library(gmodels)
library(ROSE)
library(randomForest)
library(DMwR)
library(rpart)


## Etapa 1 - Coletando os Dados

#Leitura do arquivo original(200 milhões de observações) reduzido para 30000 observações apenas, pois o hardaware do notebook tem apenas 4 GB RAM

sample <- read.csv(file="train.csv",header=TRUE,sep=",",nrows = 30000,stringsAsFactors = FALSE)

#Arquivo com 30000 observações
write.csv(sample,"treino1.csv")

treino1 <- read.csv(file="treino1.csv",header=TRUE, sep=",")

View(head(treino1))
str(treino1)
summary(treino1)
any(is.na(treino1))



## Etapa 2 - Pré-Processamento

# As colunas click_time, attributed_time e X foram desconsideradas para a modelo 

treino1$click_time <-NULL
treino1$attributed_time <-NULL
treino1$X <-NULL

View(head(treino1))



# 3 ETAPA: Explorando relacionamento entre as variáveis: Matriz de Correlação

cor_data <-cor(treino1)
cor_data


# Análise: As variáveis "os" e "device" possuem forte correlação; "device" e "app" também,
# mas em menor grau.

library(corrplot)
corrplot(cor_data, method = 'color')

# Modelo randomForest para criar um plot de importância das variáveis
importance <- randomForest (is_attributed ~.,
                            data = treino1, 
                            ntree = 100, nodesize = 10, importance = T)

varImpPlot(importance)


#Convertendo para factor
treino1$is_attributed <- as.factor(treino1$is_attributed)


# Divisao dos dados

split1 <- createDataPartition(y = treino1$is_attributed, p = 0.7, list = FALSE)


# Criando dados de treino e de teste
dados_treino <- treino1[split1,]
dados_teste <- treino1[-split1,]

dados_treino_label <-dados_treino$is_attributed
dados_teste_label <-dados_teste$is_attributed

#Verificando distribuição da variável target
# Observa-se que a variável target possui 99% dos dados classificados como "0"(não realizou download) e 1%
# como "1"(realizou download). Por conseguinte, é necessário realizar técnicas para balancear a variável
# a fim evitar que o modelo preditvo (ML) desempenhe de tal forma com o menor viés possível.
table(dados_treino$is_attributed) 
prop.table(table(dados_treino$is_attributed))


ggplot(dados_treino,aes(x=is_attributed, fill=is_attributed)) +
  geom_bar()


#Realizando diferentes técnicas de balanceamento de classificação

#over sampling
data_balanced_over <- ovun.sample(is_attributed ~ ., data = dados_treino, method = "over",N = 41926)$data
table(data_balanced_over$is_attributed)


#Método ROSE

data.rose <- ROSE(is_attributed ~ ., data = dados_treino,hmult.majo=0.25, hmult.mino=0.5)$data


#Método SMOTE
ctrl <- trainControl(verboseIter = FALSE,
                     sampling = "smote")


model_rf_smote <- caret::train(is_attributed ~ .,
                               data = dados_treino,
                               method = "rf",
                               preProcess = c("scale", "center"),
                               trControl = ctrl)

final_smote <- predict(model_rf_smote, newdata = dados_teste,type='raw')

confusionMatrix(dados_teste$is_attributed,final_smote)



## Etapa 4: Treinando o modelo com diferentes técnicas de balaceamento (samplings) e algorítimos

#Algorítimo Decision Tree

tree.rose <- rpart(is_attributed ~ ., data = data.rose)
tree.over <- rpart(is_attributed ~ ., data = data_balanced_over)
tree.smote <- rpart(is_attributed ~ ., data = data.smote)


## ETAPA 5: Validando os modelos

pred.tree.rose <- predict(tree.rose, newdata = dados_teste, type='class')


confusionMatrix(dados_teste$is_attributed,pred.tree.rose)

pred.tree.over <- predict(tree.over, newdata = dados_teste,type='class')
confusionMatrix(dados_teste$is_attributed,pred.tree.over)

#Accuracy ROSE : 0.733
roc.curve(dados_teste$is_attributed, pred.tree.rose)


#Accuracy Oversampling: 0.864
roc.curve(dados_teste$is_attributed, pred.tree.over)

#Accuracy SMOTE: 0.875
roc.curve(dados_teste$is_attributed, final_smote)


#Modelo Support Vector Machines sob os dados balanceados(ROSE)

library(e1071)

modelo_svm_v1 <- svm(is_attributed ~ ., 
                     data = data.rose, 
                     type = 'C-classification', 
                     kernel = 'radial') 

pred.svm.rose <-predict(modelo_svm_v1,dados_teste,type='raw')
confusionMatrix(dados_teste$is_attributed,pred.svm.rose)


#Accuracy Oversampling: 0.864
roc.curve(dados_teste$is_attributed, pred.svm.rose)


#Modelo Logistic Regression

glm <- glm(is_attributed ~.,data=data.rose, family=binomial(link='logit'))

glm.pred <- predict(glm,dados_teste,type='response')
glm.pred <- ifelse(glm.pred >0.5,1,0)
glm.pred <- as.factor(glm.pred)
View(glm.pred)

#Accuracy 0.7878
confusionMatrix(dados_teste$is_attributed,glm.pred)

roc.curve(dados_teste$is_attributed,glm.pred)
