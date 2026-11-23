# Bov Classifier

Rede neural convolucional (CNN) em TensorFlow/Keras que classifica fotos de bovinos em duas classes: touro ou búfalo.

## Contexto

Trabalho da disciplina de Inteligência Artificial, no terceiro semestre do curso de Análise e Desenvolvimento de Sistemas, feito individualmente em 2023.

## O que tem aqui

- `classificador.ipynb`: carrega as imagens, aplica aumento de dados, define e treina a CNN e salva o modelo em `modelo_aula.h5`.
- `usandoClassificador.ipynb`: carrega o modelo salvo e classifica uma imagem isolada ou todas as imagens da pasta `test/`, mostrando a classe e a probabilidade.
- `modelo_aula.h5`: modelo já treinado, para usar sem treinar de novo.
- `data/classe0/` (touros) e `data/classe1/` (búfalos): 10 imagens de cada classe para treino e validação.
- `test/`: 6 imagens separadas para testar o modelo.

## Modelo

- Entrada de 150 x 150 pixels em RGB, normalizada para 0 a 1.
- Duas camadas `Conv2D` (32 e 64 filtros, 3 x 3, ReLU), cada uma seguida de `MaxPooling2D`.
- `Flatten`, `Dense` de 64 neurônios com ReLU, `Dropout` de 0,5 e saída `Dense` de 1 neurônio com sigmoide.
- Otimizador Adam, perda `binary_crossentropy`, 50 épocas, lotes de 32.
- `ImageDataGenerator` com `validation_split` de 25% e aumento de dados (cisalhamento, zoom e espelhamento horizontal).
- Saída abaixo de 0,5 é classe 0 (touro); a partir de 0,5, classe 1 (búfalo).

## Stack

Python 3.11, TensorFlow 2 (Keras), NumPy, SciPy e Jupyter Notebook.

## Como rodar

```bash
pip install tensorflow numpy scipy notebook
jupyter notebook
```

Abra `classificador.ipynb` para treinar (gera `modelo_aula.h5`) ou vá direto para `usandoClassificador.ipynb` para classificar as imagens de `test/` com o modelo que já está no repositório. Os caminhos são relativos, então rode os notebooks a partir da raiz do projeto.

## O que aprendi

- O ciclo completo de um classificador de imagens: preparar as pastas por classe, treinar, salvar o modelo e reaproveitá-lo em outro notebook só para inferência.
- Como o aumento de dados e o `Dropout` ajudam a reduzir o sobreajuste quando há poucas imagens.
- Com só 20 imagens, a validação usa poucas amostras e a acurácia oscila bastante entre épocas: o resultado serve para estudo, não como medida confiável. O próximo passo natural seria um conjunto maior e transfer learning com uma rede pré-treinada.
