# Assistente Virtual - Subject Cluster
![Language](https://img.shields.io/badge/language-Python-orange)
![Repo Version](https://img.shields.io/badge/version-v0.1-blue)

## 📜 Resumo
Esse repositório é dedicado a códigos e estudos desenvolvidos em cima de mensagens que o chatbot teve problemas ao responder. A ideia é agrupar os principais tópicos que o assistente virtual teve dificuldades para identificar e responder.

## 🛠️ Afazeres
Os afazeres existentes nesse projeto até então são:
- [ ] **Tratamento de Dados**: Tratamentos e coleta de dados importantes para os estudos;
     - [ ] Tranformação dos caracteres para letras minúsculas;
     - [ ] Limpeza de caracteres especiais e pontuação utilizando RegEx;
     - [ ] Remoção de palavras irrelevantes (ex: preposições, conjunções, artigos e etc);
     - [ ] Aplicação de Lematização;
- [ ] **Machine Learning**: Desenvolvimento e aplicação de modelos de NLP;
     - [ ] Aplicar modelo TF-IDF;
     - [ ] (opcional) Aplicar normalização de dados;
     - [ ] Descobrir o número ideal de componentes utilizando PCA ou Truncated SVD;
     - [ ] Descobrir o número ótimo de clusters utilizando Elbow Method e Silhouette;
     - [ ] Aplicar modelo K-means;
- [ ] **Tratamento da Saída**: Transformar a saída do modelo de K-means em informações interpretáveis;

---
## 📫 Contribuição
Para contribuir para esse projeto, siga os seguintes passos:
1. Clone esse repositório e crie uma nova branch;
2. Faça suas mudanças no projeto;
3. Faça seus commits de acordo com as mudanças feitas;
4. Suba suas mudanças no github;
5. Abra uma Pull Request para as mudanças feitas.

*obs1:* Mensagem de commit devem estar de acordo com o commit semântico, descrito no exemplo:
```
<tipo>[escopo opcional]: <descrição>

[corpo opcional]

[rodapé opcional]
```
Outros exemplos de commit semântico podem ser vistos em: [semantic git examples](https://www.conventionalcommits.org/en/v1.0.0/);

---
## 📁 Folder structure
O projeto é estruturado conforme descrito abaixo:
```
├───data ->             Armazenar os dados a serem analisados
│ ├───external ->       Dados de fontes de terceiros
│ ├───interim ->        Dados provisórios que foram transformados
│ ├───processed  ->     Os conjuntos de dados canônicos finais para modelagem
│ └───raw ->            O dump de dados original e imutável
├───models ->           Os modelos pickles são armazenados aqui
├───notebooks ->        Experimentos e análises estão aqui
└───src ->              Experimentos bem-sucedidos modularizados em arquivos .py
     ├───data ->        Usado para fazer o ETL
     ├───features ->    Responsável pela principal funcionalidade do repo
     ├───test ->        Casos de teste criados para verificar a funcionalidade
     ├───models ->      Relacionado ao treinamento, teste e criação de modelos de ML
     ├───utils ->       Recursos úteis para outros módulos
     └───preview ->     Usado para dataviz
```
*obs:* Todos os dados armazenados na pasta `/data/` está inclusa no `.gitignore`;