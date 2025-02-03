
## Reconhecimentos e Direitos Autorais
@autor: [DENILSON DA SILVA ALVES, TIAGO DE LIMA BATISTA, DANIEL NUNES DUARTE]
@data última versão: [02/02/2025]
@versão: 1.0
@Agradecimentos: Universidade Federal do Maranhão (UFMA), Professor Doutor Thales Levi Azevedo Valente, e colegas de curso.
Copyright/License
Este material é resultado de um trabalho acadêmico para a disciplina "MINERAÇÃO DE DADOS E APLICAÇÕES NA ENGENHARIA", sob a orientação do professor Dr. THALES LEVI AZEVEDO VALENTE, semestre letivo 2024.2, curso Engenharia da Computação, na Universidade Federal do Maranhão (UFMA). Todo o material sob esta licença é software livre: pode ser usado para fins acadêmicos e comerciais sem nenhum custo. Não há papelada, nem royalties, nem restrições de "copyleft" do tipo GNU. Ele é licenciado sob os termos da Licença MIT, conforme descrito abaixo, e, portanto, é compatível com a GPL e também se qualifica como software de código aberto. É de domínio público. Os detalhes legais estão abaixo. O espírito desta licença é que você é livre para usar este material para qualquer finalidade, sem nenhum custo. O único requisito é que, se você usá-los, nos dê crédito.


## Dataset Utilizado

O dataset utilizado neste projeto foi criado a partir de uma instituição de ensino superior e combina informações de diferentes bases de dados relacionadas a estudantes matriculados em cursos de graduação, como agronomia, design, educação, enfermagem, jornalismo, gestão, serviço social e tecnologias. As informações incluem:

- Dados demográficos e socioeconômicos dos estudantes;
- Histórico acadêmico no momento da matrícula;
- Desempenho acadêmico ao final do primeiro e segundo semestres.

O dataset é utilizado para construir modelos de classificação que preveem desistência e sucesso acadêmico. O problema foi formulado como uma tarefa de classificação em três categorias: desistente (*dropout*), matriculado (*enrolled*) e formado (*graduate*) ao final da duração normal do curso.

### Referências

M.V. Martins, D. Tolledo, J. Machado, L. M.T. Baptista, V. Realinho. (2021) "Early prediction of student's performance in higher education: a case study" Trends and Applications in Information Systems and Technologies, vol. 1, in Advances in Intelligent Systems and Computing series. Springer. DOI: 10.1007/978-3-030-72657-7_16

## Objetivos do Projeto

1. **Tratamento dos Dados:**
   - Identificar e lidar com valores ausentes;
   - Normalizar e padronizar as variáveis;
   - Realizar análises exploratórias.

2. **Pré-Processamento:**
   - Seleção de características relevantes;
   - Codificação de dados categóricos;
   - Divisão do dataset em conjuntos de treino e teste.

3. **Aplicação de Modelos:**
   - Modelos de classificação para prever desistência ou sucesso acadêmico;
   - Avaliação de métricas como precisão, *recall* e *F1-score*.

## Estrutura do Repositório

- **`data/`**: Contém o dataset original e os arquivos derivados após o tratamento.
- **`notebooks/`**: Scripts e notebooks utilizados para análise, pré-processamento e modelagem.
- **`models/`**: Modelos treinados e seus respectivos resultados.
- **`reports/`**: Relatórios e visualizações gerados durante o projeto.
- **`src/`**: Código-fonte organizado para facilitar reprodutibilidade e modularidade.

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/danielduartt/data-mining-project.git
   ```

2. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute os notebooks ou scripts na ordem especificada em `notebooks/README.md`.


## Licença

Este projeto está licenciado sob os termos da [MIT License](LICENSE).
Esta Permissão é concedida, gratuitamente, a qualquer pessoa que obtenha uma cópia deste software e dos arquivos de documentação associados (o "Software"), para lidar no Software sem restrição, incluindo sem limitação os direitos de usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do Software, e permitir pessoas a quem o Software é fornecido a fazê-lo, sujeito às seguintes condições:


---

Esperamos que este repositório seja útil como um exemplo prático de mineração de dados aplicada a problemas reais. Aproveite!

