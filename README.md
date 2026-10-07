# Packoscope

Projeto da disciplina CIN0144 (Aprendizado de Máquina e Ciência de Dados, CIn-UFPE) que classifica fluxos de rede da base CIC-IDS-2017 como benignos ou maliciosos.

| Arquivo | Conteúdo |
|---|---|
| `make_dataset.py` | Junta os 8 CSV da base em `data/pacotes.parquet`. |
| `eda.ipynb` | Análise exploratória (Entrega 1). |
| `main.ipynb` | Experimento de pré-processamento com kNN, 72 pipelines (Entrega 2). |
| `resultados/` | Tabela com uma linha por pipeline, gerada pelo `main.ipynb`. |

## Como rodar o experimento da Entrega 2

O passo a passo abaixo parte de uma máquina sem nada instalado. Quem já tem o `uv` e a base consolidada pode pular para o passo 4.

### Requisitos

- **Memória:** pelo menos 8 GB livres. A preparação dos dados chega a cerca de 7,2 GB no início do notebook, por isso é recomendado um computador com 16 GB e os outros programas fechados.
- **Disco:** cerca de 2 GB para os CSV, o parquet e o ambiente Python.
- **Tempo:** de 1 a 2 horas em um processador de 12 threads. Com menos núcleos, o tempo aumenta.
- **Sistema:** Linux, macOS ou Windows. Os comandos abaixo valem para os três, com as diferenças indicadas.

### 1. Instalar o uv

O `uv` instala a versão certa do Python e todas as bibliotecas com as mesmas versões usadas pela equipe.

Linux e macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Depois da instalação, feche e abra o terminal de novo e confira com `uv --version`.

### 2. Baixar o repositório

```bash
git clone https://github.com/vinilauter/packoscope.git
cd packoscope
```

Se o repositório já estiver clonado, entre na pasta e rode `git pull` para pegar a versão mais recente.

**Todos os comandos a partir daqui devem ser rodados dentro da pasta `packoscope`.** Fora dela, o `uv` não encontra o projeto e mostra erros como `No pyproject.toml found` ou `Failed to spawn: jupyter`.

### 3. Baixar a base e gerar o parquet

A base não fica no Git, porque os CSV têm centenas de MB.

1. Na página do dataset (https://www.unb.ca/cic/datasets/ids-2017.html), baixe o arquivo `MachineLearningCSV.zip`. O site pede nome e e-mail antes do download.
2. Descompacte o arquivo e copie os 8 arquivos `.csv` para a pasta `data/raw/` do repositório. Os CSV precisam ficar direto em `data/raw/`, e não em uma subpasta (o zip costuma criar uma pasta `MachineLearningCVE`).
3. Gere o parquet:

```bash
uv run python make_dataset.py
```

O primeiro `uv run` também cria o ambiente e instala as bibliotecas, o que leva alguns minutos. No final, o script mostra um diagnóstico da base. Confira se aparece `2,830,743 x 80` na linha de linhas e colunas e se o arquivo `data/pacotes.parquet` foi criado.

### 4. Rodar o experimento

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 main.ipynb
```

O comando executa o notebook inteiro e salva as saídas dentro do próprio `main.ipynb`. Durante a execução:

- O terminal não mostra progresso. Ele precisa ficar aberto até o fim.
- O computador não pode suspender. Desative a suspensão automática ou, no macOS, coloque `caffeinate -i` antes do comando.
- O andamento pode ser acompanhado pelo arquivo `resultados/pipelines_n300000.csv`, que ganha uma linha a cada pipeline concluído. No final ele tem 73 linhas (72 pipelines mais o cabeçalho).

Para contar as linhas, em outro terminal dentro da pasta `packoscope`:

```bash
wc -l resultados/pipelines_n300000.csv
```

No Windows (PowerShell):

```powershell
(Get-Content resultados\pipelines_n300000.csv).Count
```

### 5. Conferir o resultado

Abra o `main.ipynb` (por exemplo, com `uv run jupyter lab`) e confira:

- A célula da preparação mostra `linhas: 2,830,743 -> 2,199,994`.
- A verificação de duplicatas mostra `copias do teste no treino: 0` nos 5 folds.
- A tabela completa mostra `72 combinacoes | falhas: 0`.
- O teste de permutação, no final, tem AUC perto de 0,5 nas três linhas.

Como a semente é fixa, as métricas devem ser iguais em qualquer máquina. Os tempos de execução mudam conforme o computador.

### Problemas comuns

| Mensagem ou situação | O que fazer |
|---|---|
| `No pyproject.toml found` ou `Failed to spawn: jupyter` | O terminal está fora da pasta `packoscope`. Entre nela com `cd packoscope`. |
| `nenhum csv encontrado em data/raw` | Os CSV estão em uma subpasta. Mova os 8 arquivos para `data/raw/`. |
| `FileNotFoundError` para `data/pacotes.parquet` | Falta o passo 3. Rode o `make_dataset.py`. |
| O processo é encerrado sozinho ou aparece `MemoryError` | Falta memória. Feche outros programas e rode de novo. |
| A execução foi interrompida no meio | Rode o mesmo comando do passo 4. Os pipelines já gravados em `resultados/` são pulados. |
| É preciso refazer tudo do zero | Apague `resultados/pipelines_n300000.csv` e rode o passo 4 de novo. |

### Depois de rodar

O `main.ipynb` executado e o arquivo `resultados/pipelines_n300000.csv` entram no Git. Combine com a equipe quem faz o commit, porque duas pessoas commitando o mesmo notebook executado geram conflito.
