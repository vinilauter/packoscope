import argparse
import codecs
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# caminho relativo a este arquivo, nao ao cwd de quem chama o script
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "pacotes.parquet"
SAMPLE_DIR = ROOT / "data" / "sample"

LABEL = "Label"
SOURCE = "source_file"  # metadado: veio de qual csv. nunca usar como feature
SEED = 42

# variantes do travessao quebrado que aparecem no rotulo "web attack", a
# depender de qual copia da base cada um baixou. todas viram hifen ascii
TRAVESSOES = ("�", "\x96", "–", "—")


def curto(caminho):
    # relativo a raiz do repo quando da, absoluto quando nao da (ex: --raw-dir
    # apontando pra fora do repo)
    caminho = Path(caminho).resolve()
    try:
        return caminho.relative_to(ROOT)
    except ValueError:
        return caminho


def detectar_encoding(caminho, bloco=1 << 20):
    # o read_csv do pandas nao levanta erro com byte invalido, mesmo com
    # encoding_errors='strict' -- ele troca por � e segue. entao decodifica
    # os bytes crus na mao pra pegar isso antes de ler
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    with open(caminho, "rb") as fh:
        while pedaco := fh.read(bloco):
            try:
                decoder.decode(pedaco)
            except UnicodeDecodeError:
                return "latin-1"
    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return "latin-1"
    return "utf-8"


def normalizar_rotulos(serie):
    # junta as variantes do rotulo num texto so, devolve tambem quantas linhas mudaram
    limpa = serie.astype("string").str.strip()
    for travessao in TRAVESSOES:
        limpa = limpa.str.replace(travessao, "-", regex=False)
    limpa = limpa.str.replace(r"\s+", " ", regex=True)
    mudou = int((limpa != serie.astype("string")).sum())
    return limpa, mudou


def preparar(df, origem):
    # arruma nomes, rotulo e tipos de um arquivo por vez, antes do concat --
    # reduzir o tipo aqui corta o pico de memoria quase pela metade
    df.columns = df.columns.str.strip()  # sem isso ' Label' != 'Label' e da KeyError

    df[LABEL], mudou = normalizar_rotulos(df[LABEL])

    # float32 so serve se o maior valor FINITO couber. inf fica de fora da conta
    # de proposito, ele sobrevive a conversao porque e achado, nao estouro
    floats = df.select_dtypes(include="float64").columns
    if len(floats):
        maior = df[floats].replace([np.inf, -np.inf], np.nan).abs().max().max()
        limite = np.finfo("float32").max
        if maior > limite:
            sys.exit(
                f"{origem}: float32 nao serve, maior finito e {maior:.4g} > {limite:.4g}"
            )
        df[floats] = df[floats].astype("float32")

    # nao forcar int32: algumas colunas de header length tem minimo na casa de
    # -3,2e10, que estoura int32 e volta com o sinal trocado -- apagando o
    # "valor fora de dominio" que a entrega 1 precisa reportar. o downcast
    # abaixo escolhe o menor tipo que couber, coluna a coluna, sem perda
    for col in df.select_dtypes(include="int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df.assign(**{SOURCE: origem}), mudou


def carregar(raw_dir):
    arquivos = sorted(raw_dir.glob("*.csv"))  # sorted = ordem estavel entre maquinas
    if not arquivos:
        sys.exit(
            f"nenhum csv encontrado em {raw_dir}\n"
            "baixe a base do site do cic/unb e descompacte ali."
        )

    print(f"lendo {len(arquivos)} arquivos de {curto(raw_dir)}/")
    partes = []
    for caminho in arquivos:
        encoding = detectar_encoding(caminho)
        df = pd.read_csv(caminho, encoding=encoding)
        df, mudou = preparar(df, caminho.stem)

        avisos = []
        if encoding != "utf-8":
            avisos.append(f"encoding {encoding}")
        if mudou:
            avisos.append(f"{mudou:,} rotulos normalizados")
        sufixo = ("   <- " + ", ".join(avisos)) if avisos else ""
        print(f"  {caminho.name:<52} {len(df):>9,} linhas{sufixo}")
        partes.append(df)

    # concat alinha por nome de coluna e nao reclama de divergencia -- um
    # arquivo com coluna a mais gera uma coluna cheia de nulo, entao confere antes
    referencia = set(partes[0].columns)
    for caminho, parte in zip(arquivos, partes):
        faltando = referencia - set(parte.columns)
        sobrando = set(parte.columns) - referencia
        if faltando or sobrando:
            sys.exit(
                f"{caminho.name} tem colunas diferentes do primeiro arquivo\n"
                f"  faltando: {sorted(faltando)}\n  sobrando: {sorted(sobrando)}"
            )

    # o strip pode colidir dois nomes que so diferiam pelo espaco
    repetidas = partes[0].columns[partes[0].columns.duplicated()].tolist()
    if repetidas:
        sys.exit(f"nomes de coluna repetidos depois do strip: {repetidas}")

    df = pd.concat(partes, ignore_index=True)
    partes.clear()  # libera as 8 copias antes de seguir

    df[LABEL] = df[LABEL].astype("category")
    df[SOURCE] = df[SOURCE].astype("category")
    return df


def diagnosticar(df):
    # so conta os problemas conhecidos, nao corrige nenhum
    numericas = df.select_dtypes(include="number")

    def infinitos(col):
        return np.isinf(numericas[col].to_numpy(dtype="float64", na_value=np.nan))

    col_inf = [c for c in numericas.columns if infinitos(c).any()]
    n_inf = sum(int(infinitos(c).sum()) for c in col_inf)
    n_nan = int(df.isna().to_numpy().sum())

    # duplicata ignorando a origem: e assim que o modelo vai ver as linhas
    features = [c for c in df.columns if c != SOURCE]
    n_dup = int(df.duplicated(subset=features).sum())

    constantes = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    negativas = [c for c in numericas.columns if numericas[c].min() < 0]

    print("\n" + "=" * 64)
    print("diagnostico -- nada abaixo foi corrigido, tudo continua no arquivo")
    print("=" * 64)
    print(f"  linhas x colunas          {df.shape[0]:,} x {df.shape[1]}")
    print(
        f"  memoria                   {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
    )
    print(f"  valores infinitos         {n_inf:,}  em {col_inf}")
    print(f"  valores ausentes (NaN)    {n_nan:,}")
    print(
        f"  linhas duplicadas         {n_dup:,}  ({n_dup / len(df):.1%}, ignorando {SOURCE})"
    )
    print(f"  colunas constantes        {len(constantes)}")
    for col in constantes:
        print(f"                              {col}")
    print(f"  colunas com min negativo  {len(negativas)}")
    print(f"\n  classes ({df[LABEL].nunique()}):")
    for classe, n in df[LABEL].value_counts().items():
        alerta = "   <- rara demais p/ 10 folds" if n < 100 else ""
        print(f"    {n:>9,}  {classe}{alerta}")
    print("=" * 64)


def gravar_amostra(df, n_por_classe, destino):
    # amostra estratificada e versionavel, pro notebook rodar sem a base inteira
    pedacos = [
        grupo.sample(n=min(len(grupo), n_por_classe), random_state=SEED)
        for _, grupo in df.groupby(LABEL, observed=True)
    ]
    amostra = pd.concat(pedacos).sort_index()
    destino.parent.mkdir(parents=True, exist_ok=True)
    amostra.to_parquet(destino, engine="pyarrow", compression="snappy", index=False)
    print(
        f"\namostra: {len(amostra):,} linhas, {amostra[LABEL].nunique()} classes"
        f" -> {curto(destino)} ({destino.stat().st_size / 1024**2:.1f} MB)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="consolida os csvs da cic-ids-2017 em um parquet."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR, help="pasta com os csvs"
    )
    parser.add_argument("--out", type=Path, default=OUT_FILE, help="parquet de saida")
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="grava tambem data/sample/amostra.parquet com ate n linhas por classe",
    )
    args = parser.parse_args()

    df = carregar(args.raw_dir)
    diagnosticar(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", compression="snappy", index=False)
    print(f"\ngravado: {curto(args.out)} ({args.out.stat().st_size / 1024**2:.1f} MB)")

    if args.sample:
        gravar_amostra(df, args.sample, SAMPLE_DIR / "amostra.parquet")


if __name__ == "__main__":
    main()
