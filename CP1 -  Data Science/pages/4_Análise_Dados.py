import os, sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
    
from utils import (
    carregar_dados, classificar_variaveis,
    ic_media, t_test_duas_amostras, anova_oneway_por_industria,
    media_ic_por_grupo, crescimento_vagas
)

st.set_page_config(page_title="Análise de Dados", layout="wide")
st.title("Análise de Dados — AI Job Trends")

df = carregar_dados()

st.markdown("""
🔎 Observei que a coluna **Job Status** possui valores ausentes.  
Para manter a consistência dos dados e não perder registros, esses valores foram substituídos por **'Não Informado'**.
""")

st.subheader("Visão Geral do Dataset")
st.write(f"**Dimensão:** {df.shape[0]} linhas × {df.shape[1]} colunas")
st.dataframe(df.head(10), use_container_width=True)

st.header("1) Apresentação dos dados e tipos de variáveis")

tipos_df = classificar_variaveis(df)
st.subheader("🔎 Classificação (Nominal/Ordinal/Discreta/Contínua)")
st.dataframe(tipos_df, use_container_width=True)

st.markdown("""
**Perguntas de análise que responderemos:**
1. O **risco de automação** afeta os **salários**?
2. Existem **diferenças salariais entre indústrias**?
3. Como o **nível de impacto da IA** se distribui por setor e **faixa salarial**?
4. As **vagas projetadas para 2030** crescem em relação a 2024? Em quais indústrias?
5. **Diversidade de gênero** e **trabalho remoto** se relacionam?
""")

st.header("2. Medidas Centrais e Análise Inicial")

st.subheader("Variável escolhida: Experience Required (Years)")

exp = df["Experience Required (Years)"].dropna()

media = exp.mean()
mediana = exp.median()
moda = exp.mode()[0] if not exp.mode().empty else "N/A"
desvio = exp.std()
variancia = exp.var()

st.write(f"**Média:** {media:.2f} anos")
st.write(f"**Mediana:** {mediana:.2f} anos")
st.write(f"**Moda:** {moda} anos")
st.write(f"**Desvio Padrão:** {desvio:.2f}")
st.write(f"**Variância:** {variancia:.2f}")

st.markdown("""
➡️ Observamos que a média e a mediana são próximas, indicando uma distribuição relativamente **simétrica**.  
➡️ O desvio padrão mostra a **dispersão de anos de experiência exigida**, com algumas vagas pedindo bem mais do que a maioria.  
""")

st.subheader("📈 Relação entre Experiência e Salário")

if "Salary Range (Converted USD)" in df.columns:
    exp_sal = df[["Experience Required (Years)", "Salary Range (Converted USD)"]].dropna()

    corr = exp_sal.corr().iloc[0,1]

    st.write(f"Coeficiente de correlação: **{corr:.2f}**")

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=exp_sal,
        x="Experience Required (Years)",
        y="Salary Range (Converted USD)",
        alpha=0.6,
        ax=ax
    )
    sns.regplot(
        data=exp_sal,
        x="Experience Required (Years)",
        y="Salary Range (Converted USD)",
        scatter=False,
        color="red",
        ax=ax
    )
    ax.set_title("Experiência vs Salário (USD)")
    st.pyplot(fig)

    st.markdown("""
    ➡️ Existe uma **correlação positiva**: vagas que exigem mais experiência tendem a oferecer **salários mais altos**.  
    Contudo, a relação não é perfeita, mostrando que outros fatores (como área de atuação, setor e localidade) também impactam.
    """)

st.subheader("P1) O risco de automação afeta os salários?")

df_temp = df[["Automation Risk (%)", "Median Salary (USD)"]].dropna()

df_temp["Faixa de Risco"] = pd.cut(
    df_temp["Automation Risk (%)"],
    bins=[0, 25, 50, 75, 100],
    labels=["Baixo (0-25%)", "Médio (26-50%)", "Alto (51-75%)", "Muito Alto (76-100%)"]
)

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(
    data=df_temp,
    x="Faixa de Risco",
    y="Median Salary (USD)",
    ax=ax
)
ax.set_title("Salários por Faixa de Risco de Automação")
ax.set_xlabel("Faixa de Risco de Automação")
ax.set_ylabel("Salário (USD)")
st.pyplot(fig)

st.markdown("""
➡️ O boxplot mostra a variação dos salários medianos de acordo com a faixa de risco de automação.
Profissões com alto risco de automação tendem a apresentar salários mais baixos e menos dispersos, refletindo menor valorização de mercado.  
""")


st.header("P2) Existem diferenças salariais entre indústrias?")
if "Industry" in df and "Median Salary (USD)" in df:
    agg = media_ic_por_grupo(df, "Industry", "Median Salary (USD)")
    if not agg.empty:
        fig_bar = px.bar(agg, x="Grupo", y="Média", error_y="IC_95", title="Média salarial por Indústria (com IC95%)")
        fig_bar.update_layout(xaxis_title="Indústria", yaxis_title="Salário médio (USD)")
        st.plotly_chart(fig_bar, use_container_width=True)

        an = anova_oneway_por_industria(df, "Median Salary (USD)", min_n=100)
        if an:
            st.markdown(
                f"""
**Teste global (ANOVA)** para diferenças de média salarial entre indústrias com **n≥100**:  
F = {an['F']:.2f}, p = {an['p']:.4f}  
**Interpretação:** p {"< 0.05 ⇒ há" if an['p']<0.05 else "≥ 0.05 ⇒ não há"} evidências de diferenças **significativas** entre as médias salariais dos grupos analisados.
"""
            )
        st.caption("Nota: ICs calculados por t-Student; barras de erro representam ±IC95%.")
    else:
        st.warning("Não há indústrias com amostra suficiente para cálculo de IC.")
else:
    st.warning("Colunas necessárias não encontradas.")

st.header("P3) Como o nível de impacto da IA se distribui por setor e faixa salarial?")
if set(["Industry","AI Impact Level","Median Salary (USD)"]).issubset(df.columns):
    col1, col2 = st.columns([1.2,1])
    with col1:
        fig_stk = px.histogram(df, x="Industry", color="AI Impact Level", barmode="stack", title="Distribuição de Nível de Impacto por Indústria")
        fig_stk.update_layout(xaxis_title="Indústria", yaxis_title="Contagem")
        st.plotly_chart(fig_stk, use_container_width=True)
    with col2:
        fig_box = px.box(df, x="AI Impact Level", y="Median Salary (USD)", color="AI Impact Level", title="Salário por Nível de Impacto")
        st.plotly_chart(fig_box, use_container_width=True)

    med_por_impacto = df.groupby("AI Impact Level")["Median Salary (USD)"].median().dropna()
    st.markdown("**Mediana salarial por nível de impacto:**")
    st.table(med_por_impacto.to_frame("Mediana (USD)"))
    st.markdown(
        "**Interpretação:** setores com **impacto High** tendem a apresentar **faixas salariais diferentes** dos níveis Low/Moderate; a distribuição por indústria mostra onde o impacto está mais concentrado."
    )

st.header("P4) As vagas projetadas para 2030 crescem em relação a 2024? Em quais indústrias?")
if set(["Job Openings (2024)","Projected Openings (2030)"]).issubset(df.columns):
    total_24 = df["Job Openings (2024)"].sum()
    total_30 = df["Projected Openings (2030)"].sum()
    delta = (total_30 - total_24) / total_24 * 100 if total_24 else float("nan")
    st.markdown(f"**Total de vagas 2024:** {total_24:,.0f} | **Total projetado 2030:** {total_30:,.0f} | **Variação:** {delta:.2f}%")

    cres = crescimento_vagas(df)
    if not cres.empty:
        top = cres.head(15)
        fig_cres = px.bar(top, x="Industry", y="Crescimento (%)", title="Top 15 indústrias por crescimento projetado de vagas (%)")
        st.plotly_chart(fig_cres, use_container_width=True)
        st.caption("Interpretação: valores positivos sugerem **expansão** de oportunidades até 2030; negativos indicam **contração**.")
else:
    st.warning("Colunas de vagas não encontradas.")

st.subheader("P5) Diversidade de gênero e trabalho remoto se relacionam?")

df_temp = df[["Gender Diversity (%)", "Remote Work Ratio (%)"]].dropna()

df_temp["Faixa Diversidade"] = pd.cut(
    df_temp["Gender Diversity (%)"],
    bins=[0, 25, 50, 75, 100],
    labels=["Baixa (0-25%)", "Moderada (26-50%)", "Alta (51-75%)", "Muito Alta (76-100%)"]
)

agrupado = df_temp.groupby("Faixa Diversidade")['Remote Work Ratio (%)'].mean().reset_index()

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(
    data=agrupado,
    x="Faixa Diversidade",
    y="Remote Work Ratio (%)",
    ax=ax
)
ax.set_title("Média de Trabalho Remoto por Faixa de Diversidade de Gênero")
ax.set_xlabel("Faixa de Diversidade de Gênero")
ax.set_ylabel("% de Trabalho Remoto")
st.pyplot(fig)

st.markdown("""
➡️ O gráfico evidencia como a diversidade de gênero varia entre empresas com diferentes proporções de trabalho remoto.

Em faixas de baixo trabalho remoto (0–25%), observa-se maior dispersão, sugerindo que a diversidade depende mais de fatores organizacionais específicos do que da modalidade de trabalho.
""")

st.header("4) Intervalos de Confiança e Testes de Hipótese")

if "Median Salary (USD)" in df:
    media, li, ls = ic_media(df["Median Salary (USD)"])
    st.subheader("IC da média salarial (global)")
    st.markdown(f"**Média** = {media:,.2f} USD | **IC95%** = [{li:,.2f} ; {ls:,.2f}]")
    st.caption("Justificativa: variável contínua; amostra grande; uso de **t-Student** para estimar o IC da média.")

st.subheader("Teste de Hipótese — Salários: IT vs Finance")
if "Industry" in df and "Median Salary (USD)" in df:
    if df["Industry"].isin(["IT","Finance"]).any():
        it = df.loc[df["Industry"]=="IT","Median Salary (USD)"]
        fin = df.loc[df["Industry"]=="Finance","Median Salary (USD)"]
        teste = t_test_duas_amostras(it, fin)
        if teste:
            st.markdown(
                f"""
**H0:** μ_IT = μ_Finance  |  **H1:** μ_IT ≠ μ_Finance  
**t({len(it.dropna())+len(fin.dropna())-2} aprox.)** = {teste['t']:.3f}, **p** = {teste['p']:.4f}  
**Médias:** IT = {teste['mean1']:,.2f} USD | Finance = {teste['mean2']:,.2f} USD  
**Tamanho de efeito (Cohen's d, Welch):** {teste['cohens_d']:.2f}
"""
            )
            if teste["p"] < 0.05:
                st.success("Conclusão: **rejeitamos H0** (5%) — evidência de diferença salarial entre IT e Finance.")
            else:
                st.info("Conclusão: **não rejeitamos H0** (5%) — não há evidência de diferença salarial.")
        else:
            st.warning("Amostras insuficientes para o teste t.")
    else:
        st.warning("Indústrias 'IT' e 'Finance' não encontradas no dataset.")

if set(["Industry","Median Salary (USD)"]).issubset(df.columns):
    sub = df[df["Industry"].isin(["IT","Finance"])]
    if not sub.empty:
        fig_box = px.box(sub, x="Industry", y="Median Salary (USD)", color="Industry", title="Salários — IT vs Finance")
        st.plotly_chart(fig_box, use_container_width=True)

st.caption("""
**Justificativas dos métodos:**  
- **IC da média (t-Student):** variável contínua, amostra grande; estimativa do parâmetro populacional (média).  
- **Teste t de duas amostras (Welch):** comparação de médias entre dois grupos independentes (IT vs Finance) sem assumir variâncias iguais.  
- **ANOVA one-way:** comparação global de mais de dois grupos (indústrias) para verificar se ao menos uma média difere.  
- **Correlação (Pearson/Spearman):** quantifica relação linear/monótona entre variáveis numéricas.
""")
