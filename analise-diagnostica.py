import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Carregar dados
df = pd.read_csv('vendas_rede_varejo.csv', sep=';', decimal=',')

# Converter data para datetime
df['data_venda'] = pd.to_datetime(df['data_venda'])

# Calcular lucro
df['lucro'] = df['valor_total'] - df['custo_total']
df['margem_lucro'] = (df['lucro'] / df['valor_total']) * 100

print("="*80)
print("ANÁLISE DIAGNÓSTICA DE VENDAS")
print("="*80)
print(f"\nTotal de registros: {len(df)}")
print(f"Período: {df['data_venda'].min().strftime('%d/%m/%Y')} a {df['data_venda'].max().strftime('%d/%m/%Y')}")

# ============================================================================
# 1. ESTATÍSTICAS DESCRITIVAS DO LUCRO
# ============================================================================
print("\n" + "="*80)
print("1. ESTATÍSTICAS DO LUCRO")
print("="*80)
print(f"\nLucro Total: R$ {df['lucro'].sum():,.2f}")
print(f"Lucro Médio por Venda: R$ {df['lucro'].mean():,.2f}")
print(f"Margem de Lucro Média: {df['margem_lucro'].mean():.2f}%")
print(f"Desvio Padrão do Lucro: R$ {df['lucro'].std():,.2f}")

# ============================================================================
# 2. ANÁLISE DE CORRELAÇÃO
# ============================================================================
print("\n" + "="*80)
print("2. MATRIZ DE CORRELAÇÃO - FATORES QUE IMPACTAM O LUCRO")
print("="*80)

# Selecionar variáveis numéricas para correlação
variaveis_correlacao = ['lucro', 'quantidade', 'preco_unitario', 
                        'valor_total', 'custo_total', 'satisfacao_cliente']
df_corr = df[variaveis_correlacao].copy()

# Calcular matriz de correlação
correlacao = df_corr.corr()

# Exibir correlações com lucro
print("\nCorrelação com LUCRO:")
print("-" * 50)
corr_lucro = correlacao['lucro'].sort_values(ascending=False)
for var, valor in corr_lucro.items():
    if var != 'lucro':
        interpretacao = ""
        if abs(valor) > 0.7:
            interpretacao = "FORTE"
        elif abs(valor) > 0.4:
            interpretacao = "MODERADA"
        else:
            interpretacao = "FRACA"
        print(f"{var:25s}: {valor:6.3f}  ({interpretacao})")

# ============================================================================
# 3. MAPA DE CALOR DE CORRELAÇÃO
# ============================================================================
print("\n" + "="*80)
print("3. GERANDO MAPA DE CALOR...")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(correlacao, dtype=bool))
sns.heatmap(correlacao, 
            mask=mask,
            annot=True, 
            fmt='.3f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Mapa de Calor - Correlação entre Variáveis\n(Lucro, Quantidade, Preço e Satisfação)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('mapa_calor_correlacao.png', dpi=300, bbox_inches='tight')
print("✓ Mapa de calor salvo: mapa_calor_correlacao.png")

# ============================================================================
# 4. GRÁFICO DE DISPERSÃO: SATISFAÇÃO × VALOR TOTAL
# ============================================================================
print("\n" + "="*80)
print("4. ANÁLISE: SATISFAÇÃO DO CLIENTE × VALOR TOTAL")
print("="*80)

# Calcular correlação
corr_satisfacao_valor = df['satisfacao_cliente'].corr(df['valor_total'])
print(f"\nCorrelação Satisfação × Valor Total: {corr_satisfacao_valor:.3f}")

# Teste de significância
statistic, p_value = stats.pearsonr(df['satisfacao_cliente'], df['valor_total'])
print(f"P-valor: {p_value:.4f}")
if p_value < 0.05:
    print("✓ Correlação estatisticamente significativa (p < 0.05)")
else:
    print("✗ Correlação NÃO estatisticamente significativa (p ≥ 0.05)")

# Criar gráfico de dispersão
fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(df['satisfacao_cliente'], 
                     df['valor_total'], 
                     c=df['lucro'],
                     cmap='viridis',
                     alpha=0.6,
                     s=50,
                     edgecolors='black',
                     linewidth=0.5)

# Linha de tendência
z = np.polyfit(df['satisfacao_cliente'], df['valor_total'], 1)
p = np.poly1d(z)
ax.plot(df['satisfacao_cliente'].sort_values(), 
        p(df['satisfacao_cliente'].sort_values()), 
        "r--", 
        linewidth=2, 
        label=f'Tendência (r={corr_satisfacao_valor:.3f})')

ax.set_xlabel('Satisfação do Cliente (1-10)', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor Total (R$)', fontsize=12, fontweight='bold')
ax.set_title('Dispersão: Satisfação do Cliente × Valor Total da Venda\n(Cor indica nível de lucro)', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Lucro (R$)', fontsize=11)

plt.tight_layout()
plt.savefig('dispersao_satisfacao_valor.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico de dispersão salvo: dispersao_satisfacao_valor.png")

# ============================================================================
# 5. ANÁLISE POR CANAL DE VENDA
# ============================================================================
print("\n" + "="*80)
print("5. ANÁLISE POR CANAL DE VENDA")
print("="*80)

analise_canal = df.groupby('canal_venda').agg({
    'lucro': ['sum', 'mean'],
    'margem_lucro': 'mean',
    'satisfacao_cliente': 'mean',
    'valor_total': 'sum'
}).round(2)

print("\n", analise_canal)

# Gráfico comparativo
axes = plt.subplots(2, 2, figsize=(14, 10))

# Lucro total por canal
df.groupby('canal_venda')['lucro'].sum().plot(kind='bar', ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Lucro Total por Canal', fontweight='bold')
axes[0, 0].set_ylabel('Lucro (R$)')
axes[0, 0].tick_params(axis='x', rotation=45)fig, 

# Margem de lucro por canal
df.groupby('canal_venda')['margem_lucro'].mean().plot(kind='bar', ax=axes[0, 1], color='coral')
axes[0, 1].set_title('Margem de Lucro Média por Canal (%)', fontweight='bold')
axes[0, 1].set_ylabel('Margem (%)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Satisfação por canal
df.groupby('canal_venda')['satisfacao_cliente'].mean().plot(kind='bar', ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Satisfação Média por Canal', fontweight='bold')
axes[1, 0].set_ylabel('Satisfação (1-10)')
axes[1, 0].axhline(y=df['satisfacao_cliente'].mean(), color='r', linestyle='--', label='Média Geral')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', rotation=45)

# Lucro vs Satisfação por canal
for canal in df['canal_venda'].unique():
    dados_canal = df[df['canal_venda'] == canal]
    axes[1, 1].scatter(dados_canal['satisfacao_cliente'], 
                       dados_canal['lucro'],
                       label=canal,
                       alpha=0.6,
                       s=30)
axes[1, 1].set_xlabel('Satisfação do Cliente')
axes[1, 1].set_ylabel('Lucro (R$)')
axes[1, 1].set_title('Lucro × Satisfação por Canal', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analise_canais.png', dpi=300, bbox_inches='tight')
print("\n✓ Análise por canais salva: analise_canais.png")

# ============================================================================
# 6. ANÁLISE POR CATEGORIA
# ============================================================================
print("\n" + "="*80)
print("6. ANÁLISE POR CATEGORIA DE PRODUTO")
print("="*80)

analise_categoria = df.groupby('categoria_produto').agg({
    'lucro': ['sum', 'mean'],
    'margem_lucro': 'mean',
    'satisfacao_cliente': 'mean',
    'quantidade': 'sum'
}).round(2)

print("\n", analise_categoria)

# ============================================================================
# 7. ANÁLISE POR CAMPANHA
# ============================================================================
print("\n" + "="*80)
print("7. IMPACTO DAS CAMPANHAS NO LUCRO E SATISFAÇÃO")
print("="*80)

analise_campanha = df.groupby('campanha').agg({
    'lucro': ['sum', 'mean'],
    'satisfacao_cliente': 'mean',
    'valor_total': 'sum'
}).round(2)

print("\n", analise_campanha.sort_values(('lucro', 'sum'), ascending=False))

# ============================================================================
# 8. INSIGHTS E RECOMENDAÇÕES
# ============================================================================
print("\n" + "="*80)
print("8. PRINCIPAIS INSIGHTS")
print("="*80)

# Melhor e pior canal
melhor_canal = df.groupby('canal_venda')['lucro'].sum().idxmax()
pior_canal = df.groupby('canal_venda')['lucro'].sum().idxmin()

# Categoria mais lucrativa
melhor_categoria = df.groupby('categoria_produto')['lucro'].sum().idxmax()

# Região mais lucrativa
melhor_regiao = df.groupby('regiao')['lucro'].sum().idxmax()

print(f"\n✓ Canal mais lucrativo: {melhor_canal}")
print(f"✓ Categoria mais lucrativa: {melhor_categoria}")
print(f"✓ Região mais lucrativa: {melhor_regiao}")
print(f"\n✓ Correlação Satisfação × Valor: {corr_satisfacao_valor:.3f}")
print(f"  → {'Positiva' if corr_satisfacao_valor > 0 else 'Negativa'}, "
      f"{'fraca' if abs(corr_satisfacao_valor) < 0.3 else 'moderada' if abs(corr_satisfacao_valor) < 0.7 else 'forte'}")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA!")
print("="*80)
print("\nArquivos gerados:")
print("  1. mapa_calor_correlacao.png")
print("  2. dispersao_satisfacao_valor.png")
print("  3. analise_canais.png")
print("="*80)