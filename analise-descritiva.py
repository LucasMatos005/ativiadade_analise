import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Carregar dados
df = pd.read_csv('vendas_rede_varejo.csv', sep=';', decimal=',')

# Converter data para datetime
df['data_venda'] = pd.to_datetime(df['data_venda'])

# Limpar nomes das colunas (remover espaços)
df.columns = df.columns.str.strip()

print("="*60)
print("ANÁLISE DE VENDAS - REDE DE VAREJO")
print("="*60)

# 1. FATURAMENTO MÉDIO DIÁRIO
print("\n1. FATURAMENTO MÉDIO DIÁRIO")
print("-"*60)
faturamento_diario = df.groupby('data_venda')['valor_total'].sum()
media_diaria = faturamento_diario.mean()
print(f"Faturamento médio diário: R$ {media_diaria:,.2f}")
print(f"Faturamento total: R$ {df['valor_total'].sum():,.2f}")
print(f"Número de dias com vendas: {len(faturamento_diario)}")

# 2. GRÁFICO: DISTRIBUIÇÃO POR CANAL DE VENDA
print("\n2. Gerando gráfico de distribuição por canal de venda...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Canal de venda - Quantidade
canal_vendas = df.groupby('canal_venda').agg({
    'valor_total': 'sum',
    'quantidade': 'sum'
}).sort_values('valor_total', ascending=False)

axes[0, 0].bar(canal_vendas.index, canal_vendas['valor_total'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 0].set_title('Faturamento por Canal de Venda', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Canal de Venda')
axes[0, 0].set_ylabel('Valor Total (R$)')
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(canal_vendas['valor_total']):
    axes[0, 0].text(i, v, f'R${v/1000:.0f}K', ha='center', va='bottom')

# 3. GRÁFICO: DISTRIBUIÇÃO POR REGIÃO
print("3. Gerando gráfico de distribuição por região...")
regiao_vendas = df.groupby('regiao')['valor_total'].sum().sort_values(ascending=False)

axes[0, 1].barh(regiao_vendas.index, regiao_vendas.values, color=['#95E1D3', '#F38181', '#EAFFD0', '#FCE38A', '#AA96DA'])
axes[0, 1].set_title('Faturamento por Região', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Valor Total (R$)')
axes[0, 1].set_ylabel('Região')
for i, v in enumerate(regiao_vendas.values):
    axes[0, 1].text(v, i, f' R${v/1000:.0f}K', va='center')

# 4. CATEGORIAS MAIS VENDIDAS
print("\n4. CATEGORIAS MAIS VENDIDAS")
print("-"*60)
categorias = df.groupby('categoria_produto').agg({
    'valor_total': 'sum',
    'quantidade': 'sum',
    'custo_total': 'sum'
}).sort_values('valor_total', ascending=False)

categorias['margem_lucro'] = ((categorias['valor_total'] - categorias['custo_total']) / categorias['valor_total'] * 100)

print(categorias)
print(f"\nMargem de lucro média geral: {categorias['margem_lucro'].mean():.2f}%")

# Gráfico de categorias
axes[1, 0].bar(categorias.index, categorias['valor_total'], color=['#667BC6', '#DA7297', '#FADA7A', '#82A0D8', '#C4E1F6'])
axes[1, 0].set_title('Faturamento por Categoria de Produto', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Categoria')
axes[1, 0].set_ylabel('Valor Total (R$)')
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(categorias['valor_total']):
    axes[1, 0].text(i, v, f'R${v/1000:.0f}K', ha='center', va='bottom', fontsize=8)

# 5. HISTOGRAMA DE SATISFAÇÃO DO CLIENTE
print("\n5. Gerando histograma de satisfação do cliente...")
axes[1, 1].hist(df['satisfacao_cliente'], bins=20, color='#6C5CE7', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Distribuição de Satisfação do Cliente', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Nota de Satisfação')
axes[1, 1].set_ylabel('Frequência')
axes[1, 1].axvline(df['satisfacao_cliente'].mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {df["satisfacao_cliente"].mean():.2f}')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analise_vendas.png', dpi=300, bbox_inches='tight')
print("\nGráficos salvos em 'analise_vendas.png'")

# ESTATÍSTICAS ADICIONAIS
print("\n" + "="*60)
print("ESTATÍSTICAS ADICIONAIS")
print("="*60)
print(f"\nSatisfação média dos clientes: {df['satisfacao_cliente'].mean():.2f}")
print(f"Ticket médio: R$ {df['valor_total'].mean():,.2f}")
print(f"Total de transações: {len(df)}")
print(f"Produto mais vendido: {df.groupby('categoria_produto')['quantidade'].sum().idxmax()}")

# Margem de lucro por categoria
print("\n" + "-"*60)
print("MARGEM DE LUCRO POR CATEGORIA:")
print("-"*60)
for cat, margem in categorias['margem_lucro'].items():
    print(f"{cat:20s}: {margem:6.2f}%")

plt.show()

print("\n" + "="*60)
print("ANÁLISE CONCLUÍDA!")
print("="*60)