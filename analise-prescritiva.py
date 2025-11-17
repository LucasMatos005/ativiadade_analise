import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Carregar dados
df = pd.read_csv('vendas_rede_varejo.csv', sep=';', decimal=',')
df['data_venda'] = pd.to_datetime(df['data_venda'])
df.columns = df.columns.str.strip()

# Calcular lucro
df['lucro'] = df['valor_total'] - df['custo_total']
df['margem_lucro'] = (df['lucro'] / df['valor_total']) * 100

print("="*80)
print("ANÁLISE PRESCRITIVA DE VENDAS")
print("="*80)

# ============================================================================
# 1. CENÁRIO ATUAL
# ============================================================================
print("\n" + "="*80)
print("1. CENÁRIO ATUAL")
print("="*80)
print(f"Faturamento Total:     R$ {df['valor_total'].sum():,.2f}")
print(f"Lucro Total:           R$ {df['lucro'].sum():,.2f}")
print(f"Margem de Lucro Média: {df['margem_lucro'].mean():.2f}%")

# ============================================================================
# 2. MODELO PREDITIVO
# ============================================================================
print("\n" + "="*80)
print("2. TREINAMENTO DO MODELO")
print("="*80)

df['tem_campanha'] = (df['campanha'] != 'Nenhuma').astype(int)

X = df[['quantidade', 'preco_unitario', 'tem_campanha']]
y = df['valor_total']

modelo = LinearRegression()
modelo.fit(X, y)

print(f"✓ Modelo treinado (R² = {modelo.score(X, y):.4f})")

# ============================================================================
# 3. SIMULAÇÃO DE CENÁRIOS
# ============================================================================
print("\n" + "="*80)
print("3. SIMULAÇÃO: MATRIZ QUANTIDADE × PREÇO")
print("="*80)

# Criar grade de simulação
qtd_range = np.linspace(5, 30, 20)
preco_range = np.linspace(100, 3000, 20)
matriz_lucro = np.zeros((len(preco_range), len(qtd_range)))

for i, preco in enumerate(preco_range):
    for j, qtd in enumerate(qtd_range):
        entrada = np.array([[qtd, preco, 1]])
        valor_previsto = modelo.predict(entrada)[0]
        custo_estimado = qtd * preco * 0.6
        lucro = valor_previsto - custo_estimado
        matriz_lucro[i, j] = lucro

# Encontrar combinação ótima
idx_max = np.unravel_index(matriz_lucro.argmax(), matriz_lucro.shape)
preco_otimo = preco_range[idx_max[0]]
qtd_otima = qtd_range[idx_max[1]]
lucro_maximo = matriz_lucro[idx_max]

print(f"Combinação Ótima:")
print(f"  • Quantidade:     {qtd_otima:.0f} unidades")
print(f"  • Preço:          R$ {preco_otimo:.2f}")
print(f"  • Lucro Previsto: R$ {lucro_maximo:.2f}")

# ============================================================================
# 4. VISUALIZAÇÃO
# ============================================================================
print("\n" + "="*80)
print("4. GERANDO GRÁFICO")
print("="*80)

# Heatmap - Matriz Quantidade × Preço (usando imshow)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(matriz_lucro, cmap='RdYlGn', aspect='auto', origin='lower',
               extent=[qtd_range.min(), qtd_range.max(), preco_range.min(), preco_range.max()])
ax.scatter(qtd_otima, preco_otimo, color='blue', s=300, marker='*', 
           edgecolors='white', linewidth=3, label='Ponto Ótimo', zorder=5)
ax.set_xlabel('Quantidade', fontsize=13, fontweight='bold')
ax.set_ylabel('Preço Unitário (R$)', fontsize=13, fontweight='bold')
ax.set_title('Matriz de Lucro: Quantidade × Preço\n(Cenário Ótimo Destacado)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=12)
cbar = plt.colorbar(im, ax=ax, label='Lucro (R$)')
cbar.ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('prescritivo_matriz_quantidade_preco.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráfico salvo com sucesso")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA!")
print("="*80)
print("\nArquivo gerado:")
print("  • prescritivo_matriz_quantidade_preco.png")
print("="*80)

plt.show()