import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Carregar dados
df = pd.read_csv('vendas_rede_varejo.csv', sep=';', decimal=',')
df['data_venda'] = pd.to_datetime(df['data_venda'])
df.columns = df.columns.str.strip()

print("="*80)
print("ANÁLISE PREDITIVA DE VENDAS")
print("="*80)

# ============================================================================
# 1. PREPARAÇÃO E MODELAGEM
# ============================================================================
print("\n" + "="*80)
print("1. PREPARAÇÃO DOS DADOS")
print("="*80)

# Criar variável dummy para campanha
df['tem_campanha'] = (df['campanha'] != 'Nenhuma').astype(int)

# Preparar dados
X = df[['quantidade', 'preco_unitario', 'tem_campanha']]
y = df['valor_total']

# Dividir em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Conjunto de treino: {len(X_train)} registros")
print(f"Conjunto de teste: {len(X_test)} registros")

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("\n" + "="*80)
print("2. COEFICIENTES DO MODELO")
print("="*80)
print(f"Quantidade:      R$ {modelo.coef_[0]:10.2f}")
print(f"Preço Unitário:  R$ {modelo.coef_[1]:10.2f}")
print(f"Campanha:        R$ {modelo.coef_[2]:10.2f}")
print(f"Intercepto:      R$ {modelo.intercept_:10.2f}")

# ============================================================================
# 3. MÉTRICAS DE DESEMPENHO
# ============================================================================
print("\n" + "="*80)
print("3. AVALIAÇÃO DO MODELO")
print("="*80)

# Fazer as previsões
y_pred = modelo.predict(X_test)

# Calcular as métricas
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² (Coef. Determinação): {r2:.4f}")
print(f"MAE (Erro Abs. Médio):   R$ {mae:,.2f}")
print(f"RMSE (Raiz Erro Quad.):  R$ {rmse:,.2f}")
print(f"\nO modelo explica {r2*100:.2f}% da variação no valor total")

# ============================================================================
# 4. VISUALIZAÇÕES
# ============================================================================
print("\n" + "="*80)
print("4. GERANDO GRÁFICOS")
print("="*80)

residuos = y_test - y_pred

# Gráfico 1: Valores Reais vs Previstos
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test, y_pred, alpha=0.6, c='steelblue', edgecolors='black', s=50)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', linewidth=2, label='Previsão Perfeita')
ax.set_xlabel('Valores Reais (R$)', fontsize=12, fontweight='bold')
ax.set_ylabel('Valores Previstos (R$)', fontsize=12, fontweight='bold')
ax.set_title(f'Valores Reais vs Previstos\nR² = {r2:.4f}', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('preditivo_valores_reais_vs_previstos.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 2: Análise de Resíduos
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_pred, residuos, alpha=0.6, c='coral', edgecolors='black', s=50)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Valores Previstos (R$)', fontsize=12, fontweight='bold')
ax.set_ylabel('Resíduos (R$)', fontsize=12, fontweight='bold')
ax.set_title('Análise de Resíduos', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('preditivo_analise_residuos.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 3: Distribuição dos Resíduos
fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(residuos, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax.set_xlabel('Resíduos (R$)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
ax.set_title('Distribuição dos Resíduos', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('preditivo_distribuicao_residuos.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráficos salvos com sucesso")

# ============================================================================
# 5. EXEMPLOS DE PREVISÃO
# ============================================================================
print("\n" + "="*80)
print("5. EXEMPLOS DE PREVISÃO")
print("="*80)

cenarios = pd.DataFrame({
    'quantidade': [10, 15, 20, 15, 20],
    'preco_unitario': [500, 1000, 1500, 1000, 1500],
    'tem_campanha': [0, 0, 0, 1, 1]
})

previsoes = modelo.predict(cenarios)

print(f"{'Qtd':>4} {'Preço':>8} {'Campanha':>10} {'Valor Previsto':>18}")
print("-" * 60)
for i, row in cenarios.iterrows():
    camp = 'Sim' if row['tem_campanha'] == 1 else 'Não'
    print(f"{row['quantidade']:4.0f} R$ {row['preco_unitario']:6.0f} {camp:>10s} R$ {previsoes[i]:14.2f}")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA!")
print("="*80)
print("\nArquivos gerados:")
print("  1. preditivo_valores_reais_vs_previstos.png")
print("  2. preditivo_analise_residuos.png")
print("  3. preditivo_distribuicao_residuos.png")
print("="*80)

plt.show()