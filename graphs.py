
"""# 3. Visualizations"""

#dependency installs again
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openpyxl
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Load data
out_file="Milton_preprocessed_with_predictions.xlsx"
df = pd.read_excel(out_file)
print(f"Using this file for visualizations: {out_file}")

# Function to extract individual labels from combined entries
def expand_labels(df, column_name):
    """Expand multi-label entries into separate rows for each label"""
    expanded_rows = []
    for idx, row in df.iterrows():
        labels = str(row[column_name]).split(',')
        labels = [label.strip() for label in labels]
        for label in labels:
            new_row = row.copy()
            new_row[column_name + '_expanded'] = label
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

# Create expanded dataset
df_expanded = expand_labels(df, 'predicted_crosscheck')

# Determine location column
location_col = 'County Name' if 'County Name' in df.columns and df['County Name'].notna().sum() > 0 else 'Franchise Name'

# ============================================================
# FIGURE 1: predicted_crosscheck ANALYSIS (AS ENTITIES)
# ============================================================
predicted_crosscheck_counts = df['predicted_crosscheck'].value_counts()
fig1 = plt.figure(figsize=(20, 12))

# 1. predicted_crosscheck Distribution Pie (Top 8 + Others)
ax1 = plt.subplot(2, 3, 1)
top_n = 7
if len(predicted_crosscheck_counts) > top_n:
    top_categories = predicted_crosscheck_counts.head(top_n)
    other_count = predicted_crosscheck_counts.iloc[top_n:].sum()
    plot_data = pd.concat([top_categories, pd.Series({'others': other_count})])
else:
    plot_data = predicted_crosscheck_counts

colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
wedges, texts, autotexts = ax1.pie(plot_data, labels=None, autopct='%1.1f%%',
                                     colors=colors, startangle=90, pctdistance=0.85)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')
ax1.set_title('predicted_crosscheck Distribution\n(As Single Entity)', fontsize=14, fontweight='bold', pad=20)
ax1.legend(plot_data.index, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)

# 2. Top 10 predicted_crosscheck Categories (Horizontal Bar)
ax2 = plt.subplot(2, 3, 2)
top10_predicted_crosscheck = predicted_crosscheck_counts.head(10)
bars = ax2.barh(range(len(top10_predicted_crosscheck)), top10_predicted_crosscheck.values, color=plt.cm.Spectral(np.linspace(0, 1, 10)))
ax2.set_yticks(range(len(top10_predicted_crosscheck)))
ax2.set_yticklabels(top10_predicted_crosscheck.index, fontsize=10)
ax2.set_xlabel('Count', fontsize=11, fontweight='bold')
ax2.set_title('Top 10 predicted_crosscheck Categories\n(As Single Entity)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, top10_predicted_crosscheck.values)):
    ax2.text(val, i, f' {val:,}', va='center', fontweight='bold', fontsize=9)

# 3. Customers Affected by Top predicted_crosscheck Categories
ax3 = plt.subplot(2, 3, 3)
cust_by_predicted_crosscheck = df.groupby('predicted_crosscheck')['CI'].sum().sort_values(ascending=False).head(10)
bars = ax3.barh(range(len(cust_by_predicted_crosscheck)), cust_by_predicted_crosscheck.values,
                color=plt.cm.viridis(np.linspace(0, 1, 10)))
ax3.set_yticks(range(len(cust_by_predicted_crosscheck)))
ax3.set_yticklabels(cust_by_predicted_crosscheck.index, fontsize=10)
ax3.set_xlabel('Customers Affected', fontsize=11, fontweight='bold')
ax3.set_title('Top 10: Customers Affected\n(As Single Entity)', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, cust_by_predicted_crosscheck.values)):
    ax3.text(val, i, f' {val:,.0f}', va='center', fontweight='bold', fontsize=9)

# 5. Heatmap: Top 10 Locations vs Top 5 predicted_crosscheck
ax5 = plt.subplot(2, 3, 5)
top_locations = df[location_col].value_counts().head(10).index
top_predicted_crosschecks = df['predicted_crosscheck'].value_counts().head(5).index
pivot_data = df[df[location_col].isin(top_locations) & df['predicted_crosscheck'].isin(top_predicted_crosschecks)].pivot_table(
    values='CI', index=location_col, columns='predicted_crosscheck', aggfunc='sum', fill_value=0)
sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=1, linecolor='gray', ax=ax5, cbar_kws={'label': 'Customers'})
ax5.set_title(f'Customers Affected Heatmap\nTop 10 {location_col} vs Top 5 predicted_crosscheck',
              fontsize=12, fontweight='bold')
ax5.set_xlabel('predicted_crosscheck', fontsize=10, fontweight='bold')
ax5.set_ylabel(location_col, fontsize=10, fontweight='bold')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax5.yaxis.get_majorticklabels(), fontsize=9)

# 6. Distribution Stats
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
stats_text = f"""
SUMMARY STATISTICS (As Entity)

Total Records: {len(df):,}

Top 5 predicted_crosscheck Categories:
"""
for idx, (cat, count) in enumerate(predicted_crosscheck_counts.head(5).items(), 1):
    stats_text += f"{idx}. {cat}: {count:,}\n"

stats_text += f"\nTotal Customers Affected: {df['CI'].sum():,.0f}"
stats_text += f"\nAverage per Record: {df['CI'].mean():.1f}"
stats_text += f"\nMedian per Record: {df['CI'].median():.1f}"

ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('predicted_crosscheck_analysis_entity.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# FIGURE 2: predicted_crosscheck ANALYSIS (EXPANDED/MULTI-LABEL)
# ============================================================
fig2 = plt.figure(figsize=(20, 12))

expanded_counts = df_expanded['predicted_crosscheck_expanded'].value_counts()

# 1. Expanded predicted_crosscheck Distribution
ax1 = plt.subplot(2, 3, 1)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
wedges, texts, autotexts = ax1.pie(expanded_counts, labels=expanded_counts.index,
                                     autopct='%1.1f%%', colors=colors, startangle=90,
                                     pctdistance=0.85)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_weight('bold')
ax1.set_title('predicted_crosscheck Distribution\n(Expanded - Each Label Counted)', fontsize=14, fontweight='bold', pad=20)

# 2. Expanded vs Entity Comparison
ax2 = plt.subplot(2, 3, 2)
entity_main = df['predicted_crosscheck'].str.split(',').str[0].str.strip().value_counts().head(6)
expanded_main = df_expanded['predicted_crosscheck_expanded'].value_counts().head(6)
x = np.arange(len(entity_main))
width = 0.35
bars1 = ax2.bar(x - width/2, entity_main.values, width, label='As Entity (Primary)', color='#FF6B6B', alpha=0.8)
bars2 = ax2.bar(x + width/2, expanded_main.values, width, label='Expanded (All)', color='#4ECDC4', alpha=0.8)
ax2.set_xlabel('Category', fontsize=11, fontweight='bold')
ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
ax2.set_title('Entity vs Expanded Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(entity_main.index, rotation=45, ha='right', fontsize=10)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 3. Customers Affected (Expanded)
ax3 = plt.subplot(2, 3, 3)
cust_expanded = df_expanded.groupby('predicted_crosscheck_expanded')['CI'].sum().sort_values(ascending=False)
bars = ax3.barh(range(len(cust_expanded)), cust_expanded.values,
                color=plt.cm.coolwarm(np.linspace(0, 1, len(cust_expanded))))
ax3.set_yticks(range(len(cust_expanded)))
ax3.set_yticklabels(cust_expanded.index, fontsize=11)
ax3.set_xlabel('Customers Affected', fontsize=11, fontweight='bold')
ax3.set_title('Customers Affected\n(Expanded Labels)', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, cust_expanded.values)):
    ax3.text(val, i, f' {val:,.0f}', va='center', fontweight='bold', fontsize=10)

# 4. Stacked Area Chart for Top Locations
ax4 = plt.subplot(2, 3, 4)
top10_locs = df[location_col].value_counts().head(10).index
location_predicted_crosscheck = df[df[location_col].isin(top10_locs)].groupby([location_col, 'predicted_crosscheck']).size().unstack(fill_value=0)
top_predicted_crosscheck_cols = df['predicted_crosscheck'].value_counts().head(5).index
location_predicted_crosscheck_top = location_predicted_crosscheck[top_predicted_crosscheck_cols] if all(col in location_predicted_crosscheck.columns for col in top_predicted_crosscheck_cols) else location_predicted_crosscheck
location_predicted_crosscheck_top.plot(kind='bar', stacked=True, ax=ax4,
                              colormap='tab10', edgecolor='black', linewidth=0.5)
ax4.set_title(f'Top 10 {location_col} by predicted_crosscheck\n(Stacked)', fontsize=14, fontweight='bold')
ax4.set_xlabel(location_col, fontsize=11, fontweight='bold')
ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
ax4.legend(title='predicted_crosscheck', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax4.tick_params(axis='x', rotation=45, labelsize=9)
ax4.grid(axis='y', alpha=0.3)

# 5. Multi-label Analysis
ax5 = plt.subplot(2, 3, 5)
multi_label_count = df['predicted_crosscheck'].str.contains(',').sum()
single_label_count = len(df) - multi_label_count
labels = ['Single Label', 'Multi-Label']
sizes = [single_label_count, multi_label_count]
colors = ['#95E1D3', '#F38181']
explode = (0.05, 0.1)
wedges, texts, autotexts = ax5.pie(sizes, labels=labels, autopct='%1.1f%%',
                                     colors=colors, explode=explode, shadow=True, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')
ax5.set_title('Single vs Multi-Label Tickets', fontsize=14, fontweight='bold', pad=20)

# 6. Expanded Statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
stats_text = f"""
EXPANDED ANALYSIS

Total Records: {len(df):,}
Total Expanded Entries: {len(df_expanded):,}
Multi-Label Tickets: {multi_label_count:,} ({multi_label_count/len(df)*100:.1f}%)

Expanded Category Counts:
"""
for idx, (cat, count) in enumerate(expanded_counts.items(), 1):
    stats_text += f"{idx}. {cat}: {count:,}\n"

stats_text += f"\nTotal Customers (Expanded): {df_expanded['CI'].sum():,.0f}"

ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('predicted_crosscheck_analysis_expanded.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Customers Affected vs Cause (Entity)
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111)
cust_by_cause = df.groupby('predicted_crosscheck')['CI'].sum().sort_values(ascending=False).head(10)
bars = ax2.barh(range(len(cust_by_cause)), cust_by_cause.values,
                color=plt.cm.viridis(np.linspace(0, 1, 10)))
ax2.set_yticks(range(len(cust_by_cause)))
ax2.set_yticklabels(cust_by_cause.index, fontsize=10)
ax2.set_xlabel('Customers Affected', fontsize=11, fontweight='bold')
ax2.set_title('Customers Affected by Cause (Entity)', fontsize=14, fontweight='bold', pad=20)
ax2.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, cust_by_cause.values)):
    ax2.text(val, i, f' {val:,.0f}', va='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('customers_affected_entity.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig2)

# 3. Location vs Cause (Entity)
fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111)
pivot_data = df.groupby([location_col, 'predicted_crosscheck']).size().unstack(fill_value=0)
top_locs = df[location_col].value_counts().head(10).index
top_causes = df['predicted_crosscheck'].value_counts().head(5).index
plot_data = pivot_data.loc[top_locs, top_causes]
plot_data.plot(kind='barh', stacked=True, ax=ax3, colormap='tab10')
ax3.set_title(f'{location_col} by Cause (Entity)', fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('Number of Incidents', fontsize=11, fontweight='bold')
ax3.set_ylabel(location_col, fontsize=11, fontweight='bold')
ax3.legend(title='Cause', bbox_to_anchor=(1.05, 1), fontsize=8)
plt.tight_layout()
plt.savefig('location_vs_cause_entity_stacked.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig3)

# 5. Customers Affected vs Cause (Expanded)
fig5 = plt.figure(figsize=(10, 8))
ax5 = fig5.add_subplot(111)
cust_expanded = df_expanded.groupby('predicted_crosscheck_expanded')['CI'].sum().sort_values(ascending=False)
bars = ax5.barh(range(len(cust_expanded)), cust_expanded.values,
                color=plt.cm.coolwarm(np.linspace(0, 1, len(cust_expanded))))
ax5.set_yticks(range(len(cust_expanded)))
ax5.set_yticklabels(cust_expanded.index, fontsize=10)
ax5.set_xlabel('Customers Affected', fontsize=11, fontweight='bold')
ax5.set_title('Customers Affected by Cause (Expanded)', fontsize=14, fontweight='bold', pad=20)
ax5.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, cust_expanded.values)):
    ax5.text(val, i, f' {val:,.0f}', va='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('customers_affected_expanded.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig5)

# 6. Location vs Cause (Expanded)
fig6 = plt.figure(figsize=(10, 8))
ax6 = fig6.add_subplot(111)
pivot_data_exp = df_expanded.groupby([location_col, 'predicted_crosscheck_expanded']).size().unstack(fill_value=0)
top_locs = df[location_col].value_counts().head(10).index
top_causes_exp = df_expanded['predicted_crosscheck_expanded'].value_counts().head(5).index
plot_data_exp = pivot_data_exp.loc[top_locs, top_causes_exp]
plot_data_exp.plot(kind='barh', stacked=True, ax=ax6, colormap='tab10')
ax6.set_title(f'{location_col} by Cause (Expanded)', fontsize=14, fontweight='bold', pad=20)
ax6.set_xlabel('Number of Incidents', fontsize=11, fontweight='bold')
ax6.set_ylabel(location_col, fontsize=11, fontweight='bold')
ax6.legend(title='Cause', bbox_to_anchor=(1.05, 1), fontsize=8)
plt.tight_layout()
plt.savefig('location_vs_cause_expanded_stacked.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig6)

# 3. Location vs Cause (Entity) - Heatmap version
fig3_heatmap = plt.figure(figsize=(10, 8))
ax3_heatmap = fig3_heatmap.add_subplot(111)
pivot_data = df.groupby([location_col, 'predicted_crosscheck']).size().unstack(fill_value=0)
top_locs = df[location_col].value_counts().head(10).index
top_causes = df['predicted_crosscheck'].value_counts().head(5).index
heatmap_data = pivot_data.loc[top_locs, top_causes]
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax3_heatmap,
            annot_kws={'size': 9}, cbar_kws={'label': 'Number of Incidents'})
ax3_heatmap.set_title(f'{location_col} by Cause (Entity)', fontsize=14, fontweight='bold', pad=20)
ax3_heatmap.set_xlabel('Cause', fontsize=11, fontweight='bold')
ax3_heatmap.set_ylabel(location_col, fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('location_vs_cause_entity_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig3_heatmap)

# ============================================================
# EXPANDED GRAPHS
# ============================================================

# 4. Cause Distribution Pie (Expanded)
fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(111)
expanded_counts = df_expanded['predicted_crosscheck_expanded'].value_counts()
colors = plt.cm.Set2(np.linspace(0, 1, len(expanded_counts)))
wedges, texts, autotexts = ax4.pie(expanded_counts, autopct='%1.1f%%', colors=colors, startangle=90,
                                   pctdistance=0.85, textprops={'fontsize': 10, 'weight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white' if float(autotext.get_text().strip('%')) > 5 else 'black')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')

ax4.legend(expanded_counts.index, title='Causes', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
ax4.set_title('Cause Distribution (Expanded)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('cause_distribution_expanded_pie.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig4)


# 6. Location vs Cause (Expanded) - Heatmap version
fig6_heatmap = plt.figure(figsize=(10, 8))
ax6_heatmap = fig6_heatmap.add_subplot(111)
pivot_data_exp = df_expanded.groupby([location_col, 'predicted_crosscheck_expanded']).size().unstack(fill_value=0)
top_locs = df[location_col].value_counts().head(10).index
top_causes_exp = df_expanded['predicted_crosscheck_expanded'].value_counts().head(5).index
heatmap_data_exp = pivot_data_exp.loc[top_locs, top_causes_exp]
sns.heatmap(heatmap_data_exp, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax6_heatmap,
            annot_kws={'size': 9}, cbar_kws={'label': 'Number of Incidents'})
ax6_heatmap.set_title(f'{location_col} by Cause (Expanded)', fontsize=14, fontweight='bold', pad=20)
ax6_heatmap.set_xlabel('Cause', fontsize=11, fontweight='bold')
ax6_heatmap.set_ylabel(location_col, fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('location_vs_cause_expanded_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig6_heatmap)

# 1. Cause Distribution (Entity) - Bar version
fig1_bar = plt.figure(figsize=(10, 8))
ax1_bar = fig1_bar.add_subplot(111)
cause_counts = df['predicted_crosscheck'].value_counts()
top_n = 7
if len(cause_counts) > top_n:
    top_categories = cause_counts.head(top_n)
    other_count = cause_counts.iloc[top_n:].sum()
    plot_data = pd.concat([top_categories, pd.Series({'Others': other_count})])
else:
    plot_data = cause_counts

ax1_bar.bar(plot_data.index, plot_data.values,
        color=sns.color_palette("pastel", len(plot_data)), alpha=0.8)
ax1_bar.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=10, color='black')
ax1_bar.set_ylabel('Number of Incidents', fontsize=11, fontweight='bold', color='black')
ax1_bar.set_xlabel('Cause', fontsize=11, fontweight='bold', color='black')
ax1_bar.set_title('Cause Distribution (Entity)', fontsize=14, fontweight='bold', pad=20, color='black')
ax1_bar.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('cause_distribution_entity_bar.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig1_bar)

# 2. Customers Affected and Incidents by Cause (Entity)
fig2_bar_line = plt.figure(figsize=(12, 8))
ax2_bl = fig2_bar_line.add_subplot(111)
cust_by_cause = df.groupby('predicted_crosscheck').agg({'CI': 'sum', 'predicted_crosscheck': 'count'}).rename(columns={'predicted_crosscheck': 'Count'}).sort_values('CI', ascending=False).head(10)
ax2_bl.bar(cust_by_cause.index, cust_by_cause['Count'],
        color=sns.color_palette("pastel", len(cust_by_cause)), alpha=0.6, label='Incident Count')
ax2_bl.set_ylabel('Number of Incidents', fontsize=11, fontweight='bold', color='black')
ax2_bl.tick_params(axis='y', labelcolor='black')

# Overlay line graph for customers affected
ax2_bl_twin = ax2_bl.twinx()
ax2_bl_twin.plot(cust_by_cause.index, cust_by_cause['CI'], color='teal', marker='o', linewidth=2, label='Customers Affected')
ax2_bl_twin.set_ylabel('Customers Affected', fontsize=11, fontweight='bold', color='teal')
ax2_bl_twin.tick_params(axis='y', labelcolor='teal')

ax2_bl.set_xticklabels(cust_by_cause.index, rotation=45, ha='right', fontsize=10, color='black')
ax2_bl.set_xlabel('Cause', fontsize=11, fontweight='bold', color='black')
ax2_bl.set_title('Customers Affected and Incidents by Cause (Entity)', fontsize=14, fontweight='bold', pad=20, color='black')
ax2_bl.grid(True, axis='y', alpha=0.3)
ax2_bl.legend(loc='upper left', fontsize=8, labelcolor='black', facecolor='white')
ax2_bl_twin.legend(loc='upper right', fontsize=8, labelcolor='teal', facecolor='white')
plt.tight_layout()
plt.savefig('ci_and_incidents_entity.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig2_bar_line)

# 4. Cause Distribution (Expanded) - Bar version
fig4_bar = plt.figure(figsize=(10, 8))
ax4_bar = fig4_bar.add_subplot(111)
expanded_counts = df_expanded['predicted_crosscheck_expanded'].value_counts()
ax4_bar.bar(expanded_counts.index, expanded_counts.values,
        color=sns.color_palette("pastel", len(expanded_counts)), alpha=0.8)
ax4_bar.set_xticklabels(expanded_counts.index, rotation=45, ha='right', fontsize=10, color='black')
ax4_bar.set_ylabel('Number of Incidents', fontsize=11, fontweight='bold', color='black')
ax4_bar.set_xlabel('Cause', fontsize=11, fontweight='bold', color='black')
ax4_bar.set_title('Cause Distribution (Expanded)', fontsize=14, fontweight='bold', pad=20, color='black')
ax4_bar.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('cause_distribution_expanded_bar.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig4_bar)

# 5. Customers Affected and Incidents by Cause (Expanded)
fig5_bar_line = plt.figure(figsize=(12, 8))
ax5_bl = fig5_bar_line.add_subplot(111)
cust_expanded = df_expanded.groupby('predicted_crosscheck_expanded').agg({'CI': 'sum', 'predicted_crosscheck_expanded': 'count'}).rename(columns={'predicted_crosscheck_expanded': 'Count'}).sort_values('CI', ascending=False)
ax5_bl.bar(cust_expanded.index, cust_expanded['Count'],
        color=sns.color_palette("pastel", len(cust_expanded)), alpha=0.6, label='Incident Count')
ax5_bl.set_ylabel('Number of Incidents', fontsize=11, fontweight='bold', color='black')
ax5_bl.tick_params(axis='y', labelcolor='black')

# Overlay line graph for customers affected
ax5_bl_twin = ax5_bl.twinx()
ax5_bl_twin.plot(cust_expanded.index, cust_expanded['CI'], color='teal', marker='o', linewidth=2, label='Customers Affected')
ax5_bl_twin.set_ylabel('Customers Affected', fontsize=11, fontweight='bold', color='teal')
ax5_bl_twin.tick_params(axis='y', labelcolor='teal')

ax5_bl.set_xticklabels(cust_expanded.index, rotation=45, ha='right', fontsize=10, color='black')
ax5_bl.set_xlabel('Cause', fontsize=11, fontweight='bold', color='black')
ax5_bl.set_title('Customers Affected and Incidents by Cause (Expanded)', fontsize=14, fontweight='bold', pad=20, color='black')
ax5_bl.grid(True, axis='y', alpha=0.3)
ax5_bl.legend(loc='upper left', fontsize=8, labelcolor='black', facecolor='white')
ax5_bl_twin.legend(loc='upper right', fontsize=8, labelcolor='teal', facecolor='white')
plt.tight_layout()
plt.savefig('ci_and_incidents_expanded.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig5_bar_line)

# Set style for last figure
sns.set_palette("pastel")

# Define pastel color mapping for causes (explicit pastel shades)
cause_colors = {
    'Flooding': '#8FBDE8',      # pastel blue
    'Vegetation': '#A7DCA5',    # pastel green
    'Damaged Pole': '#F7A5A5',  # pastel coral/red
    'Other': '#D3D3D3'          # light gray
}

# 6. Location vs Cause (Expanded) - Grouped bar + line version
fig6_grouped = plt.figure(figsize=(14, 8))
ax6_grouped = fig6_grouped.add_subplot(111)

# Prepare data
pivot_data_exp = df_expanded.groupby([location_col, 'predicted_crosscheck_expanded']).agg({
    'CI': 'sum',
    'predicted_crosscheck_expanded': 'size'
}).rename(columns={'predicted_crosscheck_expanded': 'Count'}).reset_index()

top_locs = df[location_col].value_counts().head(10).index
bar_data = pivot_data_exp[pivot_data_exp[location_col].isin(top_locs)]

# Get unique causes from the predicted_crosscheck_expanded column
unique_causes = sorted(bar_data['predicted_crosscheck_expanded'].unique())  # Sort for consistent order

# Build a color map: prefer user-specified pastel cause_colors, else use seaborn pastel palette
default_colors = sns.color_palette('pastel', n_colors=max(3, len(unique_causes)))
color_map = {}
for i, cause in enumerate(unique_causes):
    key = cause.strip()
    # seaborn returns RGB tuples accepted by matplotlib; fallback to pastel palette
    color_map[cause] = cause_colors.get(key, default_colors[i % len(default_colors)])

# Set up bar positions
bar_width = 0.2
bar_positions = np.arange(len(top_locs))

# Plot bars for each cause and ensure we don't create duplicate legend labels
plotted_causes = set()
for idx, cause in enumerate(unique_causes):
    cause_data = bar_data[bar_data['predicted_crosscheck_expanded'] == cause]
    county_counts = {row[location_col]: row['Count'] for _, row in cause_data.iterrows()}
    counts = [county_counts.get(loc, 0) for loc in top_locs]
    color = color_map[cause]
    label = cause if cause not in plotted_causes else None
    ax6_grouped.bar(bar_positions + idx * bar_width, counts, bar_width,
            color=color, alpha=0.9, label=label, edgecolor='black', linewidth=0.3)
    plotted_causes.add(cause)

# Overlay line graph for customers affected and capture its handle
cust_affected = bar_data.groupby(location_col)['CI'].sum().reindex(top_locs).fillna(0)
ax6_grouped_twin = ax6_grouped.twinx()
line_handle, = ax6_grouped_twin.plot(bar_positions + bar_width * (len(unique_causes) - 1) / 2, cust_affected,
              color='#2A9D8F', marker='o', linewidth=2, label='Customers Affected')  # teal-ish pastel line
ax6_grouped_twin.set_ylabel('Customers Affected', fontsize=11, fontweight='bold', color='#2A9D8F')
ax6_grouped_twin.tick_params(axis='y', labelcolor='#2A9D8F')

# Customize axes and title (clearer title)
ax6_grouped.set_ylabel('Number of Incidents', fontsize=11, fontweight='bold', color='black')
ax6_grouped.tick_params(axis='y', labelcolor='black')
ax6_grouped.set_xticks(bar_positions + bar_width * (len(unique_causes) - 1) / 2)
ax6_grouped.set_xticklabels(top_locs, rotation=45, ha='right', fontsize=9, color='black')
ax6_grouped.set_xlabel(location_col, fontsize=11, fontweight='bold', color='black')
ax6_grouped.set_title(f'Top Locations by Cause (Expanded) — Customers Affected', fontsize=14, fontweight='bold', pad=24, color='black')
ax6_grouped.grid(True, axis='y', alpha=0.3)

# Build legend: colored patches for causes + the customers-affected line
cause_patches = [mpatches.Patch(facecolor=color_map[c], edgecolor='black', label=c) for c in unique_causes]
legend_handles = cause_patches + [line_handle]
ax6_grouped.legend(handles=legend_handles,
           title='Cause (bars) / Customers Affected (line)',
           loc='center left',
           bbox_to_anchor=(1.32, 0.5),  # move legend further right for padding
           fontsize=9,
           title_fontsize=10,
           frameon=True,
           borderpad=1.2,
           labelspacing=0.6,
           handletextpad=0.6)

plt.tight_layout()
plt.savefig('location_vs_cause_expanded_grouped.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig6_grouped)

# Print detailed summary
print("\n" + "="*80)
print("DETAILED predicted_crosscheck ANALYSIS")
print("="*80)
print(f"\nTotal Records: {len(df):,}")
print(f"Total Expanded Entries: {len(df_expanded):,}")
print(f"\n--- AS SINGLE ENTITY ---")
print("\nTop 10 predicted_crosscheck Categories:")
print(predicted_crosscheck_counts.head(10))
print(f"\n--- EXPANDED (MULTI-LABEL) ---")
print("\nExpanded predicted_crosscheck Counts:")
print(expanded_counts)
print(f"\n--- MULTI-LABEL ANALYSIS ---")
multi_label_count = df['predicted_crosscheck'].str.contains(',').sum()
single_label_count = len(df) - multi_label_count
print(f"Single-label tickets: {single_label_count:,} ({single_label_count/len(df)*100:.1f}%)")
print(f"Multi-label tickets: {multi_label_count:,} ({multi_label_count/len(df)*100:.1f}%)")
print("="*80)

