import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file (update filename if different)
with open('./results/bert-rakuten-final/classification_report_20260128_013209.json', 'r') as f:
    data = json.load(f)

# Extract per-class metrics
per_class = data['per_class_metrics']

# Filter out the aggregate metrics (accuracy, macro avg, weighted avg)
classes = []
f1_scores = []

for class_name, metrics in per_class.items():
    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
        classes.append(class_name)
        f1_scores.append(metrics['f1-score'])

# Sort by class name (numeric order)
sorted_indices = np.argsort([int(c) for c in classes])
classes = [classes[i] for i in sorted_indices]
f1_scores = [f1_scores[i] for i in sorted_indices]

# Create the bar chart
fig, ax = plt.subplots(figsize=(14, 6))

bars = ax.bar(range(len(classes)), f1_scores, color='#3b82f6', alpha=0.8, 
              edgecolor='black', linewidth=0.5)

# Customize the plot
ax.set_xlabel('Product Type Code (Class)', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('F1-Score by Product Type Code', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add a horizontal line for macro average F1
weighted_f1 = data['overall_metrics']['weighted_f1']
ax.axhline(y=weighted_f1, color='red', linestyle='--', linewidth=2, 
           label=f'Weighted F1: {weighted_f1:.3f}')
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('./streamlit/images/f1_scores_by_class.png', dpi=300, bbox_inches='tight')
plt.show()


