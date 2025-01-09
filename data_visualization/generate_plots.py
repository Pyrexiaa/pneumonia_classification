import matplotlib.pyplot as plt

def plot_pie_chart(labels, sizes, title, pctdistance=0.5, labeldistance=1.4):
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f', '#76d7ea', '#f4d03f', '#58d68d']
    
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.1f%%', startangle=140,
        pctdistance=pctdistance, labeldistance=labeldistance, colors=colors, shadow=True
    )

    # Add custom labels with names outside the pie chart
    for i, text in enumerate(texts):
        text.set_text(labels[i])
        text.set_horizontalalignment('center')
        text.set_verticalalignment('center')

    # Add custom percentage and value labels in the center of each slice
    for i, a in enumerate(autotexts):
        a.set_text(f'{sizes[i]}\n({a.get_text()})')

    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title(title, pad=40)
    plt.show()

def create_distribution_charts(data_dict, explode_type):
    # Data preparation
    labels = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
    # Create color gradients for each category
    colors = {
        'train': ['#1a75ff', '#66b3ff', '#99ccff'],  # different blues
        'test': ['#33cc33', '#70db70', '#99e699'],    # different greens
        'val': ['#ff66b3', '#ff99cc', '#ffcce6']      # different pinks
    }
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    sizes = []
    colors_list = []
    explode = []
    combined_labels = []
    
    for category, category_colors in colors.items():
        for i, label in enumerate(labels):
            key = f'{category} {label.lower()}'
            sizes.append(data_dict[key])
            colors_list.append(category_colors[i])
            explode.append(0.05 if category == explode_type else 0)
    
    for category in ['Train', 'Test', 'Val']:
        for label in labels:
            combined_labels.append(f'{category} {label}')
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=combined_labels,
        colors=colors_list,
        autopct='%1.2f%%',
        shadow=True,
        startangle=90,
        pctdistance=0.85
    )
    
    # Set title based on which sections are exploded
    plt.title(f'Restructed Data Distribution')
    
    plt.show()

# Dataset values
ori_dataset_train = 5216
ori_dataset_test = 624
ori_dataset_val = 16

ori_dataset_normal = 1583
ori_dataset_bacterial_pneumonia = 2780
ori_dataset_viral_pneumonia = 1493

re_categorised_dataset_train = 4632
re_categorised_dataset_test = 624
re_categorised_dataset_val = 600

re_categorised_train_images_normal = 1149
re_categorised_train_images_bacterial_pneumonia = 2338
re_categorised_train_images_viral_pneumonia = 1145

re_categorised_test_images_normal = 234
re_categorised_test_images_bacterial_pneumonia = 242
re_categorised_test_images_viral_pneumonia = 148

re_categorised_validation_images_normal = 200
re_categorised_validation_images_bacterial_pneumonia = 200
re_categorised_validation_images_viral_pneumonia = 200

# Dataset titles and labels
datasets = [
    ("Data Distribution of Train, Test and Validation Set", ["Train", "Test", "Validation"], [ori_dataset_train, ori_dataset_test, ori_dataset_val]),
    ("Number of images for Normal, Bacterial Pneumonia and Viral Pneumonia", ["Normal", "Bacterial\nPneumonia", "Viral\nPneumonia"], [ori_dataset_normal, ori_dataset_bacterial_pneumonia, ori_dataset_viral_pneumonia]),
    ("Data Distribution of Recategorised Train, Test and Validation Set", ["Train", "Test", "Validation"], [re_categorised_dataset_train, re_categorised_dataset_test, re_categorised_dataset_val]),
    ("Data Distribution of Multiclass Train Set", ["Normal", "Bacterial\nPneumonia", "Viral\nPneumonia"], [re_categorised_train_images_normal, re_categorised_train_images_bacterial_pneumonia, re_categorised_train_images_viral_pneumonia]),
    ("Data Distribution of Multiclass Test Set", ["Normal", "Bacterial\nPneumonia", "Viral\nPneumonia"], [re_categorised_test_images_normal, re_categorised_test_images_bacterial_pneumonia, re_categorised_test_images_viral_pneumonia]),
    ("Data Distribution of Multiclass Validation Set", ["Normal", "Bacterial\nPneumonia", "Viral\nPneumonia"], [re_categorised_validation_images_normal, re_categorised_validation_images_bacterial_pneumonia, re_categorised_validation_images_viral_pneumonia]),
]

data = {
    'train normal': re_categorised_train_images_normal,
    'train bacterial pneumonia': re_categorised_train_images_bacterial_pneumonia,
    'train viral pneumonia': re_categorised_train_images_viral_pneumonia,
    'test normal': re_categorised_test_images_normal,
    'test bacterial pneumonia': re_categorised_test_images_bacterial_pneumonia,
    'test viral pneumonia': re_categorised_test_images_viral_pneumonia,
    'val normal': re_categorised_validation_images_normal,
    'val bacterial pneumonia': re_categorised_validation_images_bacterial_pneumonia,
    'val viral pneumonia': re_categorised_validation_images_viral_pneumonia
}

# Plot each dataset
first = True
for title, labels, sizes in datasets:
    if first:
        plot_pie_chart(labels, sizes, title, 0.9)
        first = False
    else:
        plot_pie_chart(labels, sizes, title,0.6)

for section in ['train', 'test', 'val']:
    create_distribution_charts(data, section)
