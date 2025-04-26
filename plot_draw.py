import matplotlib.pyplot as plt
import seaborn as sns

# Global variables
TARGET = 'HighRisk'

def draw_highrisk_count_plot(df):
    """
    Create a count plot for HighRisk column.

    Parameters:
        df (pd.DataFrame): original DataFrame with HighRisk column.
    """

    # Set up count plot size.
    plt.figure(figsize=(10, 10))

    # Plot a count plot.
    ax = sns.countplot(x=df)

    # Label for x-axis and y-axis
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.ylabel("Count")

    # Calculate the number for each case
    for p in ax.patches:

        # Get the height of the column (AKA the number for each case)
        height = p.get_height()

        # Put it at the top of the bar
        ax.text(p.get_x() + p.get_width() / 2., height + 1, int(height),
                ha="center", fontsize=12)

    # Title of count plot
    plt.title("Heart Risk count (Yes = Heart Risk, No = No Heart Risk)",
              fontsize=20, fontweight='bold'
              )

    plt.show()


def plot_all_features_heatmap(features):
    """
    Create a heatmap between all feature and target.

    Parameters:
        features (pd.DataFrame): DataFrame contains all features.
    """

    # Set up heatmap size.
    plt.figure(figsize=(15, 15))

    # Calculate the correlation matrix.
    corr = features.corr()

    # Sort the correlations in descending order
    sorted_corr = corr[[TARGET]].sort_values(by=TARGET, ascending=False)

    # Plot a heatmap.
    sns.heatmap(sorted_corr, annot=True, cmap="coolwarm", fmt=".2f")

    # Title of heatmap
    plt.title(f"Heatmap comparison between all features and {TARGET}",
              fontsize=20, fontweight='bold'
              )

    plt.show()


def plot_selected_features_heatmap(features):
    """
    Create a heatmap between 10 selected feature and target.

    Parameters:
        features (pd.DataFrame): DataFrame contains 10 selected features.
    """

    # Set up heatmap size.
    plt.figure(figsize=(15, 15))

    # Calculate the correlation matrix.
    corr = features.corr()

    # Plot a heatmap.
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    # Title of heatmap
    plt.title(f"Heatmap comparison between 10 selected features and {TARGET}",
              fontsize=20, fontweight='bold'
              )

    plt.show()
