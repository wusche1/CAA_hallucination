import pandas as pd
import matplotlib.pyplot as plt
import os

viridis = plt.cm.get_cmap(
    "viridis", 8
)  # Getting 8 distinct colors from the Viridis color map

# Define the color maps using Viridis colors
# Note: The Viridis color map ranges from dark (0) to light (7), so the indices are adjusted accordingly
truth_colors = {
    "(correct)": viridis(7),  # Light Yellow
    "(begun)": viridis(6),  # Yellow-Green
    "(false)": viridis(4),  # Light Blue
    "(refused)": viridis(3),  # Dark Blue
    "(ignorant)": viridis(5),  # Mid-Blue
    "(unrelated)": viridis(2),  # Blue-Green
    "(other)": viridis(1),  # Mid-Green
    "failed": viridis(0),  # Dark Blue-Black
}

fiction_colors = {
    "(deny)": viridis(7),  # Light Yellow
    "(clarify)": viridis(6),  # Yellow-Green
    "(accept)": viridis(4),  # Light Blue
    "(fiction)": viridis(3),  # Dark Blue
    "(ignore)": viridis(5),  # Mid-Blue
    "(other)": viridis(1),  # Mid-Green
    "failed": viridis(0),  # Dark Blue-Black
}


def plot_ratings_bar_charts_fiction(
    ax, path, coeff_list, colors, answer_type, steering_method="fiction"
):
    plt.tight_layout()

    # We need to keep track of the offset for the bars for each coeff
    offsets = range(len(coeff_list))[::-1]

    for offset, coeff in zip(offsets, coeff_list):
        datapoint_path = os.path.join(path, f"{steering_method}_steered_{coeff}.csv")
        datapoint_df = pd.read_csv(datapoint_path)

        # Normalize the ratings to get percentages
        fiction_ratings = datapoint_df[answer_type].value_counts(normalize=True)
        width = 1  # Set a smaller width to fit all bars
        bottom = 0  # Start the bar at 0

        # Plot each rating for the current coeff
        for rating, color in colors.items():
            height = fiction_ratings.get(rating, 0)
            ax.bar(offset, height, width, color=color, bottom=bottom)
            bottom += height

    # Set x-ticks to be the middle of the group of bars for each coeff
    ax.set_xticks(list(offsets)[::-1])
    ax.set_xticklabels(coeff_list)

    # Set y-axis limits
    ax.set_ylim(0, 1)


def plot_ratings(
    data_path,
    question_types,
    names,
    coeffs,
    plt_folder,
    plot_type,
    resizer=0.75,
    steering_method="fiction",
):
    if not os.path.exists(plt_folder):
        os.makedirs(plt_folder)
    fig, axs = plt.subplots(
        2, 2, figsize=(10 * resizer, 8 * resizer)
    )  # Adjust figsize as needed
    axs = axs.flatten()  # Flatten the axis array for easier access

    # Choose colors and rating type based on plot_type
    if plot_type == "fiction":
        colors = fiction_colors
        rating_type = "fiction_rating_0"
    elif plot_type == "truth":
        colors = truth_colors
        rating_type = "truth_rating_0"
    else:
        raise ValueError("plot_type must be either 'fiction' or 'truth'")

    # Create custom legend handles
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in colors]
    labels = list(colors.keys())

    for i, (question_type, name) in enumerate(zip(question_types, names)):
        print(question_type)
        plot_ratings_bar_charts_fiction(
            axs[i],
            f"{data_path}/{question_type}",
            coeffs,
            colors,
            rating_type,
            steering_method=steering_method,
        )
        axs[i].set_title(name, fontsize=16)

        # Set x-axis labels for bottom row and remove for top row
        if i >= 2:  # Only for the bottom row
            axs[i].set_xlabel("Multiplier", fontsize=14)
        else:
            axs[i].set_xticklabels([])

        # Set y-axis labels for left column
        if i % 2 == 0:  # Only for the left column
            axs[i].set_ylabel("Fraction of Ratings", fontsize=14)
        else:
            axs[i].set_yticklabels([])

    # Adjust layout and display the legend
    plt.tight_layout()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

    # Save and show the plot
    image_name = f"combined_rating_plot_{plot_type}"
    plt.savefig(
        f"{plt_folder}/{image_name}.svg", format="svg", bbox_inches="tight"
    )  # Saving as SVG
    plt.savefig(
        f"{plt_folder}/{image_name}.png", format="png", dpi=300, bbox_inches="tight"
    )  # Saving as PNG with higher dpi
    plt.show()
    return
