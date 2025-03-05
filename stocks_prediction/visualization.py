import matplotlib.pyplot as plt


def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.savefig("output.png")
