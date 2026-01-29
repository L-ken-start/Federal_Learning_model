"""
Simplified Federated Learning Results Visualization
One file, no Chinese characters, works everywhere
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def show_simple_plot(result_dir='./federated_learning_results'):
    """
    Show simple training results plot
    Just run: python show_results.py
    """
    # 1. Auto-find result files
    result_path = Path(result_dir)
    json_files = list(result_path.glob('*.json'))

    if not json_files:
        print(f"❌ No result files found in {result_dir}")
        print("Please run federated learning training first")
        return

    # Use the latest result file
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    result_file = json_files[0]

    print(f"📊 Loading result file: {result_file}")

    # 2. Load data
    with open(result_file, 'r') as f:
        data = json.load(f)

    history = data.get('history', {})

    if 'test_accuracy' not in history or not history['test_accuracy']:
        print("❌ No test accuracy data in result file")
        return

    # 3. Extract data
    accuracies = history['test_accuracy']
    losses = history.get('test_loss', [0] * len(accuracies))
    rounds = list(range(len(accuracies)))

    # 4. Create plots (one figure, two subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 4.1 Accuracy plot
    ax1.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Federated Learning - Accuracy')
    ax1.grid(True, alpha=0.3)

    # Mark best accuracy
    best_idx = np.argmax(accuracies)
    best_acc = accuracies[best_idx]
    ax1.plot(best_idx, best_acc, 'r*', markersize=15, label=f'Best: {best_acc:.2f}%')
    ax1.legend()

    # 4.2 Loss plot
    ax2.plot(rounds, losses, 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Federated Learning - Loss')
    ax2.grid(True, alpha=0.3)

    # 5. Display training info
    config = data.get('config', {})
    print(f"\n📈 Training Information:")
    print(f"   Total Rounds: {len(rounds)}")
    print(f"   Initial Accuracy: {accuracies[0]:.2f}%")
    print(f"   Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"   Best Accuracy: {best_acc:.2f}%")
    print(f"   Improvement: {accuracies[-1] - accuracies[0]:.2f}%")

    if 'round_time' in history and history['round_time']:
        times = history['round_time']
        print(f"   Total Training Time: {sum(times):.1f} seconds")
        print(f"   Average per Round: {np.mean(times):.1f} seconds")

    print(f"\n⚙️  Training Configuration:")
    print(f"   Number of Clients: {config.get('num_clients', 'Unknown')}")
    print(f"   Aggregation Method: {config.get('aggregation_method', 'Unknown')}")
    print(f"   Model Type: {config.get('model_type', 'Unknown')}")

    # 6. Show and save the plot
    plt.tight_layout()

    # Auto-save the plot
    plot_file = result_path / 'simple_training_plot.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Plot saved: {plot_file}")

    # Try to show the plot, if fails just save it
    try:
        plt.show()
    except Exception as e:
        print(f"⚠️  Could not display plot: {e}")
        print(f"✅ But the plot was saved to: {plot_file}")

def show_text_only():
    """
    Text-only version, no plot, no matplotlib needed
    """
    result_dir = './federated_learning_results'
    result_path = Path(result_dir)
    json_files = list(result_path.glob('*.json'))

    if not json_files:
        print(f"❌ No result files found in {result_dir}")
        return

    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    result_file = json_files[0]

    print(f"📊 Loading: {result_file}")

    with open(result_file, 'r') as f:
        data = json.load(f)

    history = data.get('history', {})

    if 'test_accuracy' not in history:
        print("❌ No accuracy data")
        return

    accuracies = history['test_accuracy']

    print("\n" + "="*60)
    print("FEDERATED LEARNING RESULTS")
    print("="*60)

    print(f"\n📈 ACCURACY PROGRESS:")
    for i, acc in enumerate(accuracies):
        arrow = ""
        if i > 0:
            if acc > accuracies[i-1]:
                arrow = " ↗"
            elif acc < accuracies[i-1]:
                arrow = " ↘"
            else:
                arrow = " →"

        if i == 0:
            arrow = " (START)"
        elif i == len(accuracies) - 1:
            arrow = " (FINAL)"

        print(f"  Round {i:2d}: {acc:6.2f}%{arrow}")

    print(f"\n✅ SUMMARY:")
    print(f"  Total Rounds: {len(accuracies)}")
    print(f"  Start Accuracy: {accuracies[0]:.2f}%")
    print(f"  Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"  Best Accuracy: {max(accuracies):.2f}%")
    print(f"  Improvement: {accuracies[-1] - accuracies[0]:.2f}%")

    # Simple ASCII chart
    print(f"\n📊 ACCURACY CHART:")

    if len(accuracies) <= 1:
        print("  (Not enough data for chart)")
        return

    # Scale to 20 width
    max_acc = max(accuracies)
    min_acc = min(accuracies)

    # Header
    print("   Accuracy | " + "".join([str(i%10) for i in range(len(accuracies))]))
    print("   ---------+-" + "-" * len(accuracies))

    # Draw bars
    for level in range(10, 0, -1):
        threshold = min_acc + (max_acc - min_acc) * level / 10
        line = f"   {threshold:5.1f}%  | "

        for acc in accuracies:
            if acc >= threshold:
                line += "█"
            else:
                line += " "

        print(line)

    print("   ---------+-" + "-" * len(accuracies))
    print("            | Rounds")

    print("\n" + "="*60)

if __name__ == "__main__":
    print("=" * 50)
    print("FEDERATED LEARNING RESULTS VISUALIZATION")
    print("=" * 50)

    # Check if matplotlib is available
    try:
        import matplotlib
        print("✅ Matplotlib is available, showing plots...")
        show_simple_plot()
    except ImportError:
        print("⚠️  Matplotlib not found, showing text-only results...")
        show_text_only()

    print("\n✅ Done!")
    print("=" * 50)