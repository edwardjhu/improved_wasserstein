from analyze import Line, ApproximateAccuracy, plot_empirical_accuracy_per_sigma_against_original_one_sample
import argparse

parser = argparse.ArgumentParser(description='Plot attack results.')
parser.add_argument('--result', required=True)
parser.add_argument('--title', default='Attack Result')
parser.add_argument('--save', default='attack_result')

args = parser.parse_args()

plot_empirical_accuracy_per_sigma_against_original_one_sample(
    args.save, args.title, 1.95, \
    [
        Line(ApproximateAccuracy(args.result), f"This Attack")
    ],
    [], radius_step=0.01, xlabel='1-Wasserstein Radius')
