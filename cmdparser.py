import argparse

parser = argparse.ArgumentParser(description='Save path for the model')
parser.add_argument('--name', type=str, default='phase1_model', help='Name to use to save the model')
parser.add_argument('--hencoder', type=str, default=None, help="History encoder to use")
parser.add_argument('--loss', type=str, default=None, help="Loss to use")
parser.add_argument('--lr', type=float, default=1e-3, help="History encoder to use")
parser.add_argument('--epochs', type=float, default=50, help="Number of epochs")
parser.add_argument('--bs', type=int, default=32, help="Batch size")
parser.add_argument('--ego', action='store_true', default=False, help="Wether to encode the ego state")
parser.add_argument('--plot', action='store_true', default=False, help="Wether to save training plots")