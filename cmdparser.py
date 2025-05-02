import argparse

parser = argparse.ArgumentParser(description='Model training script')
parser.add_argument('--name', type=str, default='phase1_model', help='Name to use to save the model. Name is used to create a folder in the models directory')
parser.add_argument('--hencoder', type=str, default=None, help="History encoder to use. Default is no encoding")
parser.add_argument('--loss', type=str, default=None, help="Loss to use, default is MSE")
parser.add_argument('--lr', type=float, default=1e-3, help="History encoder to use, default is 1e-3")
parser.add_argument('--epochs', type=float, default=50, help="Number of epochs, default is 50")
parser.add_argument('--bs', type=int, default=32, help="Batch size, default is 32")
parser.add_argument('--ego', action='store_true', default=False, help="Wether to encode the ego state, default is False")
parser.add_argument('--plot', action='store_true', default=False, help="Wether to save training plots, default is False")