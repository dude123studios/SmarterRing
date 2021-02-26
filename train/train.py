from train.prepare_imgs import pairs1, pairs2, labels
from models.face_recognition200_model import model
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('epochs',type=int,help='Number of Epochs to train on your images')
parser.add_argument('batch_size', type=int,help='Batch Size for Training',default=1)

args = parser.parse_args()
inputs = {'inputA': pairs1, 'inputB': pairs2}
model.fit(inputs, labels, epochs=args.epochs, batch_size=args.batch_size)