from alexnet import AlexNet
from utils import get_files_and_labels, get_n_classes
def main():
    lbl_file = "./data/labels.txt"
    files, labels = get_files_and_labels(lbl_file)

    n_classes = get_n_classes(labels)



    model = AlexNet(n_classes=n_classes)

if __name__ == "__main__":
    main()