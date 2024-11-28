from utils import create_dataset
import os


def main():
    background_path = 'DATASET/Background'
    output_path = 'output/path/to/save/your/dataset'
    N = 100
    max_cells = 2000

    create_dataset(background_path, output_path, N, max_cells)


if __name__ == '__main__':
    main()
