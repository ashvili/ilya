import os
import argparse

import pandas as pd
import torch

from ml.model import NeuralNetwork, train, predict
from ml.utils import measure_elapsed_time
from ml.dataset import get_datasets


@measure_elapsed_time
def main(datasets_filepath,
         epochs=10,
         batch_size=64,
         lr=1e-3,
         nproc=1,
         log_interval=50,
         output_folder='./temp/'):
    print(f'{nproc=}')

    train_dataset, test_dataset = get_datasets(datasets_filepath)
    model = NeuralNetwork(7).to('cpu')

    train(
        model=model,
        train_dataset=train_dataset,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_kwargs={
            'lr': lr,
        },
        nproc=nproc,
        log_interval=log_interval,
        output_folder=output_folder,
    )
    print('train finished')

    x_min, x_max = train_dataset.borders['X_x']['MIN'], train_dataset.borders['X_x']['MAX']
    y_min, y_max = train_dataset.borders['X_y']['MIN'], train_dataset.borders['X_y']['MAX']
    z_min, z_max = train_dataset.borders['X_z']['MIN'], train_dataset.borders['X_z']['MAX']

    block_size = 5

    # normalize coords
    _tx = torch.arange(x_min, x_max, step=block_size)
    _ty = torch.arange(y_min, y_max, step=block_size)
    _tz = torch.arange(z_min, z_max, step=block_size)

    n_tx = torch.div(torch.sub(_tx, min(_tx)), (max(_tx) - min(_tx)))
    n_ty = torch.div(torch.sub(_ty, min(_ty)), (max(_ty) - min(_ty)))
    n_tz = torch.div(torch.sub(_tz, min(_tz)), (max(_tz) - min(_tz)))

    # n_tx = n_tx[:100]
    # n_ty = n_ty[:100]
    # n_tz = n_tz[:100]

    print('Cartesian product started')
    predict_data = torch.cartesian_prod(n_tx, n_ty, n_tz)
    print('Cartesian product finished')

    # predict
    print('Prediction started')
    result = predict(model, predict_data, nproc)
    print('Prediction finished')

    # denormalize
    result[:, 0] = torch.add(torch.mul(result[:, 0], (x_max - x_min)), x_min)
    result[:, 1] = torch.add(torch.mul(result[:, 1], (y_max - y_min)), y_min)
    result[:, 2] = torch.add(torch.mul(result[:, 2], (z_max - z_min)), z_min)

    df = pd.DataFrame(result.numpy())

    df.iloc[:, 3] = df.iloc[:, 3].astype(int).map(test_dataset.contents)

    extra_columns = [f'prob_{i}' for i in range(len(df.columns[4:]))]

    df.columns = ['x', 'y', 'z', 'content_id', *extra_columns]
    df['_x'] = block_size
    df['_y'] = block_size
    df['_z'] = block_size

    print('Saving...')
    contents = [d for _, d in df.groupby(['content_id'])]
    ores_count = len(contents)
    for idx, predicted in enumerate(contents):
        c_id = predicted['content_id'].iloc[0]
        fpath = f'{output_folder}predicted_{c_id}.csv'

        predicted.to_csv(fpath, index=False)
        print(f'[ {idx + 1} / {ores_count} ] Done, saved to {os.path.abspath(fpath)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train network and predict blocks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'blocks_file_path',
        metavar='blocks-file-path',
        type=str,
        help='path to file with blocks',
    )
    parser.add_argument(
        '--nproc',
        default=os.cpu_count(),
        type=int,
        help='CPU count'
    )
    parser.add_argument(
        '-e', '--epochs',
        default=100,
        type=int,
        help='Training epochs count'
    )
    parser.add_argument(
        '-bs', '--batch-size',
        default=63,
        type=int,
        help='Batch size epochs count'
    )
    parser.add_argument(
        '-lr', '--learning-rate',
        default=1e-3,
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '-o', '--output-folder',
        dest='output_folder',
        default='./temp/',
        type=str,
        help='path to save output files',
    )

    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    main(
        args.blocks_file_path,
        nproc=args.nproc,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        output_folder=args.output_folder,
    )
