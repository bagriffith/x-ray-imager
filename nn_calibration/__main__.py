import argparse
import pathlib
import logging
import sys
import numpy as np
import nn_calibration

top_logger = logging.getLogger(__package__)
# top_logger.setLevel(logging.DEBUG)
# for child in ['pca', 'cluster', 'interpolate', 'lookup']:
#     child_logger = top_logger.getChild(child)
#     child_logger.setLevel(logging.DEBUG)
# file_handler = logging.FileHandler('nn_calibration-debug.log')
# top_logger.addHandler(file_handler)
# logging.getLogger('booms_packet').setLevel(logging.INFO)
# logging.getLogger('booms_packet.stream').setLevel(logging.INFO)
# logging.getLogger('booms_packet.stream').addHandler(file_handler)
logger = top_logger.getChild(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

pca_logger = top_logger.getChild('pca')
pca_logger.setLevel(logging.INFO)
pca_logger.addHandler(logging.StreamHandler(sys.stdout))

# logger.addHandler(logging.StreamHandler())
# logging.getLogger('nn_calibration.pca').setLevel(logging.INFO)
# logging.getLogger('nn_calibration.pca').addHandler(logging.StreamHandler())

def int_positive(int_str):
    """Converts a string of a positive integer to an int.
    """
    arg_value = int(int_str)
    if arg_value < 1:
        raise ValueError('Integer is not positive')
    return arg_value


def cluster(args):
    """Process arguments and run the source cluster identification.
    """
    files = nn_calibration.sidecars.sidecar_path(args.filenames, args.list)
    for sidecar in files:
        logger.info('Looking for clusters in %s', sidecar)
        original = nn_calibration.plot.OUTPUT_DIR
        nn_calibration.plot.OUTPUT_DIR = original / (sidecar.stem) / ''
        nn_calibration.plot.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        nn_calibration.cluster.identify_and_save(sidecar, plot=args.plot)
        nn_calibration.plot.OUTPUT_DIR = original


def basis(args):
    """Form a PCA basis out of the given sets of calibration runs
    """
    files = nn_calibration.sidecars.sidecar_path(args.filenames, args.list)
    logger.debug(files)

    clusters = nn_calibration.cluster.load_all_centers(files)
    pca_basis = nn_calibration.pca.form_basis(clusters, args.comps)

    if args.plot:
        nn_calibration.plot.basis(pca_basis, clusters.shape[:2], mode='wire')
        nn_calibration.plot.basis(pca_basis, clusters.shape[:2], mode='mesh')

    np.savetxt(args.output, pca_basis)


def calibrate(args):
    files = args.filenames
    # Form interpolation
    if args.list:
        tmp_list = []
        for f in args.filenames:
            tmp_list.extend(nn_calibration.cluster.grab_list(f))
        files = tmp_list
    centers = nn_calibration.cluster.load_all_centers(files, mask=not args.skip_mask)
    energies, positions = nn_calibration.sidecars.load_all_info(files, mask=not args.skip_mask)

    interp = nn_calibration.interpolate.methods[args.interp_method]

    pca_basis = None
    if nn_calibration.interpolate.uses_basis[args.interp_method]:
        if args.basis is None:
            raise ValueError('No basis provided')
        pca_basis = np.loadtxt(args.basis)

    # Make a lookup table
    calibration = interp(energies, positions, centers,
                         **({'basis': pca_basis} if pca_basis is not None else dict()))

    lookup_method = nn_calibration.lookup.methods[args.lookup_method]
    lookup = nn_calibration.lookup.from_interpolation(lookup_method, calibration, args.points)
    lookup.save_to(args.output)
    # c = nn_calibration.lookup.load(args.output)


def main():
    parser = argparse.ArgumentParser(prog='BoomsNNCal',
                                     description='Process calibration data and'
                                     ' apply to a data file.')
    subparsers = parser.add_subparsers()
    parser_cluster = subparsers.add_parser('cluster')
    parser_cluster.add_argument('filenames', type=pathlib.Path, nargs='+')
    parser_cluster.add_argument('--list', '-l', action='store_true')
    parser_cluster.add_argument('--plot', '-p', action='store_true')
    parser_cluster.set_defaults(func=cluster)

    parser_basis = subparsers.add_parser('basis')
    parser_basis.add_argument('filenames', type=pathlib.Path, nargs='+')
    parser_basis.add_argument('--list', '-l', action='store_true')
    parser_basis.add_argument('--plot', '-p', action='store_true')
    parser_basis.add_argument('--output', '-o', type=pathlib.Path,
                              default=pathlib.Path('./basis.txt'))
    parser_basis.add_argument('--comps', '-n', type=int_positive,
                              default=nn_calibration.pca.DEFAULT_COMPS)
    parser_basis.set_defaults(func=basis)

    parser_calibrate = subparsers.add_parser('calibrate')
    parser_calibrate.add_argument('filenames',  type=pathlib.Path, nargs='+')
    parser_calibrate.add_argument('--points', '-n', type=int, nargs=3, default=[128, 29, 29])
    parser_calibrate.add_argument('--list', '-l', action='store_true')
    parser_calibrate.add_argument('--skip_mask', action='store_true')
    parser_calibrate.add_argument('--basis', '-b', type=pathlib.Path)
    parser_calibrate.add_argument('--output', '-o', type=pathlib.Path,
                                  default=pathlib.Path('./calibration.p'))
    parser_calibrate.add_argument('--interp_method', '-i', choices=nn_calibration.interpolate.methods.keys(), default='cubic')
    parser_calibrate.add_argument('--lookup_method', '-m', choices=nn_calibration.lookup.methods.keys(), default='tree')
    parser_calibrate.set_defaults(func=calibrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
