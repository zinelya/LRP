import os

def add_overwrite_arg(parser, will_delete_dirs=False):
    if will_delete_dirs:
        _help = ('Force overwriting of the output files.\n'
                 'CAREFUL. The whole output directory will be deleted if it '
                 'exists.')
    else:
        _help = 'Force overwriting of the output files.'
    parser.add_argument(
        '-f', dest='overwrite', action='store_true', help=_help)

def add_verbose_arg(parser):
    parser.add_argument('-v', default="WARNING", const='INFO', nargs='?',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        dest='verbose',
                        help='Produces verbose output depending on '
                             'the provided level. \nDefault level is warning, '
                             'default when using -v is info.')
    
def assert_inputs_exist(parser, required, optional=None):
    """
    Assert that all inputs exist. If not, print parser's usage and exit.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    required: string or list of paths or list of lists of paths
        Required paths to be checked.
    optional: string or list of paths or list of lists of paths
        Optional paths to be checked.
    """

    def check(path):
        if not os.path.isfile(path):
            parser.error('Input file {} does not exist'.format(path))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        if isinstance(required_file, str):
            check(required_file)
        else:
            for file in required_file:
                check(file)
    for optional_file in optional or []:
        if optional_file is not None:
            if isinstance(optional_file, str):
                check(optional_file)
            else:
                for file in optional_file:
                    check(file)

def assert_outputs_exist(parser, args, required, optional=None,
                         check_dir_exists=True):
    """
    Assert that all outputs don't exist or that if they exist, -f was used.
    If not, print parser's usage and exit.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    args: argparse namespace
        Argument list.
    required: string or list of paths to files
        Required paths to be checked.
    optional: string or list of paths to files
        Optional paths to be checked.
    check_dir_exists: bool
        Test if output directory exists.
    """

    def check(path):
        if os.path.isfile(path) and not args.overwrite:
            parser.error('Output file {} exists. Use -f to force '
                         'overwriting'.format(path))

        if check_dir_exists:
            path_dir = os.path.dirname(path)
            if path_dir and not os.path.isdir(path_dir):
                parser.error('Directory {}/ \n for a given output file '
                             'does not exists.'.format(path_dir))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file:
            check(optional_file)