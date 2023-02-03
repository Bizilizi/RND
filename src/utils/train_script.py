def overwrite_config_with_args(cli_args, config):
    """The method overwrites config fields with passed to cli arguments"""

    for k, v in vars(cli_args).items():
        if hasattr(config, k):
            setattr(config, k, type(getattr(config, k))(v))


def parse_arguments(parser):
    """
    This method constructs a new parser for cli command with new unregisters
    arguments with str type and runs `parse_args` on it.
    """

    parsed, unknown = parser.parse_known_args()

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split("=")[0], type=str)

    args = parser.parse_args()
    return args
