def overwrite_config_with_args(cli_args, config):
    """The method overwrites config fields with passed to cli arguments"""

    for k, v in vars(cli_args).items():
        if hasattr(config, k):
            attr_type = type(getattr(config, k))
            if attr_type is bool:
                bool_v = str(v).lower() in ["1", "on", "t", "true", "y", "yes"]
                setattr(config, k, bool_v)
            else:
                setattr(config, k, attr_type(v))


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
