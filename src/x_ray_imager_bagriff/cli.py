# Copyright (c) 2026 Brady Griffith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""General CLI tools."""
import logging
import click


def log_level_options(log: logging.Logger):
    """Create click option dectorators to set log file to a desired level.

    These flags wil be reused on every command throughout this project.

    Args:
        log: The Logger whose level will be changed.
    """
    def set_log_level(ctx, param, value):
        """Update the log level according to a CLI value."""
        _ = ctx, param  # Not needed.
        if value is None:
            return
        log.setLevel(value)
        handler = logging.StreamHandler()
        log.addHandler(handler)

    def decorator(func):
        func = click.option('--verbose', '-v', flag_value=logging.INFO,
                            callback=set_log_level, expose_value=False,
                            help="Print extra information during run."
                            )(func)
        func = click.option('--debug', '-d', flag_value=logging.DEBUG,
                            callback=set_log_level, expose_value=False,
                            help="Print out all debug information during run."
                            )(func)
        return func

    return decorator
