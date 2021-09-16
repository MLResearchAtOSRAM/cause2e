"""
This script replaces the head of html documentation files, in order to make github pages work.
"""

import subprocess
import glob
from pathlib import Path


def delete_directory(dirname, input_char='J\n'):
    """Deletes a directory and its content via windows command line.

    Args:
        dirname: A pathlib.Path indicating the directory.
        input_char: Optional; A string indicating the letter required for confirmation. Defaults to 'J'.
    """
    subprocess.run(f'rmdir /s {dirname}', shell=True, text=True, input=input_char)


def update_html_files(path):
    """Replaces heads of all html files in a given directory.

    Args:
        path: A pathlib.Path indicating the directory.
    """
    html_files = glob.glob(str(path / '*.html'))
    for file in html_files:
        _update_html_file(file)


def _update_html_file(filename):
    """Replaces the head of a given file.

    Args:
        filename: A string indicating the filename.
    """
    with open(filename) as f:
        text = f.read()
    new_html = '---\nlayout: default\n---' + _remove_head(text)
    with open(filename, 'w') as f:
        f.write(new_html)


def _remove_head(text):
    """Removes the head section of a string read from an html file.

    Args:
        text: A string (content of an html file).

    Returns:
        The same string but without the head section.
    """
    new_text = text.split('<head>')
    newest_text = new_text[1].split('</head>')
    return new_text[0] + newest_text[1]


sphinx = Path('.') / 'sphinx_config'
# delete previous documentation build
build = sphinx / '_build'
delete_directory(build)
# build documentation
make = sphinx / 'make.bat'
subprocess.run(f'{make} html')
# update htmls
html = sphinx / '_build' / 'html'
update_html_files(html)
# remove other build artifacts
doctrees = build / 'doctrees'
_static = html / '_static'
_sources = html / '_sources'
artifacts = [doctrees, _static, _sources]
for artifact in artifacts:
    delete_directory(artifact)
# copy folder to docs folder
docs = Path('.') / 'docs'
subprocess.run(f'copy {html} {docs}', shell=True)
