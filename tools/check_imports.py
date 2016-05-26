# -*- coding: utf-8 -*-

"""
This script is used to check imports of ipwxlearn package, so that the code of backend
should not mistakenly import another backend.
"""

import ast
import os
import re

BACKEND_NAMES = ['tensorflow', 'theano']


def relative_path(path):
    return os.path.relpath(path, os.path.abspath(os.path.join(os.path.split(__file__)[0], '../')))


def extract_backend(name):
    m = re.match('^(ipwxlearn\.glue\.|(\.\.)+)(?P<backend>(%s)).*' % '|'.join(BACKEND_NAMES), name)
    return m.group("backend") if m else None


def validate_name(path, name, restrict_backend):
    backend = extract_backend(name)
    if backend and backend not in restrict_backend:
        print('%s: import %s' % (relative_path(path), name))


def check_file(path, restrict_backend):
    def chkbody(body):
        if hasattr(body, 'body'):
            for b in body.body:
                chkbody(b)
        if isinstance(body, ast.Import):
            for name in body.names:
                validate_name(path, name.name, restrict_backend)
        elif isinstance(body, ast.ImportFrom):
            module = (body.module or '').lstrip('.')
            if body.level > 1:
                module = '.' * ((body.level - 1) if not module else body.level) + module
            for name in body.names:
                validate_name(path, '%s.%s' % (module, name.name), restrict_backend)
    with open(path, 'rb') as f:
        chkbody(ast.parse(f.read()))


def check_dir(path, restrict_backend):
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() == '.py':
                check_file(os.path.join(parent, filename), restrict_backend)


source_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../ipwxlearn'))
check_dir(os.path.join(source_dir, 'glue/common'), ['common'])
for backend in BACKEND_NAMES:
    check_dir(os.path.join(source_dir, 'glue/%s' % backend), ['common', backend])
