from setuptools import setup, find_packages

try:
    import dgm4nlp
except ImportError:
    raise ImportError("First you need to clone and install dgm4nlp")

ext_modules = []

setup(
    name='embedalign',
    license='MIT',
    author='Miguel Rios, Wilker Aziz and friends :D',
    description='A deep generative model of word representation by marginalisation of lexical alignments',
    packages=find_packages(),
    install_requirements=['tabulate', 'dill'],
    include_dirs=[],
    ext_modules=ext_modules,
)
