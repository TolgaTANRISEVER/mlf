from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='mlf',
    version='1.0.0',
    description='prepared for basic EDA and machinelearning functions operations',
    long_description_content_type="text/markdown",
    long_description=open('README.txt').read(),
    url='https://github.com/TolgaTANRISEVER/MLF',
    author='Tolga TANRISEVER',
    author_email='tanrisevertolga@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='MLF',
    packages=find_packages(include=['mlf']),
    install_requires=['pandas', 'scikit-learn', 'numpy', 'plotly'],
)