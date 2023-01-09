from distutils.core import setup

setup(
    name='tabular_rl',
    packages=['tabular_rl', 'tabular_rl.agents', 'tabular_rl.envs', 'tabular_rl.core'],
    version='0.1.0',
    license='MIT',
    description='This repository aims to contain implementations of the main tabular methods in Reinforcement '
                'Learning. The goal is to provide a framework that allows to easily compare different methods '
                'and to easily implement new ones.',
    author='Pablo',
    author_email='pablete.arino@gmail.com',
    url='https://github.com/Pabloo22/tabular_rl',
    download_url='https://github.com/Pabloo22/tabular_rl/archive/refs/tags/v0.1.0.tar.gz',
    keywords=['Reinforcement Learning', 'Machine Learning'],
    install_requires=[
        'numpy >= 1.18.0',
        'tqdm >= 4.41.0',
        'scipy >= 1.4.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)