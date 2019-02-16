from setuptools import setup, find_packages

requires = [
    'Pillow',
    'torchvision',
]

setup(
    name='pyclassify',
    version='0.1',
    description='PyClassify for transfer learning',
    author='Zeyad Yasser',
    author_email='zeyady98@gmail.com',
    url='https://github.com/zeyadyasser',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
    entry_points={
        'console_scripts': [
            'pyclassify_train = pyclassify.command:train_cmd',
            'pyclassify_run = pyclassify.command:run_cmd',
        ],
    },
)
