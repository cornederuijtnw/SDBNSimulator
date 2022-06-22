from setuptools import setup

setup(
    name='sdbnsimulator',
    version='0.1',
    packages=['allcode', 'allcode.clickmodel_fitters', 'allcode.simulator', 'allcode.util'],
    url='https://github.com/cornederuijtnw/SDBNSimulator',
    license='GNU General Public License',
    author='Corne de Ruijt',
    author_email='c.a.m.de.ruijt@vu.nl',
    install_requires=['deprecated', 'pandas==1.0.4', 'numpy==1.22.0', 'scikit-learn==0.22.1'],
    description='Simulator of SDBN click models'
)