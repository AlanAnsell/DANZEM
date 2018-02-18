from setuptools import setup

setup(name='DANZEM',
      version='0.2.6',
      description='Data Analysis tools for the New Zealand Electricity Market',
      url='https://github.com/AlanAnsell/DANZEM',
      author='Alan Ansell',
      install_requires=[
          'numpy',
          'pandas',
          'pytz'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      packages=['DANZEM'],
      include_package_data=True,
      zip_safe=False)

